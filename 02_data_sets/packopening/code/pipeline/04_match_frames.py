#!/usr/bin/env python3
"""Match extracted frames to reference cards using SIFT homography.

For each frame:
  1. Extract SIFT features.
  2. Quick match count against all set cards (Lowe's ratio test, no homography yet).
  3. Best card must beat 2nd-best by MARGIN matches (avoids ambiguous frames).
  4. Full homography on best candidate: all 4 corners must fall within the frame.
  5. Accepted frames are dewarped and written to aligned/{slug}/{card_id}_{pos}.jpg,
     with corner coordinates stored in the DB.

Quality thresholds are conservative: precision over recall.

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/04_match_frames.py --slug <slug>
    python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db, set_video_status, get_videos_by_status

# Load load_sift_features from sibling script (filename starts with digit)
_spec = _ilu.spec_from_file_location(
    "precompute_sift", Path(__file__).parent / "03_precompute_sift.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_sift_features = _mod.load_sift_features

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_MATCHES = 20
MARGIN = 5
MIN_AREA_PCT = 0.05
MAX_AREA_PCT = 0.95
LOWE_RATIO = 0.75
REF_W, REF_H = 745, 1040


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kp_array_to_kps(kp_array: np.ndarray) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(x=float(r[0]), y=float(r[1]), size=float(r[2]),
                     angle=float(r[3]), response=float(r[4]),
                     octave=int(r[5]), class_id=int(r[6]))
        for r in kp_array
    ]


def build_flann() -> cv2.FlannBasedMatcher:
    return cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})


def count_lowe_matches(frame_descs, ref_descs, flann) -> int:
    if frame_descs is None or ref_descs is None:
        return 0
    if len(ref_descs) < 2 or len(frame_descs) < 2:
        return 0
    try:
        matches = flann.knnMatch(frame_descs, ref_descs, k=2)
    except Exception:
        return 0
    return sum(1 for m, n in matches if m.distance < LOWE_RATIO * n.distance)


def homography_corners(frame_kps, ref_kps, frame_descs, ref_descs,
                       frame_h, frame_w, flann):
    """Returns (scene_corners 4×1×2, inlier_count) or (None, 0)."""
    try:
        matches = flann.knnMatch(frame_descs, ref_descs, k=2)
    except Exception:
        return None, 0
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0

    src = np.float32([ref_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([frame_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if M is None:
        return None, 0

    ref_corners = np.float32([[0,0],[REF_W,0],[REF_W,REF_H],[0,REF_H]]).reshape(-1,1,2)
    scene_corners = cv2.perspectiveTransform(ref_corners, M)

    margin = 5
    for pt in scene_corners:
        x, y = pt[0]
        if x < -margin or x > frame_w + margin or y < -margin or y > frame_h + margin:
            return None, 0

    xs = [pt[0][0] for pt in scene_corners]
    ys = [pt[0][1] for pt in scene_corners]
    area_pct = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / (frame_w * frame_h)
    if not (MIN_AREA_PCT <= area_pct <= MAX_AREA_PCT):
        return None, 0

    inliers = int(mask.sum()) if mask is not None else 0
    return scene_corners, inliers


def dewarp(frame, scene_corners):
    dst = np.float32([[0,0],[REF_W,0],[REF_W,REF_H],[0,REF_H]])
    src = scene_corners.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (REF_W, REF_H))


def load_gallery(cache_root: Path, set_codes: list[str]) -> list[dict]:
    gallery = []
    for sc in set_codes:
        sc_dir = cache_root / sc
        if not sc_dir.exists():
            print(f"  WARNING: SIFT cache missing for '{sc}' — run 03_precompute_sift.py first")
            continue
        for npz_path in sorted(sc_dir.glob("*.npz")):
            kp_array, descs = load_sift_features(npz_path)
            gallery.append({
                "card_id": npz_path.stem,
                "set_code": sc,
                "kps": _kp_array_to_kps(kp_array),
                "descs": descs,
            })
    return gallery


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video: dict, data_dir: Path, con, rebuild: bool = False) -> tuple[int, int]:
    slug = video["slug"]
    video_id = video["video_id"]
    set_codes = [s for s in re.split(r"[\s,]+", video["set_codes"].strip().lower()) if s]
    if not set_codes:
        print(f"  ERROR: no set_codes for {slug}")
        return 0, 0

    frames_dir = data_dir / "datasets" / "packopening" / "frames" / slug
    aligned_dir = data_dir / "datasets" / "packopening" / "aligned" / slug
    cache_root = data_dir / "datasets" / "packopening" / "sift_cache"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"  No frames found in {frames_dir}")
        return 0, 0

    already_done: set[str] = set()
    if not rebuild:
        rows = con.execute("SELECT frame_path FROM frames WHERE video_id=?", (video_id,)).fetchall()
        already_done = {r["frame_path"] for r in rows}

    print(f"  Loading SIFT gallery for {set_codes}...")
    gallery = load_gallery(cache_root, set_codes)
    if not gallery:
        print("  ERROR: empty gallery")
        return 0, 0
    print(f"  Gallery: {len(gallery)} cards")

    flann = build_flann()
    sift = cv2.SIFT_create()
    matched = discarded = skipped = 0

    for frame_path in frame_paths:
        rel_path = str(frame_path.relative_to(data_dir))
        if rel_path in already_done:
            skipped += 1
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            discarded += 1
            continue

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kps, frame_descs = sift.detectAndCompute(gray, None)
        if frame_kps is None or frame_descs is None or len(frame_kps) < MIN_MATCHES:
            discarded += 1
            continue

        # Quick pre-filter
        counts = sorted(
            [(count_lowe_matches(frame_descs, c["descs"], flann), i) for i, c in enumerate(gallery)],
            reverse=True,
        )
        best_n, best_i = counts[0]
        second_n = counts[1][0] if len(counts) > 1 else 0

        if best_n < MIN_MATCHES or (best_n - second_n) < MARGIN:
            discarded += 1
            continue

        # Full homography on best candidate
        best = gallery[best_i]
        scene_corners, inliers = homography_corners(
            frame_kps, best["kps"], frame_descs, best["descs"], frame_h, frame_w, flann
        )
        if scene_corners is None:
            discarded += 1
            continue

        # Save aligned crop
        frame_pos = frame_path.stem.split("_")[-1]
        aligned_path = aligned_dir / f"{best['card_id']}_{frame_pos}.jpg"
        cv2.imwrite(str(aligned_path), dewarp(frame, scene_corners), [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Normalised corners, top-left first
        corners = [(float(scene_corners[i,0,0]) / frame_w, float(scene_corners[i,0,1]) / frame_h)
                   for i in range(4)]
        idx = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
        corners = corners[idx:] + corners[:idx]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        area_pct = (max(xs) - min(xs)) * (max(ys) - min(ys))

        con.execute(
            """INSERT OR IGNORE INTO frames
               (video_id, frame_path, aligned_path, card_id, set_code,
                num_matches, corner0_x, corner0_y, corner1_x, corner1_y,
                corner2_x, corner2_y, corner3_x, corner3_y, matching_area_pct)
               VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)""",
            (video_id, rel_path, str(aligned_path.relative_to(data_dir)),
             best["card_id"], best["set_code"], inliers,
             corners[0][0], corners[0][1], corners[1][0], corners[1][1],
             corners[2][0], corners[2][1], corners[3][0], corners[3][1], area_pct),
        )
        con.commit()
        matched += 1

    print(f"  matched={matched}  discarded={discarded}  skipped={skipped}  total={len(frame_paths)}")
    return matched, discarded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="SIFT-match frames to reference cards")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true", help="Process all 'frames_extracted' videos")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    if args.all:
        videos = get_videos_by_status(con, "frames_extracted")
        if not videos:
            print("No videos with status 'frames_extracted'.")
            return
    elif args.slug:
        v = con.execute("SELECT * FROM videos WHERE slug=?", (args.slug,)).fetchone()
        if not v:
            print(f"ERROR: slug '{args.slug}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]
    else:
        v = con.execute("SELECT * FROM videos WHERE video_id=?", (args.video_id,)).fetchone()
        if not v:
            print(f"ERROR: video_id '{args.video_id}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]

    total_matched = 0
    for video in videos:
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        set_video_status(con, video["video_id"], "processing")
        try:
            m, _ = process_video(dict(video), args.data_dir, con, rebuild=args.rebuild)
            set_video_status(con, video["video_id"], "done")
            con.execute("UPDATE videos SET processed_date=date('now') WHERE video_id=?",
                        (video["video_id"],))
            con.commit()
            total_matched += m
        except Exception as e:
            import traceback; traceback.print_exc()
            set_video_status(con, video["video_id"], "error")

    print(f"\nTotal matched frames: {total_matched}")
    if total_matched:
        print("Next: python 02_data_sets/packopening/code/pipeline/05_build_manifest.py")


if __name__ == "__main__":
    main()
