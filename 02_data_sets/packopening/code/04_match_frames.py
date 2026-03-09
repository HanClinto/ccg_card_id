#!/usr/bin/env python3
"""Match extracted frames to reference cards using SIFT homography.

For each frame in datasets/packopening/frames/{slug}/:
  1. Extract SIFT features from the frame.
  2. For each reference card in the video's set(s), run FLANN matching +
     Lowe's ratio test against the pre-loaded descriptors.
  3. If >= MIN_MATCHES good matches, attempt homography.
  4. Homography accepted if all 4 projected corners fall within frame bounds.
  5. Keep the card with the most inliers; discard the frame if none qualify.

Matched frames are:
  - Dewarped and saved to datasets/packopening/aligned/{slug}/{card_id}_{frame_pos}.jpg
  - Recorded in the frames table of the SQLite DB

Conservative quality filters ensure high precision over recall. Any frame that
doesn't produce a clean, unambiguous 4-corner match is discarded.

Usage (run from project root):
    python 02_data_sets/packopening/code/04_match_frames.py --slug dQw4w9WgXcQ_lea_alpha-booster
    python 02_data_sets/packopening/code/04_match_frames.py --all

Requires:
    pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
import importlib.util as _ilu

from db import open_db, set_video_status, get_videos_by_status

# Import from 03_precompute_sift.py (filename starts with a digit — must use importlib)
_spec = _ilu.spec_from_file_location(
    "precompute_sift",
    Path(__file__).parent / "03_precompute_sift.py",
)
_precompute = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_precompute)
load_sift_features = _precompute.load_sift_features

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
MIN_MATCHES = 20        # Minimum SIFT inliers to attempt homography
MARGIN = 5              # Best card must beat 2nd-best by this many matches
MIN_AREA_PCT = 0.05     # Card must cover at least 5% of frame area
MAX_AREA_PCT = 0.95     # Card cannot cover more than 95% (degenerate homography)
LOWE_RATIO = 0.75       # Lowe's ratio test threshold

# Reference card size (Scryfall PNGs are 745×1040)
REF_W, REF_H = 745, 1040


def _kp_array_to_kps(kp_array: np.ndarray) -> list[cv2.KeyPoint]:
    kps = []
    for row in kp_array:
        kp = cv2.KeyPoint(
            x=float(row[0]), y=float(row[1]),
            size=float(row[2]), angle=float(row[3]),
            response=float(row[4]), octave=int(row[5]),
            class_id=int(row[6]),
        )
        kps.append(kp)
    return kps


def build_flann() -> cv2.FlannBasedMatcher:
    index_params = {"algorithm": 1, "trees": 5}   # FLANN_INDEX_KDTREE
    search_params = {"checks": 50}
    return cv2.FlannBasedMatcher(index_params, search_params)


def match_frame_to_card(
    frame_descs: np.ndarray,
    ref_descs: np.ndarray,
    flann: cv2.FlannBasedMatcher,
) -> int:
    """Return number of good SIFT matches (Lowe's ratio test)."""
    if frame_descs is None or ref_descs is None:
        return 0
    if len(ref_descs) < 2 or len(frame_descs) < 2:
        return 0
    try:
        matches = flann.knnMatch(frame_descs, ref_descs, k=2)
    except Exception:
        return 0
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    return len(good)


def find_homography_corners(
    frame_kps: list[cv2.KeyPoint],
    ref_kps: list[cv2.KeyPoint],
    frame_descs: np.ndarray,
    ref_descs: np.ndarray,
    frame_h: int,
    frame_w: int,
    flann: cv2.FlannBasedMatcher,
) -> tuple[np.ndarray | None, int]:
    """Compute homography and return (scene_corners 4×1×2, inlier_count) or (None, 0)."""
    if len(frame_descs) < 4 or len(ref_descs) < 4:
        return None, 0

    try:
        matches = flann.knnMatch(frame_descs, ref_descs, k=2)
    except Exception:
        return None, 0

    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0

    src_pts = np.float32([ref_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([frame_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None, 0

    inliers = int(mask.sum()) if mask is not None else 0

    # Project reference card corners into frame space
    ref_corners = np.float32([[0, 0], [REF_W, 0], [REF_W, REF_H], [0, REF_H]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(ref_corners, M)

    # All 4 corners must be within frame bounds (with small margin)
    margin = 5
    for pt in scene_corners:
        x, y = pt[0]
        if x < -margin or x > frame_w + margin or y < -margin or y > frame_h + margin:
            return None, 0

    # Area check
    xs = [pt[0][0] for pt in scene_corners]
    ys = [pt[0][1] for pt in scene_corners]
    area_pct = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / (frame_w * frame_h)
    if not (MIN_AREA_PCT <= area_pct <= MAX_AREA_PCT):
        return None, 0

    return scene_corners, inliers


def dewarp_card(frame: np.ndarray, scene_corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the matched card region to a canonical REF_W×REF_H image."""
    dst_pts = np.float32([[0, 0], [REF_W, 0], [REF_W, REF_H], [0, REF_H]])
    src_pts = scene_corners.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (REF_W, REF_H))


def load_all_sift_for_sets(cache_root: Path, set_codes: list[str]) -> list[dict]:
    """Load all .npz SIFT caches for the given sets. Returns list of card dicts."""
    cards = []
    for sc in set_codes:
        sc_dir = cache_root / sc
        if not sc_dir.exists():
            print(f"  WARNING: SIFT cache missing for set '{sc}' — run 03_precompute_sift.py first")
            continue
        for npz_path in sorted(sc_dir.glob("*.npz")):
            card_id = npz_path.stem
            kp_array, descs = load_sift_features(npz_path)
            # Store as rebuilt KeyPoint list + descriptors
            kps = _kp_array_to_kps(kp_array)
            cards.append({
                "card_id": card_id,
                "set_code": sc,
                "kps": kps,
                "descs": descs,
            })
    return cards


def process_video(video: dict, data_dir: Path, con, rebuild: bool = False) -> tuple[int, int]:
    """Match all frames for a video. Returns (matched, discarded)."""
    slug = video["slug"]
    video_id = video["video_id"]
    set_codes = [s.strip().lower() for s in video["set_codes"].split(",") if s.strip()]

    if not set_codes:
        print(f"  ERROR: no set_codes for {slug} — cannot match")
        return 0, 0

    frames_dir = data_dir / "datasets" / "packopening" / "frames" / slug
    aligned_dir = data_dir / "datasets" / "packopening" / "aligned" / slug
    cache_root = data_dir / "datasets" / "packopening" / "sift_cache"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"  No frames found in {frames_dir}")
        return 0, 0

    # Load existing DB frame paths so we can skip already-matched frames
    already_done: set[str] = set()
    if not rebuild:
        existing = con.execute(
            "SELECT frame_path FROM frames WHERE video_id=?", (video_id,)
        ).fetchall()
        already_done = {row["frame_path"] for row in existing}

    # Load SIFT gallery for this video's sets
    print(f"  Loading SIFT gallery for sets {set_codes}...")
    gallery = load_all_sift_for_sets(cache_root, set_codes)
    if not gallery:
        print("  ERROR: empty gallery — aborting")
        return 0, 0
    print(f"  Gallery: {len(gallery)} reference cards")

    flann = build_flann()
    sift = cv2.SIFT_create()

    matched = discarded = skipped = 0
    for frame_path in frame_paths:
        # Use cfg.data_dir-relative path as key
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

        # Quick pre-filter: count Lowe matches against each card (cheap, no homography)
        match_counts: list[tuple[int, int]] = []  # (count, gallery_idx)
        for gi, card in enumerate(gallery):
            n = match_frame_to_card(frame_descs, card["descs"], flann)
            match_counts.append((n, gi))

        match_counts.sort(reverse=True)
        best_count, best_gi = match_counts[0]
        second_count = match_counts[1][0] if len(match_counts) > 1 else 0

        if best_count < MIN_MATCHES:
            discarded += 1
            continue
        if best_count - second_count < MARGIN:
            discarded += 1
            continue

        # Full homography on best candidate only
        best_card = gallery[best_gi]
        scene_corners, inliers = find_homography_corners(
            frame_kps, best_card["kps"],
            frame_descs, best_card["descs"],
            frame_h, frame_w, flann,
        )
        if scene_corners is None:
            discarded += 1
            continue

        # Dewarp and save aligned image
        aligned = dewarp_card(frame, scene_corners)
        frame_pos = frame_path.stem.split("_")[-1]  # e.g. "000042" from "frame_000042"
        aligned_name = f"{best_card['card_id']}_{frame_pos}.jpg"
        aligned_path = aligned_dir / aligned_name
        cv2.imwrite(str(aligned_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Compute corner coordinates normalised to [0,1]
        corners = [(float(scene_corners[i, 0, 0]) / frame_w, float(scene_corners[i, 0, 1]) / frame_h)
                   for i in range(4)]
        # Rotate so top-left (min x+y) is first
        idx = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
        corners = corners[idx:] + corners[:idx]

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        area_pct = (max(xs) - min(xs)) * (max(ys) - min(ys))

        con.execute(
            """INSERT OR IGNORE INTO frames
               (video_id, frame_path, aligned_path, card_id, set_code,
                num_matches, corner0_x, corner0_y, corner1_x, corner1_y,
                corner2_x, corner2_y, corner3_x, corner3_y, matching_area_pct)
               VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?)""",
            (
                video_id,
                rel_path,
                str(aligned_path.relative_to(data_dir)),
                best_card["card_id"],
                best_card["set_code"],
                inliers,
                corners[0][0], corners[0][1],
                corners[1][0], corners[1][1],
                corners[2][0], corners[2][1],
                corners[3][0], corners[3][1],
                area_pct,
            ),
        )
        con.commit()
        matched += 1

    total = len(frame_paths)
    print(f"  matched={matched}  discarded={discarded}  skipped={skipped}  total={total}")
    return matched, discarded


def main() -> None:
    p = argparse.ArgumentParser(description="SIFT-match extracted frames to reference cards")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug", help="Video slug")
    group.add_argument("--all", action="store_true", help="Process all 'frames_extracted' videos")
    p.add_argument("--rebuild", action="store_true", help="Re-match even if frames already in DB")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    if args.all:
        videos = get_videos_by_status(con, "frames_extracted")
        if not videos:
            print("No videos with status 'frames_extracted' found.")
            return
        print(f"Matching {len(videos)} video(s)...")
    else:
        v = con.execute("SELECT * FROM videos WHERE slug=?", (args.slug,)).fetchone()
        if not v:
            print(f"ERROR: slug '{args.slug}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]

    total_matched = total_discarded = 0
    for video in videos:
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        set_video_status(con, video["video_id"], "processing")
        try:
            m, d = process_video(dict(video), args.data_dir, con, rebuild=args.rebuild)
            set_video_status(con, video["video_id"], "done")
            con.execute(
                "UPDATE videos SET processed_date=date('now') WHERE video_id=?",
                (video["video_id"],),
            )
            con.commit()
            total_matched += m
            total_discarded += d
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            set_video_status(con, video["video_id"], "error")

    print(f"\nTotal matched: {total_matched}  Total discarded: {total_discarded}")
    if total_matched:
        print("Next: python 02_data_sets/packopening/code/05_build_manifest.py")


if __name__ == "__main__":
    main()
