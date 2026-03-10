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
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
    _PHASH_AVAILABLE = True
except ImportError:
    _PHASH_AVAILABLE = False
    print("WARNING: imagehash not installed — pHash verification disabled. "
          "Run: pip install imagehash pillow", file=__import__("sys").stderr)

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db, set_video_status, get_videos_by_status

# Load helpers from sibling scripts (filenames start with digits)
_spec = _ilu.spec_from_file_location(
    "precompute_sift", Path(__file__).parent / "02_precompute_sift.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_sift_features = _mod.load_sift_features

_spec2 = _ilu.spec_from_file_location(
    "extract_frames", Path(__file__).parent / "03_extract_frames.py"
)
_mod2 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
extract_dense = _mod2.extract_dense
get_video_timebase = _mod2.get_video_timebase
pts_to_seconds = _mod2.pts_to_seconds
LEAD_IN = _mod2.LEAD_IN

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_MATCHES = 20
MARGIN = 5
MIN_AREA_PCT = 0.05
MAX_AREA_PCT = 0.95
LOWE_RATIO = 0.75
REF_W, REF_H = 745, 1040
MIN_QUAD_ANGLE = 15.0   # degrees; quads with sharper corners are near-edge-on / degenerate
MAX_PHASH_DIST = 50 # 20     # Hamming distance on 64-bit pHash; lower = stricter


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


def build_global_flann(gallery: list[dict]) -> tuple[cv2.FlannBasedMatcher | None, np.ndarray | None]:
    """Concatenate all gallery descriptors into one FLANN index.

    Returns (flann, desc_to_card) where desc_to_card[i] maps the i-th row of
    the concatenated matrix back to a gallery index, or (None, None) on failure.
    """
    descs_list: list[np.ndarray] = []
    mapping: list[int] = []
    for i, card in enumerate(gallery):
        d = card["descs"]
        if d is None or len(d) == 0:
            continue
        descs_list.append(d)
        mapping.extend([i] * len(d))

    if not descs_list:
        return None, None

    all_descs = np.vstack(descs_list).astype(np.float32)
    desc_to_card = np.array(mapping, dtype=np.int32)

    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    flann.add([all_descs])
    flann.train()
    return flann, desc_to_card


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


def is_valid_quad(scene_corners) -> bool:
    """Return False if the source quad is non-convex or has a near-degenerate angle."""
    pts = scene_corners.reshape(4, 2).astype(np.float32)
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.int32)):
        return False
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return False
        cos_a = np.dot(v1, v2) / denom
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if angle < MIN_QUAD_ANGLE:
            return False
    return True


def compute_phash(img_bgr):
    """Compute a pHash for an OpenCV BGR image. Returns None if imagehash unavailable."""
    if not _PHASH_AVAILABLE:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def dewarp(frame, scene_corners):
    dst = np.float32([[0,0],[REF_W,0],[REF_W,REF_H],[0,REF_H]])
    src = scene_corners.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (REF_W, REF_H))


def _cache_dir_name(set_code: str, lang: str) -> str:
    lang = lang.strip().lower() or "en"
    return set_code if lang == "en" else f"{set_code}_{lang}"


def load_gallery(cache_root: Path, set_codes: list[str], lang: str = "en") -> list[dict]:
    all_npz = []
    for sc in set_codes:
        dir_name = _cache_dir_name(sc, lang)
        sc_dir = cache_root / dir_name
        if not sc_dir.exists():
            print(f"  WARNING: SIFT cache missing for '{dir_name}' — run 02_precompute_sift.py first")
            continue
        all_npz.extend((sc, p) for p in sorted(sc_dir.glob("*.npz")))

    gallery = []
    with tqdm(all_npz, unit="card", desc="  loading gallery", leave=False) as pbar:
        for sc, npz_path in pbar:
            card_id = npz_path.stem
            kp_array, descs = load_sift_features(npz_path)
            ref_phash = None
            if _PHASH_AVAILABLE:
                ref_img_path = (cfg.scryfall_images_dir / "front"
                                / card_id[0] / card_id[1] / f"{card_id}.png")
                ref_img = cv2.imread(str(ref_img_path))
                if ref_img is not None:
                    ref_phash = compute_phash(ref_img)
            gallery.append({
                "card_id": card_id,
                "set_code": sc,
                "kps": _kp_array_to_kps(kp_array),
                "descs": descs,
                "ref_phash": ref_phash,
            })
    return gallery


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _match_batch(
    frame_paths: list[Path],
    gallery: list[dict],
    global_flann,
    desc_to_card: np.ndarray,
    homography_flann,
    sift,
    already_done: set[str],
    aligned_dir: Path,
    video_id: str,
    data_dir: Path,
    con,
    desc: str = "  matching",
) -> tuple[int, int, int, str | None]:
    """Run the SIFT matching loop over frame_paths.

    Returns (matched, discarded, skipped, first_match_rel_path).
    first_match_rel_path is the rel_path of the first accepted frame, or None.
    """
    matched = discarded = skipped = 0
    first_match_rel: str | None = None

    with tqdm(frame_paths, unit="frame", desc=desc) as pbar:
        for frame_path in pbar:
            pbar.set_postfix(matched=matched, discarded=discarded, skipped=skipped)
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

            # Quick pre-filter: single knnMatch against the global index
            try:
                raw_matches = global_flann.knnMatch(frame_descs.astype(np.float32), k=2)
            except Exception:
                discarded += 1
                continue

            card_counts = np.zeros(len(gallery), dtype=np.int32)
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < LOWE_RATIO * n.distance:
                    card_counts[desc_to_card[m.trainIdx]] += 1

            best_i = int(card_counts.argmax())
            best_n = int(card_counts[best_i])
            second_n = int(np.partition(card_counts, -2)[-2]) if len(gallery) > 1 else 0

            if best_n < MIN_MATCHES or (best_n - second_n) < MARGIN:
                discarded += 1
                continue

            # Full homography on best candidate
            best = gallery[best_i]
            scene_corners, inliers = homography_corners(
                frame_kps, best["kps"], frame_descs, best["descs"], frame_h, frame_w, homography_flann
            )
            if scene_corners is None:
                discarded += 1
                continue

            # Reject degenerate quads (non-convex or near-edge-on)
            if not is_valid_quad(scene_corners):
                discarded += 1
                continue

            # Dewarp and verify with pHash before saving
            dewarped = dewarp(frame, scene_corners)
            if best["ref_phash"] is not None:
                match_phash = compute_phash(dewarped)
                if (not _PHASH_AVAILABLE) or ((match_phash - best["ref_phash"]) > MAX_PHASH_DIST):
                    discarded += 1
                    continue

            # Save aligned crop
            frame_pos = frame_path.stem.split("_")[-1]
            aligned_path = aligned_dir / f"{best['card_id']}_{frame_pos}.jpg"
            cv2.imwrite(str(aligned_path), dewarped, [cv2.IMWRITE_JPEG_QUALITY, 92])

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
            if first_match_rel is None:
                first_match_rel = rel_path
            matched += 1

    return matched, discarded, skipped, first_match_rel


def process_video(
    video: dict,
    data_dir: Path,
    con,
    rebuild: bool = False,
    densify: bool = True,
) -> tuple[int, int]:
    slug = video["slug"]
    video_id = video["video_id"]
    set_codes = [s for s in re.split(r"[\s,]+", video["set_codes"].strip().lower()) if s]
    lang = (video.get("lang") or "en").strip().lower() or "en"
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

    lang_label = f" lang={lang}" if lang != "en" else ""
    print(f"  Loading SIFT gallery for {set_codes}{lang_label}...")
    gallery = load_gallery(cache_root, set_codes, lang=lang)
    if not gallery:
        print("  ERROR: empty gallery")
        return 0, 0
    print(f"  Gallery: {len(gallery)} cards")

    print("  Building global FLANN index...", end=" ", flush=True)
    global_flann, desc_to_card = build_global_flann(gallery)
    if global_flann is None:
        print("ERROR: could not build global FLANN index")
        return 0, 0
    print("done")

    homography_flann = build_flann()
    sift = cv2.SIFT_create()

    # --- Pass 1: match scene-change frames ---
    m1, d1, s1, first_match_rel = _match_batch(
        frame_paths, gallery, global_flann, desc_to_card,
        homography_flann, sift, already_done, aligned_dir,
        video_id, data_dir, con, desc="  pass 1",
    )
    print(f"  pass 1: matched={m1}  discarded={d1}  skipped={s1}  total={len(frame_paths)}")

    # --- Optional densification between passes ---
    m2 = d2 = s2 = 0
    already_densified = bool(video.get("densified"))
    if densify and not already_densified and first_match_rel is not None:
        mp4_path = data_dir / "datasets" / "packopening" / "raw" / slug / f"{slug}.mp4"
        if not mp4_path.exists():
            print(f"  WARNING: mp4 not found, skipping densification: {mp4_path}")
        else:
            try:
                timebase = get_video_timebase(mp4_path)
                first_ts = pts_to_seconds(first_match_rel, timebase)
                dense_from = max(0.0, first_ts - LEAD_IN)
                print(f"  → first card at {first_ts:.1f}s; dense pass from {dense_from:.1f}s")
                before = set(frames_dir.glob("frame_*.jpg"))
                extract_dense(mp4_path, frames_dir, from_seconds=dense_from)
                new_paths = sorted(set(frames_dir.glob("frame_*.jpg")) - before)
                print(f"  → {len(new_paths)} new frames extracted")
                con.execute("UPDATE videos SET densified=1 WHERE video_id=?", (video_id,))
                con.commit()
                # --- Pass 2: match the new dense frames only ---
                if new_paths:
                    m2, d2, s2, _ = _match_batch(
                        new_paths, gallery, global_flann, desc_to_card,
                        homography_flann, sift, already_done, aligned_dir,
                        video_id, data_dir, con, desc="  pass 2",
                    )
                    print(f"  pass 2: matched={m2}  discarded={d2}  skipped={s2}  total={len(new_paths)}")
            except Exception as e:
                print(f"  WARNING: densification failed: {e}", file=sys.stderr)
    elif already_densified:
        print("  (already densified — skipping dense pass)")

    total_matched = m1 + m2
    total_discarded = d1 + d2
    return total_matched, total_discarded


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
    p.add_argument("--no-densify", action="store_true",
                   help="Skip automatic dense frame extraction after first match")
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
            m, _ = process_video(dict(video), args.data_dir, con,
                                 rebuild=args.rebuild, densify=not args.no_densify)
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
