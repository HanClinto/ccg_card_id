#!/usr/bin/env python3
"""Match extracted frames to Pokemon TCG reference cards using SIFT homography.

Pokemon fork of 04_match_frames.py.  Key differences:
  - Reads SIFT cache from sift_cache/pokemon/{set_id}/{card_id}.npz
  - Uses cfg.pokemontcg_images_dir for pHash reference images
  - Uses pokemon_catalog for illustration_id lookup
  - Filters the video queue by game='pokemon'

For each frame:
  1. Extract SIFT features.
  2. Quick match count against all set cards (Lowe's ratio test, no homography yet).
  3. Best card must beat 2nd-best by MARGIN matches (avoids ambiguous frames).
  4. Full homography on best candidate: all 4 corners must fall within the frame.
  5. Accepted frames are dewarped and written to aligned/{slug}/{card_id}_{pos}.jpg,
     with corner coordinates stored in the DB.

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/04_match_frames_pokemon.py --slug <slug>
    python 02_data_sets/packopening/code/pipeline/04_match_frames_pokemon.py --all

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
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
from ccg_card_id.pokemon_catalog import pokemon_catalog
from db import open_db, set_video_status, claim_next_video

# ---------------------------------------------------------------------------
# Thresholds (same as MTG matcher)
# ---------------------------------------------------------------------------
MIN_MATCHES = 20
MARGIN = 5
MIN_AREA_PCT = 0.05
MAX_AREA_PCT = 0.95
LOWE_RATIO = 0.75
# NOTE: no global REF_W/REF_H — Pokemon catalog images vary in size (734×1024
# for most sets, different for full-art / special-illustration rares, etc.).
# Each gallery entry stores ref_w / ref_h read from the actual catalog image.
MIN_QUAD_ANGLE = 15.0


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


def _kps_to_array(kps: list[cv2.KeyPoint]) -> np.ndarray:
    if not kps:
        return np.empty((0, 7), dtype=np.float32)
    return np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id]
         for k in kps],
        dtype=np.float32,
    )


def build_flann() -> cv2.FlannBasedMatcher:
    return cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})


def build_global_flann(all_descs: np.ndarray) -> cv2.FlannBasedMatcher:
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    flann.add([all_descs])
    flann.train()
    return flann


def load_sift_features(npz_path: Path):
    data = np.load(str(npz_path))
    return data["keypoints"], data["descriptors"]


def _pokemon_image_path(card_id: str) -> Path:
    a = card_id[0] if card_id else "x"
    b = card_id[1] if len(card_id) > 1 else "x"
    return cfg.pokemontcg_images_dir / "large" / a / b / f"{card_id}.png"


def homography_corners(frame_kps, ref_kps, frame_descs, ref_descs,
                       frame_h, frame_w, flann, ref_w: int, ref_h: int):
    """Returns (scene_corners 4×1×2, inlier_count) or (None, 0).

    ref_w / ref_h are the actual pixel dimensions of the reference image so
    that the projected corners exactly cover the card and nothing more.
    """
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

    ref_corners = np.float32([[0,0],[ref_w,0],[ref_w,ref_h],[0,ref_h]]).reshape(-1,1,2)
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
    if not _PHASH_AVAILABLE:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def dewarp(frame, scene_corners, ref_w: int, ref_h: int):
    dst = np.float32([[0,0],[ref_w,0],[ref_w,ref_h],[0,ref_h]])
    src = scene_corners.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (ref_w, ref_h))


# ---------------------------------------------------------------------------
# Gallery loading (from sift_cache/pokemon/{set_id}/)
# ---------------------------------------------------------------------------

def load_gallery(
    cache_root: Path,
    set_ids: list[str],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load SIFT features for all cards in the given Pokemon set IDs.

    Returns (gallery, all_descs, desc_to_card).
    """
    all_npz: list[tuple[str, Path]] = []
    for sid in set_ids:
        sc_dir = cache_root / "pokemon" / sid
        if not sc_dir.exists():
            print(f"  WARNING: SIFT cache missing for Pokemon set '{sid}' — "
                  f"run 02_precompute_sift_pokemon.py first")
            continue
        all_npz.extend((sid, p) for p in sorted(sc_dir.glob("*.npz")))

    gallery: list[dict] = []
    with tqdm(all_npz, unit="card", desc="  loading gallery", leave=False) as pbar:
        for sid, npz_path in pbar:
            card_id = npz_path.stem
            kp_array, descs = load_sift_features(npz_path)
            ref_img_path = _pokemon_image_path(card_id)
            # Read actual reference image dimensions — Pokemon catalog images
            # vary in size (734×1024 standard, different for special-art cards).
            # Using per-card dimensions prevents a spurious border in the dewarp
            # output when REF_W/REF_H doesn't match the catalog image.
            ref_w, ref_h = 734, 1024  # safe fallback
            ref_img = cv2.imread(str(ref_img_path))
            if ref_img is not None:
                ref_h, ref_w = ref_img.shape[:2]
            ref_phash = None
            if _PHASH_AVAILABLE and ref_img is not None:
                ref_phash = compute_phash(ref_img)
            gallery.append({
                "card_id":   card_id,
                "set_code":  sid,
                "kps":       _kp_array_to_kps(kp_array),
                "descs":     descs,
                "ref_phash": ref_phash,
                "ref_w":     ref_w,
                "ref_h":     ref_h,
            })

    descs_list: list[np.ndarray] = []
    mapping: list[int] = []
    for i, card in enumerate(gallery):
        d = card["descs"]
        if d is None or len(d) == 0:
            continue
        descs_list.append(d)
        mapping.extend([i] * len(d))

    if not descs_list:
        return gallery, np.empty((0, 128), dtype=np.float32), np.empty(0, dtype=np.int32)

    all_descs    = np.vstack(descs_list).astype(np.float32)
    desc_to_card = np.array(mapping, dtype=np.int32)
    return gallery, all_descs, desc_to_card


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _build_illust_lookup(set_ids: list[str]) -> dict[str, str]:
    """Build {card_id: illustration_id} from the pokemon_catalog for the given sets."""
    cards = pokemon_catalog.cards_for_sets(set_ids)
    return {
        c["id"]: c.get("illustration_id", "")
        for c in cards
        if c.get("id") and c.get("illustration_id")
    }


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
    card_id_to_illust: dict[str, str],
) -> tuple[int, int, int]:
    matched = discarded = skipped = 0

    with tqdm(frame_paths, unit="frame", desc="  matching") as pbar:
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

            best = gallery[best_i]
            scene_corners, inliers = homography_corners(
                frame_kps, best["kps"], frame_descs, best["descs"], frame_h, frame_w,
                homography_flann, best["ref_w"], best["ref_h"],
            )
            if scene_corners is None:
                discarded += 1
                continue

            if not is_valid_quad(scene_corners):
                discarded += 1
                continue

            dewarped = dewarp(frame, scene_corners, best["ref_w"], best["ref_h"])
            phash_dist: int | None = None
            if _PHASH_AVAILABLE and best["ref_phash"] is not None:
                match_phash = compute_phash(dewarped)
                phash_dist = int(match_phash - best["ref_phash"])

            frame_pos = frame_path.stem.split("_")[-1]
            aligned_path = aligned_dir / f"{best['card_id']}_{frame_pos}.jpg"
            cv2.imwrite(str(aligned_path), dewarped, [cv2.IMWRITE_JPEG_QUALITY, 92])

            corners = [(float(scene_corners[i,0,0]) / frame_w, float(scene_corners[i,0,1]) / frame_h)
                       for i in range(4)]
            idx = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
            corners = corners[idx:] + corners[:idx]
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            area_pct = (max(xs) - min(xs)) * (max(ys) - min(ys))

            illust_id = card_id_to_illust.get(best["card_id"])
            con.execute(
                """INSERT OR IGNORE INTO frames
                   (video_id, frame_path, aligned_path, card_id, illustration_id, set_code,
                    num_matches, corner0_x, corner0_y, corner1_x, corner1_y,
                    corner2_x, corner2_y, corner3_x, corner3_y, matching_area_pct,
                    phash_dist)
                   VALUES (?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?)""",
                (video_id, rel_path, str(aligned_path.relative_to(data_dir)),
                 best["card_id"], illust_id, best["set_code"], inliers,
                 corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                 corners[2][0], corners[2][1], corners[3][0], corners[3][1], area_pct,
                 phash_dist),
            )
            con.commit()
            matched += 1

    return matched, discarded, skipped


def process_video(
    video: dict,
    data_dir: Path,
    con,
    rebuild: bool = False,
) -> tuple[int, int]:
    slug = video["slug"]
    video_id = video["video_id"]
    set_ids = [s for s in re.split(r"[\s,]+", video["set_codes"].strip().lower()) if s]
    if not set_ids:
        print(f"  ERROR: no set_codes for {slug}")
        return 0, 0

    frames_dir = data_dir / "datasets" / "packopening" / "frames" / slug
    aligned_dir = data_dir / "datasets" / "packopening" / "aligned" / slug
    cache_root  = data_dir / "datasets" / "packopening" / "sift_cache"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir} — re-run 03_extract_frames.py")

    already_done: set[str] = set()
    if not rebuild:
        rows = con.execute("SELECT frame_path FROM frames WHERE video_id=?", (video_id,)).fetchall()
        already_done = {r["frame_path"] for r in rows}
    else:
        con.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
        con.commit()

    print(f"  Loading SIFT gallery for {set_ids}...")
    gallery, all_descs, desc_to_card = load_gallery(cache_root, set_ids)
    if not gallery:
        print("  ERROR: empty gallery")
        return 0, 0
    if len(all_descs) == 0:
        print("  ERROR: gallery has no descriptors")
        return 0, 0

    print("  Building global FLANN index...", end=" ", flush=True)
    global_flann = build_global_flann(all_descs)
    print("done")

    card_id_to_illust = _build_illust_lookup(set_ids)
    print(f"  illustration_id lookup: {len(card_id_to_illust):,} cards")

    homography_flann = build_flann()
    sift = cv2.SIFT_create()

    matched, discarded, skipped = _match_batch(
        frame_paths, gallery, global_flann, desc_to_card,
        homography_flann, sift, already_done, aligned_dir,
        video_id, data_dir, con, card_id_to_illust,
    )
    print(f"  matched={matched}  discarded={discarded}  skipped={skipped}  total={len(frame_paths)}")
    return matched, discarded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="SIFT-match frames to Pokemon TCG reference cards")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true", help="Process all Pokemon 'frames_extracted' videos")
    p.add_argument("--channel",
                   help="Only process videos from this channel (use with --all)")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    if args.all:
        videos = None
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

    current_video_id: list[str | None] = [None]

    def _process_one(video) -> int:
        current_video_id[0] = video["video_id"]
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        try:
            m, _ = process_video(dict(video), args.data_dir, con, rebuild=args.rebuild)
            set_video_status(con, video["video_id"], "done")
            con.execute("UPDATE videos SET processed_date=date('now') WHERE video_id=?",
                        (video["video_id"],))
            con.commit()
            return m
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            print(f"  status reverted to 'frames_extracted' — fix the issue and re-run", file=sys.stderr)
            set_video_status(con, video["video_id"], "frames_extracted")
            return 0
        except Exception:
            import traceback; traceback.print_exc()
            set_video_status(con, video["video_id"], "error")
            return 0
        finally:
            current_video_id[0] = None

    total_matched = 0
    try:
        if videos is None:
            channel = args.channel or None
            if channel:
                print(f"Filtering to channel: {channel}")
            while True:
                video = claim_next_video(con, "frames_extracted", "processing",
                                         channel=channel, game="pokemon")
                if video is None:
                    print("No more Pokemon videos to process.")
                    break
                total_matched += _process_one(video)
        else:
            for video in videos:
                set_video_status(con, video["video_id"], "processing")
                total_matched += _process_one(video)
    except KeyboardInterrupt:
        vid_id = current_video_id[0]
        if vid_id:
            print(f"\n[interrupted] resetting {vid_id} → frames_extracted", file=sys.stderr)
            set_video_status(con, vid_id, "frames_extracted")
        else:
            print("\n[interrupted]", file=sys.stderr)
        sys.exit(0)

    print(f"\nTotal matched frames: {total_matched}")


if __name__ == "__main__":
    main()
