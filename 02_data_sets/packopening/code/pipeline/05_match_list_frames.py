#!/usr/bin/env python3
"""Second-pass SIFT matcher for The List / Special Guest cards.

After the main 04_match_frames.py pass, some frames may show List (plst) or
Special Guest (spg) cards that weren't in the main set gallery.  This script
loads only the eligible plst/spg cards for each video's host set(s), then
checks frames that the main pass did NOT match.

Eligible cards are determined by plst_by_host_set.json, which maps each host
set code to the Scryfall UUIDs of plst/spg cards that could appear in its
boosters.  Build it with:
    python 02_data_sets/packopening/code/build_plst_by_host_set.py

Results go into the list_frames table; progress is tracked in list_pass_log.

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/05_match_list_frames.py --video-id <id>
    python 02_data_sets/packopening/code/pipeline/05_match_list_frames.py --all
    python 02_data_sets/packopening/code/pipeline/05_match_list_frames.py --all --all-frames

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db

# Import helpers from sibling scripts
_spec02 = _ilu.spec_from_file_location(
    "precompute_sift", Path(__file__).parent / "02_precompute_sift.py"
)
_mod02 = _ilu.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)
load_sift_features = _mod02.load_sift_features

_spec04 = _ilu.spec_from_file_location(
    "match_frames", Path(__file__).parent / "04_match_frames.py"
)
_mod04 = _ilu.module_from_spec(_spec04)
_spec04.loader.exec_module(_mod04)

build_global_flann   = _mod04.build_global_flann
build_flann          = _mod04.build_flann
homography_corners   = _mod04.homography_corners
is_valid_quad        = _mod04.is_valid_quad
dewarp               = _mod04.dewarp
compute_phash        = _mod04.compute_phash
_kp_array_to_kps     = _mod04._kp_array_to_kps
_PHASH_AVAILABLE     = _mod04._PHASH_AVAILABLE

try:
    import imagehash
except ImportError:
    pass

MIN_MATCHES = _mod04.MIN_MATCHES
MARGIN      = _mod04.MARGIN
LOWE_RATIO  = _mod04.LOWE_RATIO
REF_W       = _mod04.REF_W
REF_H       = _mod04.REF_H

# Path to the List mapping JSON (relative to project root)
_DEFAULT_PLST_MAP = ROOT / "02_data_sets" / "packopening" / "plst_by_host_set.json"


# ---------------------------------------------------------------------------
# List-pass gallery: load only eligible plst/spg cards
# ---------------------------------------------------------------------------

def load_plst_map(map_path: Path) -> dict[str, list[str]]:
    """Load {host_set_code: [card_id, ...]} from JSON."""
    if not map_path.exists():
        raise FileNotFoundError(
            f"List map not found: {map_path}\n"
            "Run: python 02_data_sets/packopening/code/build_plst_by_host_set.py"
        )
    with open(map_path, encoding="utf-8") as f:
        return json.load(f)


def eligible_card_ids(set_codes: list[str], plst_map: dict[str, list[str]]) -> list[str]:
    """Union of plst/spg card IDs eligible for any of the given host set codes."""
    ids: set[str] = set()
    for sc in set_codes:
        ids.update(plst_map.get(sc.lower(), []))
    return sorted(ids)


def load_list_gallery(
    cache_root: Path,
    card_ids: list[str],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load SIFT features for a specific list of card IDs from plst/spg caches.

    Returns (gallery, all_descs, desc_to_card).  Cards with no cached SIFT
    features are silently skipped (run 02_precompute_sift.py --all-scryfall
    to precompute them).
    """
    gallery: list[dict] = []
    missing = 0

    for cid in card_ids:
        found = False
        for sc in ("plst", "spg"):
            npz_path = cache_root / sc / f"{cid}.npz"
            if not npz_path.exists():
                continue
            kp_array, descs = load_sift_features(npz_path)
            ref_phash = None
            if _PHASH_AVAILABLE:
                ref_img_path = (
                    cfg.scryfall_images_dir / "front"
                    / cid[0] / cid[1] / f"{cid}.png"
                )
                ref_img = cv2.imread(str(ref_img_path))
                if ref_img is not None:
                    ref_phash = compute_phash(ref_img)
            gallery.append({
                "card_id":   cid,
                "set_code":  sc,
                "kps":       _kp_array_to_kps(kp_array),
                "descs":     descs,
                "ref_phash": ref_phash,
            })
            found = True
            break
        if not found:
            missing += 1

    if missing:
        print(f"  NOTE: {missing}/{len(card_ids)} cards have no SIFT cache "
              f"(run 02_precompute_sift.py --set-codes plst spg)")

    if not gallery:
        return gallery, np.empty((0, 128), dtype=np.float32), np.empty(0, dtype=np.int32)

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
# Per-video List pass
# ---------------------------------------------------------------------------

def _match_list_batch(
    frame_paths: list[Path],
    gallery: list[dict],
    global_flann,
    desc_to_card: np.ndarray,
    homography_flann,
    sift,
    host_set_codes: list[str],
    aligned_dir: Path,
    video_id: str,
    data_dir: Path,
    con,
) -> tuple[int, int, int]:
    """Run SIFT matching; write results to list_frames. Returns (matched, discarded, skipped)."""
    matched = discarded = skipped = 0
    host_set_str = ",".join(sorted(host_set_codes))

    with tqdm(frame_paths, unit="frame", desc="  list-matching") as pbar:
        for frame_path in pbar:
            pbar.set_postfix(matched=matched, discarded=discarded, skipped=skipped)
            rel_path = str(frame_path.relative_to(data_dir))

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
                frame_kps, best["kps"], frame_descs, best["descs"],
                frame_h, frame_w, homography_flann,
            )
            if scene_corners is None or not is_valid_quad(scene_corners):
                discarded += 1
                continue

            dewarped = dewarp(frame, scene_corners)
            phash_dist: int | None = None
            if _PHASH_AVAILABLE and best["ref_phash"] is not None:
                match_phash = compute_phash(dewarped)
                phash_dist = int(match_phash - best["ref_phash"])

            frame_pos = frame_path.stem.split("_")[-1]
            aligned_path = aligned_dir / f"{best['card_id']}_{frame_pos}.jpg"
            cv2.imwrite(str(aligned_path), dewarped, [cv2.IMWRITE_JPEG_QUALITY, 92])

            corners = [
                (float(scene_corners[i, 0, 0]) / frame_w,
                 float(scene_corners[i, 0, 1]) / frame_h)
                for i in range(4)
            ]
            idx = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
            corners = corners[idx:] + corners[:idx]
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            area_pct = (max(xs) - min(xs)) * (max(ys) - min(ys))

            con.execute(
                """INSERT OR IGNORE INTO list_frames
                   (video_id, frame_path, aligned_path, card_id, set_code, host_set_code,
                    num_matches, corner0_x, corner0_y, corner1_x, corner1_y,
                    corner2_x, corner2_y, corner3_x, corner3_y, matching_area_pct,
                    phash_dist)
                   VALUES (?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?)""",
                (video_id, rel_path,
                 str(aligned_path.relative_to(data_dir)),
                 best["card_id"], best["set_code"], host_set_str,
                 inliers,
                 corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                 corners[2][0], corners[2][1], corners[3][0], corners[3][1],
                 area_pct, phash_dist),
            )
            con.commit()
            matched += 1

    return matched, discarded, skipped


def process_video_list_pass(
    video: dict,
    data_dir: Path,
    con,
    plst_map: dict[str, list[str]],
    all_frames: bool = False,
    rebuild: bool = False,
) -> int:
    """Run the List second pass for one video. Returns number of new matches."""
    video_id = video["video_id"]
    slug     = video["slug"]
    set_codes = [s for s in re.split(r"[\s,]+", video["set_codes"].strip().lower()) if s]
    if not set_codes:
        print(f"  SKIP: no set_codes")
        return 0

    # Find eligible plst/spg card IDs for this video's host sets
    card_ids = eligible_card_ids(set_codes, plst_map)
    if not card_ids:
        print(f"  SKIP: no List cards mapped for sets {set_codes}")
        _log(con, video_id, "done", n_eligible=0, n_checked=0, n_matched=0)
        return 0

    print(f"  Eligible List cards: {len(card_ids)} (sets: {set_codes})")

    # Determine which frames to check
    frames_dir  = data_dir / "datasets" / "packopening" / "frames" / slug
    aligned_dir = data_dir / "datasets" / "packopening" / "list_aligned" / slug
    cache_root  = data_dir / "datasets" / "packopening" / "sift_cache"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    all_frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not all_frame_paths:
        print(f"  ERROR: no frames in {frames_dir}")
        return 0

    if rebuild:
        con.execute("DELETE FROM list_frames WHERE video_id=?", (video_id,))
        con.commit()
        frame_paths = all_frame_paths
    elif all_frames:
        # Check every frame, even ones the main pass matched
        frame_paths = all_frame_paths
    else:
        # Only frames not matched by the main pass
        matched_paths = {
            r["frame_path"]
            for r in con.execute(
                "SELECT frame_path FROM frames WHERE video_id=?", (video_id,)
            ).fetchall()
        }
        frame_paths = [
            p for p in all_frame_paths
            if str(p.relative_to(data_dir)) not in matched_paths
        ]
        print(f"  Unmatched frames: {len(frame_paths)}/{len(all_frame_paths)}")

    if not frame_paths:
        print(f"  No frames to check (all matched by main pass).")
        _log(con, video_id, "done", n_eligible=len(card_ids),
             n_checked=0, n_matched=0)
        return 0

    # Load gallery
    print(f"  Loading List gallery ({len(card_ids)} cards)...", end=" ", flush=True)
    gallery, all_descs, desc_to_card = load_list_gallery(cache_root, card_ids)
    if not gallery or len(all_descs) == 0:
        print("empty — no SIFT data for eligible cards")
        _log(con, video_id, "done", n_eligible=len(card_ids),
             n_checked=0, n_matched=0)
        return 0
    print(f"{len(gallery)} cards with SIFT features")

    print("  Building FLANN index...", end=" ", flush=True)
    global_flann   = build_global_flann(all_descs)
    homography_flann = build_flann()
    sift = cv2.SIFT_create()
    print("done")

    matched, discarded, skipped = _match_list_batch(
        frame_paths, gallery, global_flann, desc_to_card,
        homography_flann, sift, set_codes,
        aligned_dir, video_id, data_dir, con,
    )
    print(f"  list-matched={matched}  discarded={discarded}  "
          f"skipped={skipped}  checked={len(frame_paths)}")

    _log(con, video_id, "done",
         n_eligible=len(card_ids),
         n_checked=len(frame_paths),
         n_matched=matched)
    return matched


def _log(con, video_id: str, status: str, *,
         n_eligible: int = 0, n_checked: int = 0, n_matched: int = 0) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    con.execute(
        """INSERT OR REPLACE INTO list_pass_log
           (video_id, status, n_eligible_cards, n_frames_checked, n_new_matches, run_at)
           VALUES (?,?,?,?,?,?)""",
        (video_id, status, n_eligible, n_checked, n_matched, now),
    )
    con.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Second-pass SIFT matcher for The List / Special Guest cards"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true",
                       help="Process all 'done' videos that haven't had a List pass")
    p.add_argument("--channel",
                   help="Only process videos from this channel (use with --all)")
    p.add_argument("--all-frames", action="store_true",
                   help="Check every frame, not just those unmatched by main pass")
    p.add_argument("--rebuild", action="store_true",
                   help="Delete existing list_frames for this video and rerun")
    p.add_argument("--include-reprocessed", action="store_true",
                   help="With --all, also reprocess videos that already have a List pass")
    p.add_argument("--plst-map", type=Path, default=_DEFAULT_PLST_MAP,
                   help="Path to plst_by_host_set.json")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    try:
        plst_map = load_plst_map(args.plst_map)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded List map: {len(plst_map)} host sets")

    if args.all:
        # Find 'done' videos not yet processed (or all if --include-reprocessed)
        if args.include_reprocessed:
            if args.channel:
                rows = con.execute(
                    "SELECT * FROM videos WHERE status='done' AND channel=? ORDER BY rowid DESC",
                    (args.channel,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM videos WHERE status='done' ORDER BY rowid DESC"
                ).fetchall()
        else:
            already_done = {
                r["video_id"]
                for r in con.execute(
                    "SELECT video_id FROM list_pass_log WHERE status='done'"
                ).fetchall()
            }
            if args.channel:
                rows = con.execute(
                    "SELECT * FROM videos WHERE status='done' AND channel=? ORDER BY rowid DESC",
                    (args.channel,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM videos WHERE status='done' ORDER BY rowid DESC"
                ).fetchall()
            rows = [r for r in rows if r["video_id"] not in already_done]

        if not rows:
            print("No videos to process.")
            return
        if args.channel:
            print(f"Channel filter: {args.channel}")
        print(f"Processing {len(rows)} video(s)...")
        videos = list(rows)

    elif args.slug:
        v = con.execute("SELECT * FROM videos WHERE slug=?", (args.slug,)).fetchone()
        if not v:
            print(f"ERROR: slug '{args.slug}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]
    else:
        v = con.execute(
            "SELECT * FROM videos WHERE video_id=?", (args.video_id,)
        ).fetchone()
        if not v:
            print(f"ERROR: video_id '{args.video_id}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]

    total = 0
    current_video_id: list[str | None] = [None]

    try:
        for video in videos:
            current_video_id[0] = video["video_id"]
            print(f"\n[{video['video_id']}] {video['title'][:70]}")
            try:
                _log(con, video["video_id"], "running")
                n = process_video_list_pass(
                    dict(video), args.data_dir, con, plst_map,
                    all_frames=args.all_frames,
                    rebuild=args.rebuild,
                )
                total += n
            except Exception:
                import traceback
                traceback.print_exc()
                _log(con, video["video_id"], "error")
            finally:
                current_video_id[0] = None
    except KeyboardInterrupt:
        vid_id = current_video_id[0]
        if vid_id:
            print(f"\n[interrupted] marking {vid_id} list pass as error", file=sys.stderr)
            _log(con, vid_id, "error")
        print("\n[interrupted]", file=sys.stderr)
        sys.exit(0)

    print(f"\nTotal new List matches: {total}")


if __name__ == "__main__":
    main()
