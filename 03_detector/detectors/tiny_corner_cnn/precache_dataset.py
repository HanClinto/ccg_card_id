#!/usr/bin/env python3
"""Pre-cache packopening training frames as resized 448×448 JPEGs on local fast storage.

Reading 340k individual JPEG frames from an external drive per training epoch is the
main bottleneck for TinyCornerCNN training.  This script copies every frame that will
be used for training (positive + sampled negatives) to cfg.fast_data_dir at 448×448,
which fits in ~12–15 GB on local SSD and loads ~10–20× faster per epoch.

The cache uses a size-suffixed path to distinguish it from the source frames:
    fast_data_dir / datasets / packopening / frames_448 / {slug} / {frame}.jpg

The dataset loader checks for a cached copy first and falls back to the original.

Usage (run from project root):
    python 03_detector/detectors/tiny_corner_cnn/precache_dataset.py
    python 03_detector/detectors/tiny_corner_cnn/precache_dataset.py --workers 8
    python 03_detector/detectors/tiny_corner_cnn/precache_dataset.py --max-phash-dist 20
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_DETECTOR_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_DETECTOR_DIR))

from ccg_card_id.config import cfg
from dataset import load_from_packopening_db

INPUT_SIZE = 448
JPEG_QUALITY = 90


def cache_one(src: Path, dst: Path) -> bool:
    """Resize src image to 448×448 and save as JPEG at dst. Returns True on success."""
    if dst.exists():
        return True  # already cached
    img = cv2.imread(str(src))
    if img is None:
        return False
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return False
    dst.write_bytes(buf.tobytes())
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-cache packopening frames at 448×448")
    p.add_argument("--data-dir",       type=Path, default=cfg.data_dir)
    p.add_argument("--fast-data-dir",  type=Path, default=cfg.fast_data_dir)
    p.add_argument("--packopening-db", type=Path,
                   default=cfg.data_dir / "datasets/packopening/packopening.db")
    p.add_argument("--max-phash-dist", type=int, default=20)
    p.add_argument("--neg-sample-n",   type=int, default=10_000)
    p.add_argument("--workers",        type=int, default=4,
                   help="Parallel resize workers (default: 4)")
    args = p.parse_args()

    data_dir      = args.data_dir
    fast_data_dir = args.fast_data_dir

    print(f"Source       : {data_dir}")
    print(f"Cache dest   : {fast_data_dir}")

    train_rows, val_rows = load_from_packopening_db(
        args.packopening_db, data_dir,
        neg_sample_n=args.neg_sample_n,
        val_frac=0.05,
        max_phash_dist=args.max_phash_dist,
    )
    all_rows = train_rows + val_rows
    print(f"Total frames to cache: {len(all_rows):,}")

    # Build (src, dst) pairs — skip already-cached files.
    # Cache path uses frames_448/ instead of frames/ to make the resolution explicit.
    pairs: list[tuple[Path, Path]] = []
    for row in all_rows:
        src     = data_dir / row["img_path"]
        dst_rel = row["img_path"].replace(
            "datasets/packopening/frames/", "datasets/packopening/frames_448/", 1
        )
        dst = fast_data_dir / dst_rel
        if not dst.exists():
            pairs.append((src, dst))

    already = len(all_rows) - len(pairs)
    print(f"Already cached: {already:,}  |  To process: {len(pairs):,}")

    if not pairs:
        print("Nothing to do.")
        return

    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(cache_one, src, dst): (src, dst) for src, dst in pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img", desc="Caching"):
            if fut.result():
                ok += 1
            else:
                src, _ = futures[fut]
                print(f"  FAIL: {src}")
                err += 1

    # Estimate cache size
    total_bytes = sum(
        (fast_data_dir / row["img_path"].replace(
            "datasets/packopening/frames/", "datasets/packopening/frames_448/", 1
        )).stat().st_size
        for row in all_rows[:1000]
        if (fast_data_dir / row["img_path"].replace(
            "datasets/packopening/frames/", "datasets/packopening/frames_448/", 1
        )).exists()
    )
    avg_kb = total_bytes / min(1000, len(all_rows)) / 1024
    est_gb = avg_kb * len(all_rows) / 1024 / 1024

    print(f"\nDone. {ok:,} cached, {err:,} errors.")
    print(f"Avg file size: {avg_kb:.1f} KB  |  Est. total cache: {est_gb:.1f} GB")


if __name__ == "__main__":
    main()
