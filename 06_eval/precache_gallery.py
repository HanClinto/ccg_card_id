#!/usr/bin/env python3
"""Pre-cache gallery (Scryfall reference) images as resized 224×224 JPEGs on local fast storage.

This is the gallery equivalent of 03_detector/.../precache_dataset.py.
Reading 81k Scryfall PNGs from an external drive per eval run is the main bottleneck
for ArcFace gallery embedding computation.  This script copies every image listed in
the gallery manifest to cfg.fast_data_dir at 224×224, which loads 10–20× faster.

The cache uses the original relative path structure but replaces the extension with .jpg:
    fast_data_dir / catalog/scryfall/images/png/front/{a}/{b}/{uuid}.jpg

The eval script checks for the cached JPEG first and falls back to the original PNG.

Usage (run from project root):
    python 06_eval/precache_gallery.py
    python 06_eval/precache_gallery.py --workers 8
    python 06_eval/precache_gallery.py --manifest /path/to/custom_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccg_card_id.config import cfg

INPUT_SIZE = 224
JPEG_QUALITY = 92


def cache_one(src: Path, dst: Path) -> bool:
    """Resize src image to 224×224 and save as JPEG at dst. Returns True on success."""
    if dst.exists():
        return True
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
    p = argparse.ArgumentParser(description="Pre-cache gallery images at 224×224")
    p.add_argument("--manifest",      type=Path,
                   default=cfg.data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv")
    p.add_argument("--data-dir",      type=Path, default=cfg.data_dir)
    p.add_argument("--fast-data-dir", type=Path, default=cfg.fast_data_dir)
    p.add_argument("--workers",       type=int,  default=4)
    args = p.parse_args()

    data_dir      = args.data_dir
    fast_data_dir = args.fast_data_dir

    print(f"Manifest    : {args.manifest}")
    print(f"Source      : {data_dir}")
    print(f"Cache dest  : {fast_data_dir}")

    # Read manifest — collect unique image paths (manifest may repeat paths for train/val/test)
    image_paths: list[Path] = []
    seen: set[str] = set()
    with args.manifest.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("image_path", "")
            if rel and rel not in seen:
                seen.add(rel)
                image_paths.append(Path(rel))

    print(f"Unique images in manifest: {len(image_paths):,}")

    # Build (src, dst) pairs — dst replaces extension with .jpg
    pairs: list[tuple[Path, Path]] = []
    for rel in image_paths:
        # Manifest may use absolute or relative paths
        src = rel if rel.is_absolute() else data_dir / rel
        rel_from_data = rel.relative_to(data_dir) if rel.is_absolute() else rel
        dst = (fast_data_dir / rel_from_data).with_suffix(".jpg")
        if not dst.exists():
            pairs.append((src, dst))

    already = len(image_paths) - len(pairs)
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

    sample_dsts = [dst for _, dst in pairs[:1000] if dst.exists()]
    if sample_dsts:
        total_bytes = sum(d.stat().st_size for d in sample_dsts)
        avg_kb = total_bytes / len(sample_dsts) / 1024
        est_gb = avg_kb * len(image_paths) / 1024 / 1024
        print(f"\nDone. {ok:,} cached, {err:,} errors.")
        print(f"Avg file size: {avg_kb:.1f} KB  |  Est. total cache: {est_gb:.1f} GB")
    else:
        print(f"\nDone. {ok:,} cached, {err:,} errors.")


if __name__ == "__main__":
    main()
