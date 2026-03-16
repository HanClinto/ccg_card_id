#!/usr/bin/env python3
"""Pre-cache packopening aligned training images onto fast SSD with size-suffixed folder naming.

Reads packopening manifest rows, loads source images from cfg.data_dir, resizes to --size,
and writes JPEGs to cfg.fast_data_dir under:
    datasets/packopening/aligned_<size>/<slug>/<file>.jpg

Also writes a manifest with absolute cached paths by default:
    <fast_data_dir>/datasets/packopening/manifest_aligned_<size>.csv

This standardizes resized cache naming (aligned_448, aligned_320, etc.) and avoids
ambiguous folder names like just "aligned" for downscaled caches.

Usage:
  python 04_vectorize/precache_packopening_aligned.py
  python 04_vectorize/precache_packopening_aligned.py --size 320 --workers 8
  python 04_vectorize/precache_packopening_aligned.py --manifest /path/to/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

from ccg_card_id.config import cfg


def _resolve_src(data_dir: Path, image_path_value: str) -> Path:
    p = Path(image_path_value)
    return p if p.is_absolute() else (data_dir / p)


def _dst_rel_from_src_rel(src_rel: Path, size: int) -> Path:
    parts = list(src_rel.parts)
    # Replace .../packopening/aligned/... -> .../packopening/aligned_<size>/...
    for i in range(len(parts) - 1):
        if parts[i] == "packopening" and parts[i + 1] == "aligned":
            parts[i + 1] = f"aligned_{size}"
            return Path(*parts)
    # fallback: append size suffix under packopening
    return Path("datasets") / "packopening" / f"aligned_{size}" / src_rel.name


def _cache_one(src: Path, dst: Path, size: int, jpeg_quality: int) -> bool:
    if dst.exists():
        return True
    img = cv2.imread(str(src))
    if img is None:
        return False
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return False
    dst.write_bytes(buf.tobytes())
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-cache packopening aligned images to fast SSD")
    p.add_argument("--manifest", type=Path,
                   default=cfg.data_dir / "datasets" / "packopening" / "manifest.csv")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--fast-data-dir", type=Path, default=cfg.fast_data_dir)
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--out-manifest", type=Path, default=None,
                   help="Output manifest path (default: fast_data_dir/datasets/packopening/manifest_aligned_<size>.csv)")
    p.add_argument("--relative-manifest-paths", action="store_true",
                   help="Write manifest image_path relative to fast_data_dir (default writes absolute paths)")
    args = p.parse_args()

    manifest = args.manifest
    data_dir = args.data_dir
    fast_data_dir = args.fast_data_dir
    out_manifest = args.out_manifest or (
        fast_data_dir / "datasets" / "packopening" / f"manifest_aligned_{args.size}.csv"
    )

    print(f"Source manifest : {manifest}")
    print(f"Source data dir : {data_dir}")
    print(f"Fast cache dir  : {fast_data_dir}")
    print(f"Image size      : {args.size}")

    with manifest.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pairs: list[tuple[Path, Path, Path, dict]] = []
    missing = 0
    for r in rows:
        src = _resolve_src(data_dir, r["image_path"])
        if not src.exists():
            missing += 1
            continue
        src_rel = src.relative_to(data_dir) if str(src).startswith(str(data_dir)) else Path(r["image_path"])
        dst_rel = _dst_rel_from_src_rel(src_rel, args.size).with_suffix(".jpg")
        dst = fast_data_dir / dst_rel
        pairs.append((src, dst, dst_rel, r))

    print(f"Rows in manifest : {len(rows):,}")
    print(f"Source missing   : {missing:,}")
    print(f"Rows to process  : {len(pairs):,}")

    to_run = [(s, d) for s, d, _, _ in pairs if not d.exists()]
    print(f"Already cached   : {len(pairs) - len(to_run):,}")
    print(f"Need resize/write: {len(to_run):,}")

    ok = err = 0
    if to_run:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_cache_one, s, d, args.size, args.jpeg_quality): (s, d) for s, d in to_run}
            for fut in tqdm(as_completed(futs), total=len(futs), unit="img", desc="Caching"):
                if fut.result():
                    ok += 1
                else:
                    err += 1
                    s, _ = futs[fut]
                    print(f"FAIL: {s}")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["image_path"]
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _, dst, dst_rel, r in pairs:
            rr = dict(r)
            rr["image_path"] = str(dst_rel if args.relative_manifest_paths else dst)
            w.writerow(rr)

    print(f"\nWrote manifest   : {out_manifest}")
    print(f"Cached newly     : {ok:,}")
    print(f"Cache errors     : {err:,}")


if __name__ == "__main__":
    main()
