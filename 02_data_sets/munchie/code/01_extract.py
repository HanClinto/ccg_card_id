#!/usr/bin/env python3
"""Extract munchie front-face images from munchie_fronts.zip.

Input:  <data_dir>/datasets/munchie/data/images/munchie_fronts.zip
Output: <data_dir>/datasets/munchie/images/fronts/<filename>.jpg   (812 files)

Idempotent: already-extracted files are skipped unless --rebuild is passed.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Extract munchie front images from zip")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--rebuild", action="store_true", help="Re-extract even if files already exist")
    args = p.parse_args()

    src_zip = args.data_dir / "datasets" / "munchie" / "data" / "images" / "munchie_fronts.zip"
    out_dir = args.data_dir / "datasets" / "munchie" / "images" / "fronts"

    if not src_zip.exists():
        raise FileNotFoundError(f"Source zip not found: {src_zip}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src_zip) as z:
        entries = [
            n for n in z.namelist()
            if n.lower().endswith(".jpg") and "__MACOSX" not in n and not n.endswith("/")
        ]
        extracted = skipped = 0
        for entry in entries:
            fname = Path(entry).name
            dest = out_dir / fname
            if dest.exists() and not args.rebuild:
                skipped += 1
                continue
            dest.write_bytes(z.read(entry))
            extracted += 1

    print(f"Extracted {extracted} images to {out_dir}  (skipped {skipped} already present)")


if __name__ == "__main__":
    main()
