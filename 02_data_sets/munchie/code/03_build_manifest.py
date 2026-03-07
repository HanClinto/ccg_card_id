#!/usr/bin/env python3
"""Build a training manifest CSV from resolved munchie scans.

Reads resolved.jsonl (output of 02_resolve.py) and writes a manifest.csv
with one row per usable image.  The manifest is compatible with the
04_build/mobilevit_xxs training pipeline (same column layout, extended with
illustration_id, oracle_id, lang, and source).

Train/val/test split is assigned deterministically by illustration_id so
that all scans sharing the same artwork land in the same split bucket.

Output columns:
  image_path, card_id, card_name, set_code,
  illustration_id, oracle_id, lang, source, split

Input:
  <data_dir>/datasets/munchie/resolved.jsonl
  <data_dir>/datasets/munchie/images/fronts/  (extracted by 01_extract.py)

Output:
  <data_dir>/datasets/munchie/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg

_USABLE_STATUSES = {"resolved", "ambiguous_same_art"}


def _split_by_illustration(illustration_id: str, train: float, val: float, seed: int) -> str:
    """Deterministic train/val/test split keyed on illustration_id."""
    key = f"{seed}:{illustration_id}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def main() -> None:
    p = argparse.ArgumentParser(description="Build munchie training manifest CSV")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--train-ratio", type=float, default=0.80)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild", action="store_true", help="Overwrite existing manifest.csv")
    args = p.parse_args()

    resolved_path = args.data_dir / "datasets" / "munchie" / "resolved.jsonl"
    images_dir = args.data_dir / "datasets" / "munchie" / "images" / "fronts"
    out_path = args.data_dir / "datasets" / "munchie" / "manifest.csv"

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"resolved.jsonl not found: {resolved_path}\n"
            "Run 02_resolve.py first."
        )
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images dir not found: {images_dir}\n"
            "Run 01_extract.py first."
        )

    if out_path.exists() and not args.rebuild:
        with out_path.open() as f:
            rows = sum(1 for _ in csv.DictReader(f))
        print(f"manifest.csv already exists ({rows} rows).  Pass --rebuild to regenerate.")
        return

    records = [json.loads(line) for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    fieldnames = ["image_path", "card_id", "card_name", "set_code",
                  "illustration_id", "oracle_id", "lang", "source", "split"]

    missing_image = 0
    rows_written = 0
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            if rec.get("status") not in _USABLE_STATUSES:
                continue

            img_path = images_dir / rec["front_filename"]
            if not img_path.exists():
                missing_image += 1
                continue

            illustration_id = rec.get("illustration_id", "")
            split = _split_by_illustration(
                illustration_id or rec["card_id"],
                args.train_ratio,
                args.val_ratio,
                args.seed,
            )

            writer.writerow({
                "image_path": str(img_path),
                "card_id": rec["card_id"],
                "card_name": rec.get("card_name", ""),
                "set_code": rec.get("set_code", "").lower(),
                "illustration_id": illustration_id,
                "oracle_id": rec.get("oracle_id", ""),
                "lang": rec.get("lang", "en"),
                "source": "munchie",
                "split": split,
            })
            rows_written += 1
            split_counts[split] = split_counts.get(split, 0) + 1

    print(f"Wrote {rows_written} rows to {out_path}")
    print(f"  train={split_counts['train']}  val={split_counts['val']}  test={split_counts['test']}")
    if missing_image:
        print(f"  WARNING: {missing_image} records had no image file — run 01_extract.py first")


if __name__ == "__main__":
    main()
