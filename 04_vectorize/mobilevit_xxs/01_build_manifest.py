#!/usr/bin/env python3
"""Build the artwork_id training manifest.

Combines Scryfall reference images and Munchie real-world scans, grouped by
illustration_id.  Only illustration_ids with >=2 total samples are kept —
this ensures every class has at least one positive pair for ArcFace training.

Train/val/test split is assigned deterministically by illustration_id hash
(seed=42, 85/10/5 default).

Input:
  <data_dir>/mobilevit_xxs/manifest.csv        (Scryfall reference manifest)
  <data_dir>/datasets/munchie/manifest.csv     (Munchie scan manifest)

Output:
  <data_dir>/mobilevit_xxs/artwork_id_manifest.csv

Run 01_build_manifest.py from 04_vectorize/mobilevit_xxs/ after:
  - Scryfall images are downloaded and manifest.csv exists
  - 02_data_sets/munchie/code/03_build_manifest.py has been run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg


_FIELDNAMES = [
    "image_path", "card_id", "card_name", "set_code",
    "illustration_id", "oracle_id", "lang", "source", "split",
]


def _split_by_illustration(illustration_id: str, train: float, val: float, seed: int) -> str:
    key = f"{seed}:{illustration_id}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def _load_manifest_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = Path(row["image_path"])
            if not img.exists():
                continue
            illus_id = row.get("illustration_id", "").strip()
            if not illus_id:
                continue
            records.append({k: row.get(k, "") for k in _FIELDNAMES})
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Build artwork_id combined training manifest")
    p.add_argument(
        "--scryfall-manifest",
        type=Path,
        default=cfg.data_dir / "mobilevit_xxs" / "manifest.csv",
        help="Scryfall reference image manifest (output of build_manifest_from_scryfall)",
    )
    p.add_argument(
        "--munchie-manifest",
        type=Path,
        default=cfg.data_dir / "datasets" / "munchie" / "manifest.csv",
        help="Munchie scan manifest (output of 02_data_sets/munchie/code/03_build_manifest.py)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=cfg.data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum total samples per illustration_id (default 2)",
    )
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild", action="store_true", help="Overwrite existing output")
    args = p.parse_args()

    if args.out.exists() and not args.rebuild:
        with args.out.open() as f:
            n = sum(1 for _ in csv.DictReader(f))
        print(json.dumps({"manifest": str(args.out), "rows": n, "cached": True,
                          "hint": "pass --rebuild to regenerate"}, indent=2))
        return

    # Load records from both sources
    all_records: list[dict] = []
    for src in [args.scryfall_manifest, args.munchie_manifest]:
        if not src.exists():
            print(f"WARNING: manifest not found, skipping: {src}")
            continue
        before = len(all_records)
        all_records.extend(_load_manifest_records(src))
        print(f"  loaded {len(all_records) - before} records from {src.name}")

    # Count samples per illustration_id and filter
    counts = Counter(r["illustration_id"] for r in all_records)
    kept = [r for r in all_records if counts[r["illustration_id"]] >= args.min_samples]
    filtered_out = len(all_records) - len(kept)

    # Re-assign splits by illustration_id (consistent across both sources)
    for r in kept:
        r["split"] = _split_by_illustration(
            r["illustration_id"], args.train_ratio, args.val_ratio, args.seed
        )

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)

    n_illustrations = len({r["illustration_id"] for r in kept})
    split_counts = Counter(r["split"] for r in kept)
    print(json.dumps({
        "manifest": str(args.out),
        "rows": len(kept),
        "filtered_out": filtered_out,
        "unique_illustration_ids": n_illustrations,
        "train": split_counts["train"],
        "val": split_counts["val"],
        "test": split_counts["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
