#!/usr/bin/env python3
"""Build a query manifest CSV for the daniel_scans dataset.

Reads the processed images directory and cross-references each card_id against
all_cards.json to populate illustration_id.

Filename format (images_processed/):
  {cardname}_{setcode}_{sleevetype}_{card_id_uuid}_{seq}[_foil].jpg
  e.g. evolvingwilds_C15_nosleeve_e7ee2fa1-5aed-4612-bfa9-cfbf5d282c9b_001.jpg

Input:
  <data_dir>/datasets/daniel_scans/images_processed/   (150 jpg files)
  <data_dir>/all_cards.json

Output:
  <data_dir>/datasets/daniel_scans/query_manifest.csv
  Columns: image_path, card_id, illustration_id
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
from ccg_card_id.catalog import catalog

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def main() -> None:
    p = argparse.ArgumentParser(description="Build daniel_scans query manifest")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    images_dir = args.data_dir / "datasets" / "daniel_scans" / "images_processed"
    out_path = args.data_dir / "datasets" / "daniel_scans" / "query_manifest.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    if out_path.exists() and not args.rebuild:
        with out_path.open() as f:
            n = sum(1 for _ in csv.DictReader(f))
        print(f"query_manifest.csv already exists ({n} rows). Pass --rebuild to regenerate.")
        return

    images = sorted(p for p in images_dir.glob("*.jpg"))
    missing_uuid = 0
    missing_card = 0

    img_card_ids = []
    for img in images:
        m = UUID_RE.search(img.stem)
        if not m:
            missing_uuid += 1
            img_card_ids.append((img, None))
        else:
            img_card_ids.append((img, m.group(0).lower()))

    card_ids = [cid for _, cid in img_card_ids if cid]
    card_index = catalog.cards_by_ids(card_ids)

    rows: list[dict] = []
    for img, card_id in img_card_ids:
        if card_id is None:
            continue
        card = card_index.get(card_id)
        if card is None:
            missing_card += 1
            continue
        rows.append({
            "image_path": str(img),
            "card_id": card_id,
            "illustration_id": str(card.get("illustration_id", "") or "").lower(),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "card_id", "illustration_id"])
        writer.writeheader()
        writer.writerows(rows)

    unique_cards = len({r["card_id"] for r in rows})
    unique_illus = len({r["illustration_id"] for r in rows if r["illustration_id"]})
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  unique card_ids:       {unique_cards}")
    print(f"  unique illustration_ids: {unique_illus}")
    if missing_uuid:
        print(f"  WARNING: {missing_uuid} files had no UUID in filename")
    if missing_card:
        print(f"  WARNING: {missing_card} card_ids not found in all_cards.json")


if __name__ == "__main__":
    main()
