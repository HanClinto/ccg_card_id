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
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _build_card_index(all_cards_json: Path) -> dict[str, dict]:
    """Build card_id (lower) -> card dict index from all_cards.json."""
    print(f"Loading {all_cards_json} ...", flush=True)
    cards = json.loads(all_cards_json.read_text(encoding="utf-8"))
    return {str(c["id"]).lower(): c for c in cards if "id" in c}


def main() -> None:
    p = argparse.ArgumentParser(description="Build daniel_scans query manifest")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    images_dir = args.data_dir / "datasets" / "daniel_scans" / "images_processed"
    all_cards_path = args.data_dir / "all_cards.json"
    out_path = args.data_dir / "datasets" / "daniel_scans" / "query_manifest.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not all_cards_path.exists():
        raise FileNotFoundError(f"all_cards.json not found: {all_cards_path}")

    if out_path.exists() and not args.rebuild:
        with out_path.open() as f:
            n = sum(1 for _ in csv.DictReader(f))
        print(f"query_manifest.csv already exists ({n} rows). Pass --rebuild to regenerate.")
        return

    card_index = _build_card_index(all_cards_path)

    images = sorted(p for p in images_dir.glob("*.jpg"))
    rows: list[dict] = []
    missing_uuid = 0
    missing_card = 0

    for img in images:
        m = UUID_RE.search(img.stem)
        if not m:
            missing_uuid += 1
            continue
        card_id = m.group(0).lower()
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
