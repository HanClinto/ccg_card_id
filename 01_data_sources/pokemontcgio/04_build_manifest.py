#!/usr/bin/env python3
"""Build a ManifestRow-compatible manifest.csv from the Pokemon TCG catalog.

Output path: {cfg.data_dir}/catalog/pokemontcg/manifest.csv

The manifest uses the same schema as the Scryfall/packopening manifests so
it can be passed directly to the embedding model training pipeline:

    image_path, card_id, card_name, set_code, split,
    illustration_id, oracle_id, lang, source

Split is assigned by illustration_id hash (open-set): all printings sharing
the same artwork go to the same split, so val/test contain only artwork IDs
never seen during training.

Only cards that have a local image file are included. Run 03_sync_images.py
first to download images.

Usage (run from project root):
    python 01_data_sources/pokemontcgio/04_build_manifest.py
    python 01_data_sources/pokemontcgio/04_build_manifest.py --supertype Pokémon
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg
from ccg_card_id.pokemon_catalog import PokemonCatalog

MANIFEST_FIELDNAMES = [
    "image_path", "card_id", "card_name", "set_code", "split",
    "illustration_id", "oracle_id", "lang", "source",
]


def assign_split(illustration_id: str) -> str:
    """Open-set split: all cards sharing the same illustration_id go to the same bucket."""
    h = hash(illustration_id) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def _image_path(card: dict, images_dir: Path, data_dir: Path) -> Path | None:
    """Return the absolute path to the card's local image, or None if missing."""
    card_id = card["id"]
    a = card_id[0] if card_id else "x"
    b = card_id[1] if len(card_id) > 1 else "x"
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = images_dir / "large" / a / b / f"{card_id}.{ext}"
        if p.exists():
            return p
    return None


def build_manifest(
    db_path: Path,
    images_dir: Path,
    data_dir: Path,
    out_path: Path,
    supertype: str | None = None,
) -> None:
    catalog = PokemonCatalog(db_path)
    cards = catalog.all_cards(supertype=supertype)
    print(f"Cards in catalog: {len(cards):,}" + (f"  (supertype={supertype})" if supertype else ""))

    rows = []
    missing_image = 0

    for card in tqdm(cards, desc="Building manifest", unit="card"):
        img_abs = _image_path(card, images_dir, data_dir)
        if img_abs is None:
            missing_image += 1
            continue

        # image_path relative to data_dir for portability
        try:
            img_rel = img_abs.relative_to(data_dir)
        except ValueError:
            img_rel = img_abs  # fallback to absolute if outside data_dir

        iid = card["illustration_id"]
        pokedex = card["pokedex_number"]
        oracle_id = f"{pokedex:04d}" if pokedex else ""

        rows.append({
            "image_path": str(img_rel),
            "card_id": card["id"],
            "card_name": card["name"],
            "set_code": card["set_id"],
            "split": assign_split(iid),
            "illustration_id": iid,
            "oracle_id": oracle_id,
            "lang": "en",
            "source": "pokemontcg",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    splits: dict[str, int] = {}
    iids_by_split: dict[str, set] = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        iids_by_split.setdefault(r["split"], set()).add(r["illustration_id"])

    overlap = iids_by_split.get("train", set()) & iids_by_split.get("val", set())

    print(f"\nWrote {len(rows):,} rows → {out_path}")
    print(f"Missing images (excluded): {missing_image:,}")
    print(f"Splits: train={splits.get('train',0):,}  val={splits.get('val',0):,}  test={splits.get('test',0):,}")
    print(f"Unique illustration_ids: "
          f"train={len(iids_by_split.get('train',set())):,}  "
          f"val={len(iids_by_split.get('val',set())):,}  "
          f"test={len(iids_by_split.get('test',set())):,}")
    print(f"illustration_id overlap train∩val: {len(overlap)} (should be 0)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build Pokemon TCG manifest.csv")
    p.add_argument("--db", type=Path, default=cfg.pokemontcg_db_path)
    p.add_argument("--images-dir", type=Path, default=cfg.pokemontcg_images_dir)
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--out", type=Path,
                   default=cfg.data_dir / "catalog" / "pokemontcg" / "manifest.csv")
    p.add_argument("--supertype", default=None,
                   help="Filter by supertype: 'Pokémon', 'Trainer', 'Energy' (default: all)")
    args = p.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found", file=sys.stderr)
        print("Run: python 01_data_sources/pokemontcgio/02_build_card_db.py", file=sys.stderr)
        sys.exit(1)

    build_manifest(args.db, args.images_dir, args.data_dir, args.out, supertype=args.supertype)


if __name__ == "__main__":
    main()
