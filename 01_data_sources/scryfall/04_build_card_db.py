#!/usr/bin/env python3
"""Build the fast SQLite card catalog from all_cards.json.

Reads all_cards.json from CCG_DATA_ROOT and writes a slim SQLite database
to CCG_FAST_DATA_ROOT (or CCG_DATA_ROOT if fast root is not set).

Only the fields used elsewhere in the codebase are stored; the full JSON
blob is discarded. Result is typically ~60 MB vs 2.3 GB, loads in <1 second.

Run this once after syncing Scryfall data, and again whenever all_cards.json
is updated.

Usage (run from project root):
    python 01_data_sources/scryfall/04_build_card_db.py
    python 01_data_sources/scryfall/04_build_card_db.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg
from ccg_card_id.catalog import DDL


def _front_image_uri(card: dict) -> str:
    """Extract the PNG URL for the front face."""
    uris = card.get("image_uris")
    if uris:
        return uris.get("png", "")
    faces = card.get("card_faces")
    if faces:
        return faces[0].get("image_uris", {}).get("png", "")
    return ""


def _illustration_id(card: dict) -> str:
    """Extract illustration_id, falling back to front card face for DFCs."""
    illust = card.get("illustration_id", "")
    if not illust:
        faces = card.get("card_faces")
        if faces:
            illust = faces[0].get("illustration_id", "")
    return illust


def build(src_json: Path, db_path: Path, rebuild: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not rebuild:
        src_mtime = src_json.stat().st_mtime
        db_mtime = db_path.stat().st_mtime
        if db_mtime >= src_mtime:
            print(f"DB is up to date ({db_path})")
            print("Use --rebuild to force a full rebuild.")
            return
        print("Source JSON is newer than DB — rebuilding.")

    print(f"Reading {src_json} ...")
    with src_json.open(encoding="utf-8") as f:
        cards = json.load(f)
    print(f"  {len(cards):,} cards loaded")

    tmp_path = db_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    con = sqlite3.connect(tmp_path)
    con.executescript(DDL)

    inserted = skipped = 0
    with tqdm(cards, unit="card", desc="Building DB") as pbar:
        batch = []
        for card in pbar:
            if card.get("image_status") in ("missing", "placeholder"):
                skipped += 1
                continue
            batch.append((
                card.get("id", ""),
                card.get("oracle_id", ""),
                _illustration_id(card),
                card.get("name", ""),
                card.get("lang", "en"),
                card.get("set", "").lower(),
                card.get("layout", ""),
                card.get("image_status", ""),
                _front_image_uri(card),
                card.get("collector_number", ""),
                card.get("rarity", ""),
            ))
            if len(batch) >= 10_000:
                con.executemany(
                    "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                inserted += len(batch)
                batch.clear()

        if batch:
            con.executemany(
                "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            inserted += len(batch)

    con.commit()
    con.close()

    tmp_path.replace(db_path)
    size_mb = db_path.stat().st_size / 1_048_576
    print(f"\nDone. {inserted:,} cards inserted, {skipped:,} skipped (missing/placeholder)")
    print(f"DB: {db_path} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build SQLite card catalog from all_cards.json")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild even if DB is newer than source JSON")
    p.add_argument("--src", type=Path, default=cfg.scryfall_all_cards,
                   help="Source JSON path (default: cfg.scryfall_all_cards)")
    p.add_argument("--db", type=Path, default=cfg.card_db_path,
                   help="Output DB path (default: cfg.card_db_path)")
    args = p.parse_args()

    if not args.src.exists():
        print(f"ERROR: source JSON not found: {args.src}", file=sys.stderr)
        print("Run: python 01_data_sources/scryfall/01_sync_data.py", file=sys.stderr)
        sys.exit(1)

    print(f"Source:  {args.src}")
    print(f"DB path: {args.db}")
    if args.db == cfg.card_db_path and cfg.fast_data_dir != cfg.data_dir:
        print(f"Fast storage: {cfg.fast_data_dir}  (set CCG_FAST_DATA_ROOT in .env)")
    build(args.src, args.db, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
