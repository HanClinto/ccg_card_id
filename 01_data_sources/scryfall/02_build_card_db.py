#!/usr/bin/env python3
"""Build the fast SQLite card catalog from all_cards.json + default_cards.json.

Reads all_cards.json (every card in every language) and writes a slim SQLite
database to CCG_FAST_DATA_ROOT (or CCG_DATA_ROOT if fast root is not set).

Also reads default_cards.json to mark each card's is_default flag:
  is_default=1  — Scryfall's preferred printing of this card (English where
                  available, otherwise the printed language for foreign-only
                  releases like P3K).

Only the fields used elsewhere in the codebase are stored; the full JSON
blob is discarded. Result is ~200 MB vs 2.3 GB, loads in <1 second.

Run after syncing Scryfall data (01_sync_data.py), and again whenever
all_cards.json or default_cards.json are updated.

Usage (run from project root):
    python 01_data_sources/scryfall/02_build_card_db.py
    python 01_data_sources/scryfall/02_build_card_db.py --rebuild
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


def _back_illustration_id(card: dict) -> str:
    """Return the back face illustration_id for DFCs, empty string otherwise."""
    faces = card.get("card_faces")
    if faces and len(faces) >= 2:
        return faces[1].get("illustration_id", "")
    return ""


def _back_image_uri(card: dict) -> str:
    """Return the back face PNG URL for DFCs, empty string otherwise."""
    faces = card.get("card_faces")
    if faces and len(faces) >= 2:
        return faces[1].get("image_uris", {}).get("png", "")
    return ""


def load_default_ids(default_json: Path) -> set[str]:
    """Return the set of card IDs present in default_cards.json."""
    if not default_json.exists():
        print(f"  WARNING: {default_json} not found — is_default will be 0 for all cards")
        return set()
    print(f"Reading {default_json} ...")
    with default_json.open(encoding="utf-8") as f:
        cards = json.load(f)
    ids = {c["id"].lower() for c in cards if c.get("id")}
    print(f"  {len(ids):,} default card IDs loaded")
    return ids


def build(src_json: Path, default_json: Path, db_path: Path, rebuild: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not rebuild:
        src_mtime = src_json.stat().st_mtime
        db_mtime = db_path.stat().st_mtime
        if db_mtime >= src_mtime:
            print(f"DB is up to date ({db_path})")
            print("Use --rebuild to force a full rebuild.")
            return
        print("Source JSON is newer than DB — rebuilding.")

    default_ids = load_default_ids(default_json)

    print(f"Reading {src_json} ...")
    with src_json.open(encoding="utf-8") as f:
        cards = json.load(f)
    print(f"  {len(cards):,} cards loaded")

    tmp_path = db_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    con = sqlite3.connect(tmp_path)
    con.executescript(DDL)

    set_names: dict[str, str] = {}

    inserted = skipped = 0
    with tqdm(cards, unit="card", desc="Building DB") as pbar:
        batch = []
        for card in pbar:
            sc = card.get("set", "").lower()
            sn = card.get("set_name", "")
            if sc and sn and sc not in set_names:
                set_names[sc] = sn
            if card.get("image_status") in ("missing", "placeholder"):
                skipped += 1
                continue
            cid = card.get("id", "")
            batch.append((
                cid,
                card.get("oracle_id", ""),
                _illustration_id(card),
                card.get("name", ""),
                card.get("lang", "en"),
                sc,
                card.get("layout", ""),
                card.get("image_status", ""),
                _front_image_uri(card),
                card.get("collector_number", ""),
                card.get("rarity", ""),
                _back_illustration_id(card),
                _back_image_uri(card),
                1 if cid.lower() in default_ids else 0,
            ))
            if len(batch) >= 10_000:
                con.executemany(
                    "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                inserted += len(batch)
                batch.clear()

        if batch:
            con.executemany(
                "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            inserted += len(batch)

    con.executemany(
        "INSERT OR REPLACE INTO sets VALUES (?,?)",
        list(set_names.items()),
    )
    con.commit()
    con.close()

    tmp_path.replace(db_path)
    size_mb = db_path.stat().st_size / 1_048_576
    n_default = len(default_ids)
    print(f"\nDone. {inserted:,} cards inserted, {skipped:,} skipped (missing/placeholder)")
    print(f"  is_default=1: up to {n_default:,} cards (from default_cards.json)")
    print(f"DB: {db_path} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build SQLite card catalog from Scryfall JSON")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild even if DB is newer than source JSON")
    p.add_argument("--src", type=Path, default=cfg.scryfall_all_cards,
                   help="all_cards.json path (default: cfg.scryfall_all_cards)")
    p.add_argument("--default-src", type=Path, default=cfg.scryfall_default_cards,
                   help="default_cards.json path (default: cfg.scryfall_default_cards)")
    p.add_argument("--db", type=Path, default=cfg.card_db_path,
                   help="Output DB path (default: cfg.card_db_path)")
    args = p.parse_args()

    if not args.src.exists():
        print(f"ERROR: source JSON not found: {args.src}", file=sys.stderr)
        print("Run: python 01_data_sources/scryfall/01_sync_data.py", file=sys.stderr)
        sys.exit(1)

    print(f"Source:       {args.src}")
    print(f"Default src:  {args.default_src}")
    print(f"DB path:      {args.db}")
    if args.db == cfg.card_db_path and cfg.fast_data_dir != cfg.data_dir:
        print(f"Fast storage: {cfg.fast_data_dir}  (set CCG_FAST_DATA_ROOT in .env)")
    build(args.src, args.default_src, args.db, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
