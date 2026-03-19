#!/usr/bin/env python3
"""Build the Pokemon TCG SQLite catalog from all_cards.json.

Reads {cfg.pokemontcg_all_cards} and writes a slim SQLite database to
{cfg.pokemontcg_db_path}.

Key design choices
------------------
illustration_id  — Pokemon TCG has no artwork-identity field. We synthesise
                   one as sha1(name + "|" + artist)[:16]. Cards reprinted
                   with the same name and artist share this id, mirroring
                   Scryfall's illustration_id for metric-learning purposes.

pokedex_number   — nationalPokedexNumbers[0] is the closest analog to
                   oracle_id: it groups all printings of "the same Pokémon"
                   across sets. Trainer/Energy cards get 0.

Run after syncing card data (01_sync_data.py):
    python 01_data_sources/pokemontcgio/02_build_card_db.py
    python 01_data_sources/pokemontcgio/02_build_card_db.py --rebuild

Usage (run from project root):
    python 01_data_sources/pokemontcgio/02_build_card_db.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg

DDL = """
CREATE TABLE IF NOT EXISTS sets (
    set_id   TEXT PRIMARY KEY,
    set_name TEXT NOT NULL,
    series   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS cards (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL DEFAULT '',
    set_id           TEXT NOT NULL DEFAULT '',
    number           TEXT NOT NULL DEFAULT '',
    rarity           TEXT NOT NULL DEFAULT '',
    artist           TEXT NOT NULL DEFAULT '',
    illustration_id  TEXT NOT NULL DEFAULT '',
    pokedex_number   INTEGER NOT NULL DEFAULT 0,
    image_uri_large  TEXT NOT NULL DEFAULT '',
    image_uri_small  TEXT NOT NULL DEFAULT '',
    supertype        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cards_set        ON cards(set_id);
CREATE INDEX IF NOT EXISTS idx_cards_illust     ON cards(illustration_id);
CREATE INDEX IF NOT EXISTS idx_cards_pokedex    ON cards(pokedex_number);
"""


def _illustration_id(name: str, artist: str) -> str:
    """Stable 16-char hex id from name + artist."""
    key = f"{name}|{artist}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


def build(src_json: Path, db_path: Path, rebuild: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not rebuild:
        if db_path.stat().st_mtime >= src_json.stat().st_mtime:
            print(f"DB is up to date: {db_path}")
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

    sets: dict[str, tuple[str, str]] = {}  # set_id → (set_name, series)
    batch: list[tuple] = []
    inserted = 0

    with tqdm(cards, unit="card", desc="Building DB") as pbar:
        for card in pbar:
            s = card.get("set", {})
            set_id = s.get("id", "")
            if set_id and set_id not in sets:
                sets[set_id] = (s.get("name", ""), s.get("series", ""))

            name = card.get("name", "")
            artist = card.get("artist", "")
            pokedex = (card.get("nationalPokedexNumbers") or [0])[0]
            images = card.get("images", {})

            batch.append((
                card.get("id", ""),
                name,
                set_id,
                card.get("number", ""),
                card.get("rarity", ""),
                artist,
                _illustration_id(name, artist),
                pokedex,
                images.get("large", ""),
                images.get("small", ""),
                card.get("supertype", ""),
            ))

            if len(batch) >= 5_000:
                con.executemany("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
                inserted += len(batch)
                batch.clear()

    if batch:
        con.executemany("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
        inserted += len(batch)

    con.executemany(
        "INSERT OR REPLACE INTO sets VALUES (?,?,?)",
        [(sid, name, series) for sid, (name, series) in sets.items()],
    )
    con.commit()
    con.close()

    tmp_path.replace(db_path)
    size_mb = db_path.stat().st_size / 1_048_576
    unique_illust = len({_illustration_id(c.get("name",""), c.get("artist","")) for c in cards})
    print(f"\nDone. {inserted:,} cards, {len(sets):,} sets, {unique_illust:,} unique illustration_ids")
    print(f"DB: {db_path} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build Pokemon TCG SQLite catalog")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--src", type=Path, default=cfg.pokemontcg_all_cards)
    p.add_argument("--db", type=Path, default=cfg.pokemontcg_db_path)
    args = p.parse_args()

    if not args.src.exists():
        print(f"ERROR: {args.src} not found", file=sys.stderr)
        print("Run: python 01_data_sources/pokemontcgio/01_sync_data.py", file=sys.stderr)
        sys.exit(1)

    print(f"Source: {args.src}")
    print(f"DB:     {args.db}")
    build(args.src, args.db, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
