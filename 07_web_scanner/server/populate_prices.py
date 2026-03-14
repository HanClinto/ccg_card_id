#!/usr/bin/env python3
"""Populate the `prices` table in the catalog SQLite database.

One-time migration script.  Reads default_cards.json from cfg.data_dir and
upserts price data into catalog/scryfall/cards.db.

Usage:
    python 07_web_scanner/server/populate_prices.py

Re-running is safe: the table is dropped and recreated each time so the data
stays in sync with the JSON source.  On a modern machine with the full
default_cards.json (~280k cards) this takes roughly 10-30 seconds.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — add project root so we can import ccg_card_id
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL_CREATE = """
CREATE TABLE IF NOT EXISTS prices (
    card_id         TEXT PRIMARY KEY,
    tcgplayer_id    INTEGER,
    price_usd       REAL,
    price_usd_foil  REAL,
    updated_at      TEXT NOT NULL
);
"""

_DDL_DROP = "DROP TABLE IF EXISTS prices;"

_INSERT = """
INSERT OR REPLACE INTO prices (card_id, tcgplayer_id, price_usd, price_usd_foil, updated_at)
VALUES (?, ?, ?, ?, ?);
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_price(value) -> float | None:
    """Convert a Scryfall price string like "1.23" or null to float or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def populate_prices(db_path: Path, json_path: Path, batch_size: int = 5000) -> None:
    if not json_path.exists():
        print(f"ERROR: default_cards.json not found at {json_path}")
        print("Download it from https://scryfall.com/docs/api/bulk-data")
        sys.exit(1)

    if not db_path.exists():
        print(f"ERROR: card catalog DB not found at {db_path}")
        print("Run: python 01_data_sources/scryfall/04_build_card_db.py")
        sys.exit(1)

    print(f"Loading {json_path} ...")
    with json_path.open("r", encoding="utf-8") as f:
        cards = json.load(f)
    print(f"  {len(cards):,} cards loaded")

    now = datetime.now(tz=timezone.utc).isoformat()

    print(f"Opening {db_path} ...")
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute(_DDL_DROP)
        con.execute(_DDL_CREATE)
        con.commit()

        rows_inserted = 0
        batch: list[tuple] = []

        for card in cards:
            card_id = card.get("id")
            if not card_id:
                continue

            tcgplayer_id = card.get("tcgplayer_id")  # may be None or int
            prices = card.get("prices") or {}
            price_usd = _parse_price(prices.get("usd"))
            price_usd_foil = _parse_price(prices.get("usd_foil"))

            batch.append((card_id, tcgplayer_id, price_usd, price_usd_foil, now))

            if len(batch) >= batch_size:
                con.executemany(_INSERT, batch)
                con.commit()
                rows_inserted += len(batch)
                batch.clear()
                print(f"  inserted {rows_inserted:,} rows ...", end="\r", flush=True)

        if batch:
            con.executemany(_INSERT, batch)
            con.commit()
            rows_inserted += len(batch)

        print(f"\nDone. {rows_inserted:,} rows written to prices table.")

        # Quick sanity check
        count = con.execute("SELECT COUNT(*) FROM prices WHERE price_usd IS NOT NULL").fetchone()[0]
        print(f"  {count:,} cards have a USD price.")

    finally:
        con.close()


if __name__ == "__main__":
    populate_prices(db_path=cfg.card_db_path, json_path=cfg.scryfall_default_cards)
