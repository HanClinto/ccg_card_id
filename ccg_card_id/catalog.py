"""Scryfall card catalog — fast SQLite-backed interface.

Replaces direct all_cards.json / default_cards.json loading throughout the
codebase. The underlying SQLite database is built once by:

    python 01_data_sources/scryfall/04_build_card_db.py

and lives at cfg.card_db_path (on your SSD if CCG_FAST_DATA_ROOT is set).

Typical usage
-------------
    from ccg_card_id.catalog import catalog

    cards = catalog.cards_for_sets(["leg"], lang="it")
    valid  = catalog.valid_set_codes()
    card   = catalog.card("some-uuid")

All query methods return plain dicts (not sqlite3.Row) for easy serialisation.
The connection is opened lazily on first use and shared for the process lifetime.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import cfg


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS sets (
    set_code  TEXT PRIMARY KEY,
    set_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id                   TEXT PRIMARY KEY,
    oracle_id            TEXT,
    illustration_id      TEXT,
    name                 TEXT,
    lang                 TEXT NOT NULL DEFAULT 'en',
    set_code             TEXT NOT NULL DEFAULT '',
    layout               TEXT,
    image_status         TEXT,
    image_uri_png        TEXT,
    collector_number     TEXT,
    rarity               TEXT,
    back_illustration_id TEXT,
    back_image_uri_png   TEXT
);

CREATE INDEX IF NOT EXISTS idx_cards_set_lang      ON cards(set_code, lang);
CREATE INDEX IF NOT EXISTS idx_cards_lang          ON cards(lang);
CREATE INDEX IF NOT EXISTS idx_cards_oracle        ON cards(oracle_id);
CREATE INDEX IF NOT EXISTS idx_cards_illustration  ON cards(illustration_id);
"""


# ---------------------------------------------------------------------------
# Catalog class
# ---------------------------------------------------------------------------

class Catalog:
    """Lazy-opening, process-scoped SQLite connection to the card catalog."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or cfg.card_db_path
        self._con: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"Card catalog DB not found: {self._db_path}\n"
                    "Run: python 01_data_sources/scryfall/04_build_card_db.py"
                )
            self._con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            self._con.row_factory = sqlite3.Row
        return self._con

    def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        rows = self._connect().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def card(self, card_id: str) -> dict | None:
        """Look up a single card by UUID."""
        rows = self._rows("SELECT * FROM cards WHERE id = ?", (card_id,))
        return rows[0] if rows else None

    def cards_by_ids(self, card_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch cards by UUID list. Returns {card_id: card_dict}."""
        if not card_ids:
            return {}
        placeholders = ",".join("?" * len(card_ids))
        rows = self._rows(
            f"SELECT * FROM cards WHERE id IN ({placeholders})",
            tuple(card_ids),
        )
        return {r["id"]: r for r in rows}

    def cards_for_sets(self, set_codes: list[str], lang: str = "en") -> list[dict]:
        """Return all cards matching the given set codes and language."""
        placeholders = ",".join("?" * len(set_codes))
        return self._rows(
            f"SELECT * FROM cards WHERE set_code IN ({placeholders}) AND lang = ?",
            (*[s.lower() for s in set_codes], lang.lower()),
        )

    def cards_by_name_set(self, name: str, set_code: str, lang: str = "en") -> list[dict]:
        """Return all cards matching name (case-insensitive) + set_code + lang."""
        return self._rows(
            "SELECT * FROM cards WHERE lower(name) = ? AND set_code = ? AND lang = ?",
            (name.lower(), set_code.lower(), lang.lower()),
        )

    def all_cards(self, lang: str | None = None) -> list[dict]:
        """Return all cards, optionally filtered by language code."""
        if lang:
            return self._rows("SELECT * FROM cards WHERE lang = ?", (lang.lower(),))
        return self._rows("SELECT * FROM cards", ())

    def valid_set_codes(self) -> set[str]:
        """Return all set codes present in the catalog."""
        rows = self._rows("SELECT DISTINCT set_code FROM cards")
        return {r["set_code"] for r in rows}

    def set_names(self) -> dict[str, str]:
        """Return {set_code: set_name} for all sets in the catalog."""
        rows = self._rows("SELECT set_code, set_name FROM sets")
        return {r["set_code"]: r["set_name"] for r in rows}

    def set_codes_for_lang(self, lang: str) -> set[str]:
        rows = self._rows(
            "SELECT DISTINCT set_code FROM cards WHERE lang = ?", (lang.lower(),)
        )
        return {r["set_code"] for r in rows}

    def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None


# ---------------------------------------------------------------------------
# Shared default instance
# ---------------------------------------------------------------------------

catalog = Catalog()
