"""Pokemon TCG card catalog — SQLite-backed interface.

Mirrors the public API of ccg_card_id.catalog (Scryfall) so downstream code
can work with either game without knowing the difference.

The underlying database is built by:
    python 01_data_sources/pokemontcgio/02_build_card_db.py

and lives at cfg.pokemontcg_db_path.

Typical usage
-------------
    from ccg_card_id.pokemon_catalog import pokemon_catalog

    card  = pokemon_catalog.card("base1-4")
    cards = pokemon_catalog.cards_by_ids(["base1-4", "base2-4"])
    valid = pokemon_catalog.valid_set_codes()

All query methods return plain dicts for easy serialisation.
The connection is opened lazily on first use.

Field mapping to ManifestRow
-----------------------------
    card_id          → id
    card_name        → name
    set_code         → set_id
    illustration_id  → illustration_id  (synthetic sha1(name|artist)[:16])
    oracle_id        → pokedex_number as zero-padded string ("0007" for Squirtle)
    lang             → "en"  (API is English-primary)
    source           → "pokemontcg"
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import cfg


class PokemonCatalog:
    """Lazy-opening, process-scoped SQLite connection to the Pokemon TCG catalog."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or cfg.pokemontcg_db_path
        self._con: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"Pokemon TCG catalog DB not found: {self._db_path}\n"
                    "Run: python 01_data_sources/pokemontcgio/02_build_card_db.py"
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
        """Look up a single card by Pokemon TCG ID (e.g. 'base1-4')."""
        rows = self._rows("SELECT * FROM cards WHERE id = ?", (card_id,))
        return rows[0] if rows else None

    def cards_by_ids(self, card_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch cards by ID list. Returns {card_id: card_dict}."""
        if not card_ids:
            return {}
        placeholders = ",".join("?" * len(card_ids))
        rows = self._rows(
            f"SELECT * FROM cards WHERE id IN ({placeholders})",
            tuple(card_ids),
        )
        return {r["id"]: r for r in rows}

    def cards_for_sets(self, set_ids: list[str]) -> list[dict]:
        """Return all cards for the given set IDs."""
        placeholders = ",".join("?" * len(set_ids))
        return self._rows(
            f"SELECT * FROM cards WHERE set_id IN ({placeholders})",
            tuple(set_ids),
        )

    def all_cards(self, supertype: str | None = None) -> list[dict]:
        """Return all cards, optionally filtered by supertype ('Pokémon', 'Trainer', 'Energy')."""
        if supertype:
            return self._rows("SELECT * FROM cards WHERE supertype = ?", (supertype,))
        return self._rows("SELECT * FROM cards", ())

    def valid_set_codes(self) -> set[str]:
        """Return all set IDs present in the catalog."""
        rows = self._rows("SELECT DISTINCT set_id FROM cards")
        return {r["set_id"] for r in rows}

    def set_names(self) -> dict[str, str]:
        """Return {set_id: set_name} for all sets."""
        rows = self._rows("SELECT set_id, set_name FROM sets")
        return {r["set_id"]: r["set_name"] for r in rows}

    def cards_by_illustration(self, illustration_id: str) -> list[dict]:
        """Return all cards sharing the same synthetic illustration_id."""
        return self._rows(
            "SELECT * FROM cards WHERE illustration_id = ?", (illustration_id,)
        )

    def close(self) -> None:
        if self._con:
            self._con.close()
            self._con = None


# ---------------------------------------------------------------------------
# Shared default instance
# ---------------------------------------------------------------------------

pokemon_catalog = PokemonCatalog()
