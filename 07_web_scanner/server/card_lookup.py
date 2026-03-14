#!/usr/bin/env python3
"""Card metadata lookup via the catalog SQLite database.

Wraps the catalog DB (built by 01_data_sources/scryfall/04_build_card_db.py
and price-populated by populate_prices.py) with a simple get(card_id) API
that returns everything the client needs to display a scan result.

Usage:
    from card_lookup import make_lookup

    lookup = make_lookup()
    info = lookup.get("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    # {
    #   "scryfall_id":       "...",
    #   "card_name":         "Sol Ring",
    #   "set_code":          "ltr",
    #   "set_name":          "The Lord of the Rings: Tales of Middle-earth",
    #   "tcgplayer_id":      123456,
    #   "price_usd":         1.50,
    #   "price_usd_foil":    None,
    # }
"""

import sqlite3
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — add project root so we can import ccg_card_id
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg  # noqa: E402


# ---------------------------------------------------------------------------
# CardLookup
# ---------------------------------------------------------------------------

_QUERY = """
SELECT
    c.id              AS scryfall_id,
    c.name            AS card_name,
    c.set_code        AS set_code,
    s.set_name        AS set_name,
    p.tcgplayer_id    AS tcgplayer_id,
    p.price_usd       AS price_usd,
    p.price_usd_foil  AS price_usd_foil
FROM cards c
LEFT JOIN sets   s ON s.set_code = c.set_code
LEFT JOIN prices p ON p.card_id  = c.id
WHERE c.id = ?
LIMIT 1;
"""


class CardLookup:
    """Lazy-opening SQLite connection for card metadata queries.

    Designed to be long-lived (one instance per process).  The connection is
    opened on the first call to get() and kept open for subsequent calls.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._con: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"Card catalog DB not found: {self._db_path}\n"
                    "Run: python 01_data_sources/scryfall/04_build_card_db.py\n"
                    "Then: python 07_web_scanner/server/populate_prices.py"
                )
            # Open read-only for safety; the prices table is written once
            # by populate_prices.py, not by the server.
            self._con = sqlite3.connect(
                f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
            )
            self._con.row_factory = sqlite3.Row
        return self._con

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, card_id: str) -> dict[str, Any] | None:
        """Return card metadata dict for a Scryfall UUID, or None if not found.

        The returned dict has keys:
            scryfall_id, card_name, set_code, set_name,
            tcgplayer_id, price_usd, price_usd_foil
        All values may be None if the data is absent (e.g. no price entry).
        """
        row = self._connect().execute(_QUERY, (card_id.lower(),)).fetchone()
        if row is None:
            return None
        return dict(row)

    def close(self) -> None:
        """Close the database connection."""
        if self._con is not None:
            self._con.close()
            self._con = None


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def make_lookup(db_path: Path | None = None) -> CardLookup:
    """Create a CardLookup using the project-configured DB path.

    Pass db_path to override (useful for tests).
    """
    return CardLookup(db_path or cfg.card_db_path)
