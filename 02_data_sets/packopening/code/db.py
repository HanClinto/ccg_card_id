"""SQLite helpers for the packopening pipeline.

Schema:
  videos         — one row per YouTube video (mirrors Google Sheet Tab 1)
  frames         — one row per good matched frame (main SIFT pass)
  list_frames    — second-pass matches: List / Special Guest cards found in frames
  list_pass_log  — tracks which videos have been through the List second pass
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    slug            TEXT UNIQUE NOT NULL,
    url             TEXT NOT NULL,
    channel         TEXT,
    title           TEXT,
    set_codes       TEXT,
    lang            TEXT DEFAULT '',
    status          TEXT DEFAULT 'pending',
    added_date      TEXT,
    processed_date  TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT REFERENCES videos(video_id),
    frame_path      TEXT NOT NULL,
    aligned_path    TEXT,
    card_id         TEXT NOT NULL,
    illustration_id TEXT,
    set_code        TEXT,
    num_matches     INTEGER,
    corner0_x REAL, corner0_y REAL,
    corner1_x REAL, corner1_y REAL,
    corner2_x REAL, corner2_y REAL,
    corner3_x REAL, corner3_y REAL,
    matching_area_pct REAL,
    blur_score      REAL,
    phash_dist      INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_frames_card  ON frames(card_id);
CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);

CREATE TABLE IF NOT EXISTS list_frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT REFERENCES videos(video_id),
    frame_path      TEXT NOT NULL,
    aligned_path    TEXT,
    card_id         TEXT NOT NULL,
    illustration_id TEXT,
    set_code        TEXT,
    host_set_code   TEXT,
    num_matches     INTEGER,
    corner0_x REAL, corner0_y REAL,
    corner1_x REAL, corner1_y REAL,
    corner2_x REAL, corner2_y REAL,
    corner3_x REAL, corner3_y REAL,
    matching_area_pct REAL,
    blur_score      REAL,
    phash_dist      INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_list_frames_card  ON list_frames(card_id);
CREATE INDEX IF NOT EXISTS idx_list_frames_video ON list_frames(video_id);

CREATE TABLE IF NOT EXISTS list_pass_log (
    video_id          TEXT PRIMARY KEY REFERENCES videos(video_id),
    status            TEXT DEFAULT 'pending',
    n_eligible_cards  INTEGER DEFAULT 0,
    n_frames_checked  INTEGER DEFAULT 0,
    n_new_matches     INTEGER DEFAULT 0,
    run_at            TEXT
);
"""


_MIGRATIONS = [
    "ALTER TABLE videos ADD COLUMN lang TEXT DEFAULT ''",
    "ALTER TABLE videos ADD COLUMN densified INTEGER DEFAULT 0",
    "ALTER TABLE frames ADD COLUMN phash_dist INTEGER",
    "ALTER TABLE frames ADD COLUMN human_review TEXT",
    "ALTER TABLE videos ADD COLUMN game TEXT DEFAULT 'mtg'",
]


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(DDL)
    for sql in _MIGRATIONS:
        try:
            con.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    return con


def upsert_video(con: sqlite3.Connection, **fields) -> None:
    """Insert or replace a video row."""
    cols = list(fields.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    con.execute(
        f"INSERT OR REPLACE INTO videos ({col_str}) VALUES ({placeholders})",
        list(fields.values()),
    )
    con.commit()


def set_video_status(con: sqlite3.Connection, video_id: str, status: str) -> None:
    con.execute("UPDATE videos SET status=? WHERE video_id=?", (status, video_id))
    con.commit()


def get_video(con: sqlite3.Connection, slug: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM videos WHERE slug=?", (slug,)).fetchone()


def get_videos_by_status(con: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM videos WHERE status=?", (status,)).fetchall()


def claim_next_video(
    con: sqlite3.Connection,
    from_status: str,
    to_status: str = "processing",
    channel: str | None = None,
    game: str | None = None,
) -> sqlite3.Row | None:
    """Atomically claim the next video with from_status, setting it to to_status.

    Safe to call concurrently from multiple worker processes.  SQLite's write
    serialisation ensures only one worker claims each video: the UPDATE uses
    ``AND status=from_status`` so a row that was just claimed by another worker
    will match 0 rows and the caller retries automatically.

    If channel is given, only videos whose channel column matches are considered.
    If game is given, only videos whose game column matches are considered.

    Returns the claimed row (with its original field values), or None when the
    queue is empty.
    """
    conditions = ["status=?"]
    select_params: list = [from_status]
    if channel:
        conditions.append("channel=?")
        select_params.append(channel)
    if game:
        conditions.append("game=?")
        select_params.append(game)
    where = " AND ".join(conditions)
    select_sql = f"SELECT * FROM videos WHERE {where} ORDER BY rowid DESC LIMIT 1"

    while True:
        row = con.execute(select_sql, select_params).fetchone()
        if row is None:
            return None
        cur = con.execute(
            "UPDATE videos SET status=? WHERE video_id=? AND status=?",
            (to_status, row["video_id"], from_status),
        )
        con.commit()
        if cur.rowcount == 1:
            return row
        # Another worker claimed it between our SELECT and UPDATE — retry.


def frames_for_video(con: sqlite3.Connection, video_id: str) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM frames WHERE video_id=?", (video_id,)).fetchall()
