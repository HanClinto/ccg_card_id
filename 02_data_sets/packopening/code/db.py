"""SQLite helpers for the packopening pipeline.

Schema:
  videos   — one row per YouTube video (mirrors Google Sheet Tab 1)
  frames   — one row per good matched frame
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
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_frames_card  ON frames(card_id);
CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(DDL)
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


def frames_for_video(con: sqlite3.Connection, video_id: str) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM frames WHERE video_id=?", (video_id,)).fetchall()
