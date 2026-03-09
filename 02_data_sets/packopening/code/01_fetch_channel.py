#!/usr/bin/env python3
"""Enumerate all videos on a YouTube channel and add new ones to the registry.

No API key or credits required. Uses yt-dlp to pull the channel's video list,
then inserts any previously unseen videos into the local SQLite DB with:
  - status = 'pending'
  - set_codes = '' (empty — fill in manually or run 02_annotate.py)

Re-running is safe and idempotent: videos already in the DB are skipped.

Usage (run from project root):
    # Scan OpenBoosters (default)
    python 02_data_sets/packopening/code/01_fetch_channel.py

    # Scan a different channel
    python 02_data_sets/packopening/code/01_fetch_channel.py \\
        --channel-url "https://www.youtube.com/@SomeOtherChannel"

    # Preview without writing to DB
    python 02_data_sets/packopening/code/01_fetch_channel.py --dry-run

Requires:
    pip install yt-dlp
"""
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db, upsert_video

DEFAULT_CHANNEL = "https://www.youtube.com/@OpenBoosters"


def make_slug(video_id: str, title: str) -> str:
    """Build a filesystem-safe slug: {video_id}_{short_title}."""
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", title).split()
    short = "-".join(w.lower() for w in words[:5] if len(w) > 1)
    short = re.sub(r"[^a-z0-9]+", "-", short).strip("-")[:40]
    return f"{video_id}_{short}"


def fetch_channel_videos(channel_url: str) -> list[dict]:
    """Return [{video_id, url, title, channel}] for all videos on the channel."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(uploader)s",
        "--no-warnings",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if not parts[0].strip():
            continue
        videos.append({
            "video_id": parts[0].strip(),
            "url": f"https://www.youtube.com/watch?v={parts[0].strip()}",
            "title": parts[1].strip() if len(parts) > 1 else "",
            "channel": parts[2].strip() if len(parts) > 2 else "",
        })
    return videos


def main() -> None:
    p = argparse.ArgumentParser(
        description="Enumerate a YouTube channel and add new videos to the packopening registry"
    )
    p.add_argument("--channel-url", default=DEFAULT_CHANNEL,
                   help=f"YouTube channel URL (default: {DEFAULT_CHANNEL})")
    p.add_argument("--channel-name", default=None,
                   help="Override channel name stored in DB")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be added without writing to DB")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = None if args.dry_run else open_db(db_path)

    known_ids: set[str] = set()
    if con:
        known_ids = {r[0] for r in con.execute("SELECT video_id FROM videos").fetchall()}

    print(f"Fetching video list from: {args.channel_url}")
    try:
        videos = fetch_channel_videos(args.channel_url)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(videos)} videos found on channel")
    new_videos = [v for v in videos if v["video_id"] not in known_ids]
    print(f"  {len(new_videos)} new (not yet in DB)")

    if not new_videos:
        print("Nothing to add.")
        return

    today = datetime.date.today().isoformat()
    print(f"  Writing {len(new_videos)} rows to DB...", end=" ", flush=True)
    if args.dry_run:
        for v in new_videos:
            print(f"  WOULD ADD: {v['video_id']}  {v['title'][:70]}")
    else:
        # Insert all rows in a single transaction — much faster than one commit per row
        for v in new_videos:
            slug = make_slug(v["video_id"], v["title"])
            channel = args.channel_name or v.get("channel", "")
            cols = ["video_id", "slug", "url", "channel", "title", "set_codes", "status", "added_date", "notes"]
            con.execute(
                f"INSERT OR REPLACE INTO videos ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
                [v["video_id"], slug, v["url"], channel, v["title"], "", "new", today, ""],
            )
        con.commit()
        print("done.")

    if args.dry_run:
        print(f"\nDry run — {len(new_videos)} videos would be added.")
    else:
        print(f"\nAdded {len(new_videos)} videos to {db_path}")
        print("Next: run 02_annotate.py to classify titles and promote MTG videos to 'pending'.")
        print("      Or set set_codes + status='pending' manually in the DB for specific videos.")


if __name__ == "__main__":
    main()
