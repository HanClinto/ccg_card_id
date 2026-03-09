#!/usr/bin/env python3
"""Download YouTube pack-opening videos using yt-dlp.

Looks up a video by slug (or video_id) in the SQLite registry, downloads it
to datasets/packopening/raw/{slug}/, and updates status to 'downloading' then
'downloaded' (or 'error' on failure).

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/01_download.py --slug dQw4w9WgXcQ_lea_alpha-booster
    python 02_data_sets/packopening/code/pipeline/01_download.py --video-id dQw4w9WgXcQ
    python 02_data_sets/packopening/code/pipeline/01_download.py --all
    python 02_data_sets/packopening/code/pipeline/01_download.py --slug ... --force

Requires:
    pip install yt-dlp
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db, set_video_status, get_videos_by_status


def download_video(video: dict, raw_dir: Path, force: bool = False) -> Path:
    """Download video to raw_dir/{slug}/.  Returns path to downloaded .mp4."""
    slug = video["slug"]
    out_dir = raw_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f"{slug}.mp4"
    if mp4_path.exists() and not force:
        print(f"  already downloaded: {mp4_path.name}")
        return mp4_path

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--no-playlist",
        "--output", str(out_dir / f"{slug}.%(ext)s"),
        video["url"],
    ]
    print(f"  downloading: {video['url']}")
    print(f"  → {out_dir}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp exited with code {result.returncode}")

    for ext in ("mp4", "mkv", "webm"):
        candidate = out_dir / f"{slug}.{ext}"
        if candidate.exists():
            if ext != "mp4":
                candidate.rename(mp4_path)
            return mp4_path

    raise RuntimeError(f"Downloaded file not found in {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download packopening YouTube videos")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug", help="Video slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true", help="Download all videos with status 'pending'")
    p.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    raw_dir = args.data_dir / "datasets" / "packopening" / "raw"
    con = open_db(db_path)

    if args.all:
        videos = get_videos_by_status(con, "pending")
        if not videos:
            print("No videos with status 'pending'.")
            return
        print(f"Downloading {len(videos)} pending video(s)...")
    elif args.slug:
        v = con.execute("SELECT * FROM videos WHERE slug=?", (args.slug,)).fetchone()
        if not v:
            print(f"ERROR: No video with slug '{args.slug}'.", file=sys.stderr)
            sys.exit(1)
        videos = [v]
    else:
        v = con.execute("SELECT * FROM videos WHERE video_id=?", (args.video_id,)).fetchone()
        if not v:
            print(f"ERROR: No video with video_id '{args.video_id}'.", file=sys.stderr)
            sys.exit(1)
        videos = [v]

    ok = failed = 0
    for video in videos:
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        set_video_status(con, video["video_id"], "downloading")
        try:
            mp4_path = download_video(dict(video), raw_dir, force=args.force)
            set_video_status(con, video["video_id"], "downloaded")
            print(f"  OK: {mp4_path}")
            ok += 1
        except Exception as e:
            set_video_status(con, video["video_id"], "error")
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1

    print(f"\nDownloaded {ok}. Failures: {failed}.")
    if ok:
        print("Next: python 02_data_sets/packopening/code/pipeline/02_extract_frames.py --all")


if __name__ == "__main__":
    main()
