#!/usr/bin/env python3
"""One-off audit: compare downloaded video resolution vs best available on YouTube.

For each video we downloaded below 1080p, query yt-dlp for available formats
and report whether a higher-quality version exists that we should re-download.

Usage (run from project root):
    python 02_data_sets/packopening/code/audit_video_quality.py
    python 02_data_sets/packopening/code/audit_video_quality.py --browser chrome
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg


def get_available_formats(video_id: str, browser: str) -> list[dict] | None:
    """Ask yt-dlp for all available formats without downloading. Returns format list or None on error."""
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--cookies-from-browser", browser,
        "--remote-components", "ejs:github",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return data.get("formats", [])
    except json.JSONDecodeError:
        return None


def best_mp4_height(formats: list[dict]) -> int:
    """Return the highest height available as a proper video stream (not audio-only)."""
    heights = [
        f.get("height") or 0
        for f in formats
        if f.get("height") and f.get("vcodec", "none") != "none"
    ]
    return max(heights, default=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Audit sub-1080p downloaded videos")
    p.add_argument("--browser", default="safari")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    raw_dir = args.data_dir / "datasets" / "packopening" / "raw"

    # Collect all sub-1080p downloads from their .info.json files
    sub_1080: list[tuple[int, str, Path]] = []  # (downloaded_height, video_id, info_json_path)
    for info_path in sorted(raw_dir.rglob("*.info.json")):
        try:
            d = json.loads(info_path.read_text())
        except Exception:
            continue
        h = d.get("height") or 0
        if 0 < h < 1080:
            sub_1080.append((h, d["id"], info_path))

    if not sub_1080:
        print("All downloaded videos are 1080p or better.")
        return

    print(f"Found {len(sub_1080)} sub-1080p videos. Checking available formats on YouTube...\n")
    print(f"{'video_id':<20} {'downloaded':>10} {'available':>10}  {'upgrade?'}")
    print("-" * 60)

    can_upgrade: list[tuple[str, int, int]] = []
    errors: list[str] = []

    for i, (dl_height, video_id, _) in enumerate(sub_1080, 1):
        print(f"  [{i}/{len(sub_1080)}] {video_id} ... ", end="", flush=True)
        formats = get_available_formats(video_id, args.browser)
        if formats is None:
            print("ERROR (private/deleted?)")
            errors.append(video_id)
            continue
        avail = best_mp4_height(formats)
        upgrade = "YES ↑" if avail > dl_height else "no"
        print(f"\r{video_id:<20} {dl_height:>10}p {avail:>10}p  {upgrade}")
        if avail > dl_height:
            can_upgrade.append((video_id, dl_height, avail))

    print()
    print(f"Summary: {len(can_upgrade)} can be upgraded, "
          f"{len(sub_1080) - len(can_upgrade) - len(errors)} already at max quality, "
          f"{len(errors)} errors.")

    if can_upgrade:
        print("\nVideos to re-download:")
        for video_id, dl_h, avail_h in sorted(can_upgrade, key=lambda x: x[2], reverse=True):
            print(f"  {video_id}  {dl_h}p → {avail_h}p")
        print(f"\nRe-download all with:")
        ids = " ".join(f"--video-id {v}" for v, _, _ in can_upgrade)
        print(f"  # (run each separately or update 01_download.py to accept a list)")
        for video_id, _, _ in can_upgrade:
            print(f"  python 02_data_sets/packopening/code/pipeline/01_download.py --video-id {video_id} --force")


if __name__ == "__main__":
    main()
