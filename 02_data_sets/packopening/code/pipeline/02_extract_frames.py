#!/usr/bin/env python3
"""Extract keyframes from downloaded pack-opening videos.

Uses ffmpeg I-frames only (GOP keyframes — no interlacing artifacts), then
discards blurry frames using Laplacian variance.

Output: datasets/packopening/frames/{slug}/frame_{pos:06d}.jpg

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/02_extract_frames.py --slug <slug>
    python 02_data_sets/packopening/code/pipeline/02_extract_frames.py --all

Requires: ffmpeg in PATH, pip install opencv-python-headless
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db, set_video_status, get_videos_by_status

BLUR_THRESHOLD = 100.0


def laplacian_score(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(mp4_path: Path, frames_dir: Path, rebuild: bool = False) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob("frame_*.jpg"))
    if existing and not rebuild:
        print(f"  frames already extracted ({len(existing)} found) — skipping ffmpeg")
        return len(existing)

    cmd = [
        "ffmpeg", "-skip_frame", "nokey", "-i", str(mp4_path),
        "-vsync", "vfr", "-frame_pts", "true", "-q:v", "2", "-y",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    print(f"  extracting I-frames from {mp4_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")
    count = len(list(frames_dir.glob("frame_*.jpg")))
    print(f"  extracted {count} I-frames → {frames_dir}")
    return count


def filter_blur(frames_dir: Path, threshold: float) -> tuple[int, int]:
    kept = removed = 0
    for p in sorted(frames_dir.glob("frame_*.jpg")):
        img = cv2.imread(str(p))
        if img is None or laplacian_score(img) < threshold:
            p.unlink()
            removed += 1
        else:
            kept += 1
    return kept, removed


def main() -> None:
    p = argparse.ArgumentParser(description="Extract and blur-filter keyframes")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true", help="Process all 'downloaded' videos")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    raw_dir = args.data_dir / "datasets" / "packopening" / "raw"
    frames_root = args.data_dir / "datasets" / "packopening" / "frames"
    con = open_db(db_path)

    if args.all:
        videos = get_videos_by_status(con, "downloaded")
        if not videos:
            print("No videos with status 'downloaded'.")
            return
    elif args.slug:
        v = con.execute("SELECT * FROM videos WHERE slug=?", (args.slug,)).fetchone()
        if not v:
            print(f"ERROR: slug '{args.slug}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]
    else:
        v = con.execute("SELECT * FROM videos WHERE video_id=?", (args.video_id,)).fetchone()
        if not v:
            print(f"ERROR: video_id '{args.video_id}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]

    ok = failed = 0
    for video in videos:
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        mp4_path = raw_dir / video["slug"] / f"{video['slug']}.mp4"
        if not mp4_path.exists():
            print(f"  ERROR: video file not found: {mp4_path}", file=sys.stderr)
            set_video_status(con, video["video_id"], "error")
            failed += 1
            continue
        try:
            extract_frames(mp4_path, frames_root / video["slug"], rebuild=args.rebuild)
            kept, removed = filter_blur(frames_root / video["slug"], args.blur_threshold)
            print(f"  blur filter: kept {kept}, removed {removed}")
            set_video_status(con, video["video_id"], "frames_extracted")
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            set_video_status(con, video["video_id"], "error")
            failed += 1

    print(f"\nDone: {ok} OK, {failed} failed.")
    if ok:
        print("Next: python 02_data_sets/packopening/code/pipeline/03_precompute_sift.py --set-code <code>")


if __name__ == "__main__":
    main()
