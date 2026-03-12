#!/usr/bin/env python3
"""Extract frames from downloaded pack-opening videos.

Fixed-rate extraction across the full video (default: 2 fps).

Blur filtering via Laplacian variance is available but disabled by default
(--blur-threshold 0); homography matching handles quality filtering instead.

Output: datasets/packopening/frames/{slug}/frame_{pts}.jpg

Typical workflow:
    python .../03_extract_frames.py --video-id <id>
    python .../04_match_frames.py --video-id <id>

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
from db import open_db, set_video_status, claim_next_video

EXTRACT_FPS = 2.0         # frames per second for whole-video extraction
BLUR_THRESHOLD = 0.0      # 0 = disabled; try ~30 for YouTube, ~100 for studio/flatbed


# ---------------------------------------------------------------------------
# Blur filter (optional post-processing)
# ---------------------------------------------------------------------------

def laplacian_score(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def extract_frames(
    mp4_path: Path,
    frames_dir: Path,
    fps: float = EXTRACT_FPS,
    rebuild: bool = False,
) -> int:
    """Fixed-rate extraction across the full video, preserving original PTS values.

    Uses the select+vsync vfr filter so that frame_{pts}.jpg filenames contain
    the true presentation timestamp rather than a sequential counter.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob("frame_*.jpg"))
    if existing and not rebuild:
        print(f"  frames already extracted ({len(existing)} found) — skipping (use --rebuild to redo)")
        return len(existing)

    interval = 1.0 / fps
    vf = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',format=yuvj420p"
    cmd = [
        "ffmpeg", "-i", str(mp4_path),
        "-vf", vf,
        "-vsync", "vfr", "-frame_pts", "true", "-q:v", "2", "-y",
        str(frames_dir / "frame_%d.jpg"),
    ]
    print(f"  extracting {mp4_path.name} at {fps} fps (1 frame/{interval:.2f}s) ...")
    _run_ffmpeg(cmd)
    count = len(list(frames_dir.glob("frame_*.jpg")))
    if count == 0:
        raise RuntimeError(f"Extracted 0 frames from {mp4_path.name} — check video file")
    print(f"  extracted {count} frames → {frames_dir}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from pack-opening videos")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true", help="Process all 'downloaded' videos")
    p.add_argument("--rebuild", action="store_true",
                   help="Re-extract even if frames already exist")
    p.add_argument("--fps", type=float, default=EXTRACT_FPS,
                   help=f"Extraction frame rate (default: {EXTRACT_FPS})")
    p.add_argument("--channel",
                   help="Only process videos from this channel (use with --all), e.g. '@MTGUnpacked'")
    p.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                   help="Laplacian variance threshold; 0 = keep all frames (default)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    raw_dir = args.data_dir / "datasets" / "packopening" / "raw"
    frames_root = args.data_dir / "datasets" / "packopening" / "frames"
    con = open_db(db_path)

    if args.all:
        videos = None   # sentinel: use claim_next_video loop below
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

    def _process_one(video) -> bool:
        """Extract frames for a single already-claimed video. Returns True on success."""
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        mp4_path = raw_dir / video["slug"] / f"{video['slug']}.mp4"
        if not mp4_path.exists():
            print(f"  ERROR: video file not found: {mp4_path}", file=sys.stderr)
            set_video_status(con, video["video_id"], "error")
            return False
        try:
            frames_dir = frames_root / video["slug"]
            extract_frames(mp4_path, frames_dir, fps=args.fps, rebuild=args.rebuild)
            if args.blur_threshold > 0:
                kept, removed = filter_blur(frames_dir, args.blur_threshold)
                print(f"  blur filter (threshold={args.blur_threshold}): kept {kept}, removed {removed}")
            set_video_status(con, video["video_id"], "frames_extracted")
            return True
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            print(f"  status reverted to 'downloaded' — will retry on next --all run", file=sys.stderr)
            set_video_status(con, video["video_id"], "downloaded")
            return False

    ok = failed = 0
    if videos is None:
        # --all: claim one video at a time so multiple workers can run safely
        channel = args.channel or None
        if channel:
            print(f"Filtering to channel: {channel}")
        while True:
            video = claim_next_video(con, "downloaded", "processing", channel=channel)
            if video is None:
                print("No more videos to process.")
                break
            if _process_one(video):
                ok += 1
            else:
                failed += 1
    else:
        for video in videos:
            if _process_one(video):
                ok += 1
            else:
                failed += 1

    print(f"\nDone: {ok} OK, {failed} failed.")
    if ok:
        print("Next: python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all")


if __name__ == "__main__":
    main()
