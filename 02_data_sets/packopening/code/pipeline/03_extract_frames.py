#!/usr/bin/env python3
"""Extract frames from downloaded pack-opening videos.

Two extraction modes (can be combined in sequence):

  scene (default)
      ffmpeg scene-change detection with a minimum interval between frames.
      Gives ~1 frame per distinct visual moment across the whole video.
      Good for a fast first pass to locate where cards appear.

  dense
      Fixed-rate extraction starting from a given timestamp (--dense-from).
      Use after SIFT matching identifies the first card timestamp, to get
      high-density coverage of the card-heavy portion of the video.
      Adds frames alongside existing scene frames; does not replace them.

Blur filtering via Laplacian variance is available but disabled by default
(--blur-threshold 0); homography matching handles quality filtering instead.

Output: datasets/packopening/frames/{slug}/frame_{pts}.jpg

Typical workflow:
    # Step 1: coarse scene-change pass
    python .../03_extract_frames.py --video-id <id>
    # Step 2: SIFT match (reports first-card timestamp)
    python .../04_match_frames.py --video-id <id>
    # Step 3: dense pass from first card onward
    python .../03_extract_frames.py --video-id <id> --dense-from <seconds>
    # Step 4: re-run SIFT match to pick up new frames
    python .../04_match_frames.py --video-id <id>

Requires: ffmpeg in PATH, pip install opencv-python-headless
"""
from __future__ import annotations

import argparse
import json
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

LEAD_IN = 5.0             # seconds before first card to start dense pass
BLUR_THRESHOLD = 0.0      # 0 = disabled; try ~30 for YouTube, ~100 for studio/flatbed
SCENE_THRESHOLD = 0.3     # scene-change sensitivity (0–1, lower = more frames)
SCENE_MIN_INTERVAL = 2.0  # minimum seconds between scene-change frames
DENSE_FPS = 2.0           # frames per second for dense pass


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
# Timestamp helpers
# ---------------------------------------------------------------------------

def get_video_timebase(mp4_path: Path) -> float:
    """Return the video stream timebase as a float (e.g. '1/90000' → 1.111e-5)."""
    cmd = [
        "ffprobe", "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=time_base",
        "-of", "json",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    tb_str = data["streams"][0]["time_base"]  # e.g. "1/90000"
    num, den = tb_str.split("/")
    return int(num) / int(den)


def pts_to_seconds(frame_path_str: str, timebase: float) -> float:
    """Convert the PTS embedded in a frame filename to seconds.

    Expects filenames like frame_5400000.jpg, where the number after the
    last underscore is the raw PTS written by ffmpeg -frame_pts true.
    """
    stem = Path(frame_path_str).stem      # e.g. "frame_5400000"
    pts_str = stem.rsplit("_", 1)[-1]     # e.g. "5400000"
    return int(pts_str) * timebase


# ---------------------------------------------------------------------------
# Extraction modes
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def extract_scene(
    mp4_path: Path,
    frames_dir: Path,
    scene_threshold: float = SCENE_THRESHOLD,
    min_interval: float = SCENE_MIN_INTERVAL,
    rebuild: bool = False,
) -> int:
    """Scene-change extraction across the full video."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob("frame_*.jpg"))
    if existing and not rebuild:
        print(f"  scene frames already extracted ({len(existing)} found) — skipping (use --rebuild to redo)")
        return len(existing)

    # select frames at scene changes, enforcing a minimum gap between them
    vf = f"select='gt(scene,{scene_threshold})*gte(t-prev_selected_t,{min_interval})'"
    cmd = [
        "ffmpeg", "-i", str(mp4_path),
        "-vf", vf,
        "-vsync", "vfr", "-frame_pts", "true", "-q:v", "2", "-y",
        str(frames_dir / "frame_%d.jpg"),
    ]
    print(f"  scene-change extraction from {mp4_path.name} "
          f"(threshold={scene_threshold}, min_interval={min_interval}s) ...")
    _run_ffmpeg(cmd)
    count = len(list(frames_dir.glob("frame_*.jpg")))
    print(f"  extracted {count} scene frames → {frames_dir}")
    return count


def extract_dense(
    mp4_path: Path,
    frames_dir: Path,
    from_seconds: float,
    fps: float = DENSE_FPS,
) -> int:
    """Fixed-rate extraction from `from_seconds` to end of video.

    Frames are named frame_{pts}.jpg alongside scene-change frames.
    PTS values are globally unique per video so there is no collision.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-ss", str(max(0.0, from_seconds)), "-i", str(mp4_path),
        "-vf", f"fps={fps}",
        "-frame_pts", "true", "-q:v", "2", "-y",
        str(frames_dir / "frame_%d.jpg"),
    ]
    print(f"  dense extraction from {mp4_path.name} "
          f"(from={from_seconds:.1f}s, fps={fps}) ...")
    _run_ffmpeg(cmd)
    count = len(list(frames_dir.glob("frame_*.jpg")))
    print(f"  extracted {count} total frames → {frames_dir}")
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

    scene_grp = p.add_argument_group("scene-change extraction (default first pass)")
    scene_grp.add_argument("--scene-threshold", type=float, default=SCENE_THRESHOLD,
                           help=f"Scene change sensitivity 0–1 (default: {SCENE_THRESHOLD})")
    scene_grp.add_argument("--min-interval", type=float, default=SCENE_MIN_INTERVAL,
                           help=f"Minimum seconds between scene frames (default: {SCENE_MIN_INTERVAL})")
    scene_grp.add_argument("--no-scene", action="store_true",
                           help="Skip scene-change extraction")

    dense_grp = p.add_argument_group("dense extraction (second pass after first card found)")
    dense_grp.add_argument("--dense-from", type=float, metavar="SECONDS",
                           help="Start time in seconds for dense extraction")
    dense_grp.add_argument("--dense-fps", type=float, default=DENSE_FPS,
                           help=f"Frame rate for dense pass (default: {DENSE_FPS})")

    p.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                   help="Laplacian variance threshold; 0 = keep all frames (default)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    if args.no_scene and args.dense_from is None:
        p.error("--no-scene requires --dense-from")

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
            frames_dir = frames_root / video["slug"]
            if not args.no_scene:
                extract_scene(mp4_path, frames_dir,
                              scene_threshold=args.scene_threshold,
                              min_interval=args.min_interval,
                              rebuild=args.rebuild)
            if args.dense_from is not None:
                extract_dense(mp4_path, frames_dir,
                              from_seconds=args.dense_from,
                              fps=args.dense_fps)
            if args.blur_threshold > 0:
                kept, removed = filter_blur(frames_dir, args.blur_threshold)
                print(f"  blur filter (threshold={args.blur_threshold}): kept {kept}, removed {removed}")
            set_video_status(con, video["video_id"], "frames_extracted")
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            set_video_status(con, video["video_id"], "error")
            failed += 1

    print(f"\nDone: {ok} OK, {failed} failed.")
    if ok:
        print("Next: python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --set-code <code>")


if __name__ == "__main__":
    main()
