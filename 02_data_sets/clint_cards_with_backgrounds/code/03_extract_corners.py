#!/usr/bin/env python3
"""Re-extract corner coordinates for all clint_cards_with_backgrounds frames.

The original pipeline (02_find_homography.py) ran successfully and wrote the
aligned/ images, but its data store (pickledb) never flushed, leaving
dataset.tsv empty and homography.json truncated.

This script re-runs homography matching against the cached Scryfall reference
images in 03_reference/ and writes per-frame corner data to corners.csv.

Fixes over the original script:
  - Full UUID extraction via regex (not split('_')[0] which only gets the first
    segment of a hyphenated UUID)
  - Keyed per frame, not per card, so all frames are recorded
  - Numpy values converted to Python floats before serialisation

Usage (run from project root):
    python 02_data_sets/clint_cards_with_backgrounds/code/03_extract_corners.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CLINTUTILS = ROOT.parent / "ClintUtils"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CLINTUTILS.parent))

from ccg_card_id.config import cfg
import ClintUtils.align_img as align_img

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

DATA_ROOT   = cfg.data_dir / "datasets" / "clint_cards_with_backgrounds" / "data"
GOOD_DIR    = DATA_ROOT / "04_data" / "good"
REF_DIR     = DATA_ROOT / "03_reference"
OUTPUT_CSV  = DATA_ROOT / "04_data" / "corners.csv"
RESUME_JSON = DATA_ROOT / "04_data" / "corners_progress.json"

FIELDNAMES = [
    "img_path", "card_id",
    "corner0_x", "corner0_y",
    "corner1_x", "corner1_y",
    "corner2_x", "corner2_y",
    "corner3_x", "corner3_y",
    "num_good_matches", "matching_area_pct",
]


def _float(v) -> float:
    """Convert numpy scalar or anything to a plain Python float."""
    return float(v)


def _extract_corners(scene_corners, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Return 4 (x, y) pairs normalised to [0, 1], ordered top-left first."""
    pts = [(float(scene_corners[i, 0, 0]), float(scene_corners[i, 0, 1])) for i in range(4)]
    # Rotate so top-left corner (min x+y) is first
    idx = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    pts = pts[idx:] + pts[:idx]
    # Normalise
    return [(x / img_w, y / img_h) for x, y in pts]


def main() -> None:
    frames = sorted(GOOD_DIR.glob("*.jpg"))
    print(f"Good frames: {len(frames)}")

    # Load resume state
    done: dict[str, dict] = {}
    if RESUME_JSON.exists():
        done = json.loads(RESUME_JSON.read_text())
        print(f"Resuming — {len(done)} already processed")

    todo = [f for f in frames if f.name not in done]
    print(f"Remaining: {len(todo)}")

    errors = 0
    for i, frame_path in enumerate(todo):
        m = UUID_RE.search(frame_path.name)
        if not m:
            print(f"  SKIP (no UUID): {frame_path.name}")
            continue

        card_id = m.group(0).lower()
        ref_path = REF_DIR / f"{card_id}.png"
        if not ref_path.exists():
            print(f"  SKIP (no reference): {card_id}")
            continue

        img = cv.imread(str(frame_path))
        ref = cv.imread(str(ref_path))
        if img is None or ref is None:
            print(f"  SKIP (unreadable): {frame_path.name}")
            continue

        try:
            result = align_img.align_images(img, ref)
        except Exception as e:
            print(f"  ERROR aligning {frame_path.name}: {e}")
            errors += 1
            continue

        if result.get("error"):
            print(f"  BAD match {frame_path.name}: {result.get('error_message', '')}")
            errors += 1
            continue

        img_h, img_w = img.shape[:2]
        corners = _extract_corners(result["scene_corners"], img_w, img_h)

        # Bounding-box area fraction as a quality signal
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bbox_pct = _float((max(xs) - min(xs)) * (max(ys) - min(ys)))

        record = {
            "img_path": str(frame_path.relative_to(cfg.data_dir)),
            "card_id": card_id,
            "corner0_x": corners[0][0], "corner0_y": corners[0][1],
            "corner1_x": corners[1][0], "corner1_y": corners[1][1],
            "corner2_x": corners[2][0], "corner2_y": corners[2][1],
            "corner3_x": corners[3][0], "corner3_y": corners[3][1],
            "num_good_matches": int(result["num_good_matches"]),
            "matching_area_pct": bbox_pct,
        }
        done[frame_path.name] = record

        if (i + 1) % 50 == 0:
            RESUME_JSON.write_text(json.dumps(done, indent=2))
            print(f"  [{i+1}/{len(todo)}] saved checkpoint ({len(done)} total)")

    # Final save
    RESUME_JSON.write_text(json.dumps(done, indent=2))

    # Write CSV — normalise any legacy absolute img_path to relative
    rows = []
    for v in done.values():
        if "corner0_x" not in v:
            continue
        row = dict(v)
        p = Path(row["img_path"])
        if p.is_absolute():
            try:
                row["img_path"] = str(p.relative_to(cfg.data_dir))
            except ValueError:
                pass  # leave as-is if it can't be relativised
        rows.append(row)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} frames written to {OUTPUT_CSV}")
    print(f"Errors / bad matches: {errors}")


if __name__ == "__main__":
    main()
