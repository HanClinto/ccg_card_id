#!/usr/bin/env python3
"""Extract corner coordinates for all sol_ring good frames.

Runs SIFT homography matching against the Scryfall catalog reference PNG for
each frame, and writes corner data to corners.csv.  Output format matches
clint_cards_with_backgrounds/corners.csv so both datasets can be evaluated
with the same benchmark harness.

Usage (run from project root):
    python 02_data_sets/sol_ring/code/04_extract_corners.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import cv2 as cv

_CLINTUTILS_PARENT = Path(__file__).resolve().parents[4]  # parent of ClintUtils sibling repo
sys.path.insert(0, str(_CLINTUTILS_PARENT))

from ccg_card_id.config import cfg
import ClintUtils.align_img as align_img

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

SOLRING_DIR = cfg.data_dir / "datasets" / "solring"
GOOD_DIR    = SOLRING_DIR / "04_data" / "good"
CAT_ROOT    = cfg.data_dir / "catalog" / "scryfall" / "images" / "png" / "front"
OUTPUT_CSV  = SOLRING_DIR / "04_data" / "corners.csv"
RESUME_JSON = SOLRING_DIR / "04_data" / "corners_progress.json"

FIELDNAMES = [
    "img_path", "card_id",
    "corner0_x", "corner0_y",
    "corner1_x", "corner1_y",
    "corner2_x", "corner2_y",
    "corner3_x", "corner3_y",
    "num_good_matches", "matching_area_pct",
]


def _ref_path(card_id: str) -> Path:
    return CAT_ROOT / card_id[0] / card_id[1] / f"{card_id}.png"


def _extract_corners(scene_corners, img_w: int, img_h: int) -> list[tuple[float, float]]:
    """Return 4 (x, y) pairs normalised to [0, 1], TL first (min x+y sum)."""
    pts = [(float(scene_corners[i, 0, 0]), float(scene_corners[i, 0, 1])) for i in range(4)]
    idx = min(range(4), key=lambda i: pts[i][0] + pts[i][1])
    pts = pts[idx:] + pts[:idx]
    return [(x / img_w, y / img_h) for x, y in pts]


def main() -> None:
    frames = sorted(GOOD_DIR.glob("*.jpg"))
    print(f"Good frames: {len(frames)}")

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
        ref = _ref_path(card_id)
        if not ref.exists():
            print(f"  SKIP (no reference): {card_id}")
            continue

        img = cv.imread(str(frame_path))
        ref_img = cv.imread(str(ref))
        if img is None or ref_img is None:
            print(f"  SKIP (unreadable): {frame_path.name}")
            continue

        try:
            result = align_img.align_images(img, ref_img)
        except Exception as e:
            print(f"  ERROR aligning {frame_path.name}: {e}")
            errors += 1
            done[frame_path.name] = {"error": True}
            continue

        if result.get("error"):
            print(f"  BAD {frame_path.name}: {result.get('error_message', '')}")
            errors += 1
            done[frame_path.name] = {"error": True}
            continue

        img_h, img_w = img.shape[:2]
        corners = _extract_corners(result["scene_corners"], img_w, img_h)

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bbox_pct = float((max(xs) - min(xs)) * (max(ys) - min(ys)))

        done[frame_path.name] = {
            "img_path": str(frame_path.relative_to(cfg.data_dir)),
            "card_id": card_id,
            "corner0_x": corners[0][0], "corner0_y": corners[0][1],
            "corner1_x": corners[1][0], "corner1_y": corners[1][1],
            "corner2_x": corners[2][0], "corner2_y": corners[2][1],
            "corner3_x": corners[3][0], "corner3_y": corners[3][1],
            "num_good_matches": int(result["num_good_matches"]),
            "matching_area_pct": bbox_pct,
        }

        if (i + 1) % 50 == 0:
            RESUME_JSON.write_text(json.dumps(done, indent=2))
            print(f"  [{i+1}/{len(todo)}] checkpoint ({len(done)} total)")

    RESUME_JSON.write_text(json.dumps(done, indent=2))

    rows = [v for v in done.values() if "corner0_x" in v]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} frames written to {OUTPUT_CSV}")
    print(f"Errors / bad matches: {errors}")


if __name__ == "__main__":
    main()
