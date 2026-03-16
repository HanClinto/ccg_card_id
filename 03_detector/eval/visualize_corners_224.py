#!/usr/bin/env python3
"""Render corner predictions on the 224×224 input the model actually sees.

Both ground-truth and predicted corners are in normalised [0,1] space and
remain valid across any resize (full-image, no crop), so they can be drawn
directly on the 224×224 resized image.

Usage:
    python 03_detector/eval/visualize_corners_224.py \
        --checkpoint /path/to/last.pt \
        [--corners-csv PATH] [--neg-dir PATH] [--n 10] [--out-dir debug_renders_224] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DETECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTOR_DIR))

from ccg_card_id.config import cfg
from base import sort_corners_canonical
from detectors import CannyPolyDetector
from detectors.tiny_corner_cnn.dataset import load_dataset
from detectors.tiny_corner_cnn.predict import NeuralCornerDetectorInference, INPUT_SIZE

INPUT_W = INPUT_H = INPUT_SIZE  # 224


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_STYLES = {
    "gt":      dict(color=(0,   200,  0),  label="GT",       dot=5,  thick=2),
    "canny":   dict(color=(0,   165, 255), label="Canny",    dot=5,  thick=2),
    "neural1": dict(color=(255,  50,  50), label="Neural-1", dot=5,  thick=2),
}


def _draw_quad(img, corners, style):
    h, w = img.shape[:2]
    if corners is None:
        return
    pts = (corners * np.array([w, h])).astype(np.int32)
    color, label, dot, thick = style["color"], style["label"], style["dot"], style["thick"]
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thick, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), dot, color, -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (pt[0] + dot + 2, pt[1] - dot),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (pts[0][0] + 3, pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _legend_strip(width: int) -> np.ndarray:
    strip = np.zeros((24, width, 3), dtype=np.uint8)
    x = 6
    for key, s in _STYLES.items():
        cv2.rectangle(strip, (x, 5), (x + 14, 19), s["color"], -1)
        cv2.putText(strip, s["label"], (x + 17, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, s["color"], 1, cv2.LINE_AA)
        x += 90
    return strip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _data_dir = cfg.data_dir
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir",   type=Path, default=_data_dir)
    p.add_argument("--corners-csv", type=Path,
                   default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/corners.csv")
    p.add_argument("--neg-dir", type=Path,
                   default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/bad")
    p.add_argument("--split",   choices=["val", "all"], default="all")
    p.add_argument("--n",       type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("debug_renders_224"))
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--upscale", type=int, default=3,
                   help="Upscale factor for saved image (3 → 672×672, easier to read)")
    args = p.parse_args()

    train_rows, val_rows = load_dataset(args.corners_csv, args.neg_dir, args.data_dir)
    rows = val_rows if args.split == "val" else train_rows + val_rows
    pos_rows = [r for r in rows if r["card_present"] and r["corners"] is not None]
    print(f"Positive frames available: {len(pos_rows)}")

    rng = random.Random(args.seed)
    sample_rows = rng.sample(pos_rows, min(args.n, len(pos_rows)))

    device = "mps"
    neural = NeuralCornerDetectorInference(args.checkpoint, device=device)
    canny  = CannyPolyDetector()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(sample_rows):
        img_path = args.data_dir / row["img_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [SKIP] {img_path}")
            continue

        # --- Resize to 224×224 exactly as the model's _PREPROCESS does ---
        img_pil  = Image.fromarray(img_bgr[:, :, ::-1])   # BGR→RGB PIL
        img_224  = img_pil.resize((INPUT_W, INPUT_H), Image.BILINEAR)
        img_224_bgr = np.array(img_224)[:, :, ::-1]       # back to BGR for cv2

        # --- Run detectors on ORIGINAL image (corners are normalised, still valid) ---
        gt_corners     = row["corners"]
        neural_result  = neural.detect(img_bgr)
        canny_result   = canny.detect(img_bgr)

        # --- Annotate the 224×224 image ---
        canvas = img_224_bgr.copy()
        _draw_quad(canvas, gt_corners,                                   _STYLES["gt"])
        _draw_quad(canvas, canny_result.corners  if canny_result.card_present  else None, _STYLES["canny"])
        _draw_quad(canvas, neural_result.corners if neural_result.card_present else None, _STYLES["neural1"])

        # Upscale for readability (nearest-neighbour to keep pixel-exact lines)
        if args.upscale > 1:
            scale = args.upscale
            canvas = cv2.resize(canvas, (INPUT_W * scale, INPUT_H * scale),
                                interpolation=cv2.INTER_NEAREST)

        # Stick a legend strip on top
        legend = _legend_strip(canvas.shape[1])
        canvas = np.vstack([legend, canvas])

        # Title bar with filename
        title_h = 18
        title = np.zeros((title_h, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(title, img_path.name, (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        canvas = np.vstack([title, canvas])

        out_name = f"{i:02d}_{img_path.stem}.jpg"
        out_path = args.out_dir / out_name
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  [{i+1}/{len(sample_rows)}] {out_name}")

    print(f"\nSaved {len(sample_rows)} renders ({INPUT_W*args.upscale}px) → {args.out_dir}/")


if __name__ == "__main__":
    main()
