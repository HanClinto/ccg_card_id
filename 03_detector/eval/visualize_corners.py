#!/usr/bin/env python3
"""Render corner predictions from Canny, Neural 1-stage, and Neural 2-stage onto images.

Usage:
    python 03_detector/eval/visualize_corners.py \
        --checkpoint /path/to/last.pt \
        [--corners-csv PATH] [--neg-dir PATH] [--n 8] [--out-dir debug_renders] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

DETECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTOR_DIR))

from ccg_card_id.config import cfg
from base import sort_corners_canonical
from detectors import CannyPolyDetector
from detectors.tiny_corner_cnn.dataset import load_dataset
from detectors.tiny_corner_cnn.predict import NeuralCornerDetectorInference
from eval.two_stage_test import detect_two_stage


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_COLORS = {
    "gt":      (0,   200,   0),   # green  — ground truth
    "canny":   (0,   165, 255),   # orange — Canny
    "neural1": (255,  50,  50),   # blue   — Neural 1-stage
    "neural2": (180,   0, 255),   # purple — Neural 2-stage
}

_LABELS = {
    "gt":      "GT",
    "canny":   "Canny",
    "neural1": "Neural-1",
    "neural2": "Neural-2",
}


def _draw_quad(
    img: np.ndarray,
    corners: np.ndarray | None,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
    dot_radius: int = 6,
) -> None:
    """Draw quad edges + numbered corner dots onto img in-place."""
    if corners is None:
        return
    h, w = img.shape[:2]
    pts = (corners * np.array([w, h])).astype(np.int32)

    # Draw edges
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, thickness, cv2.LINE_AA)

    # Draw corner dots + indices
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), dot_radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            img, str(i), (int(pt[0]) + dot_radius + 2, int(pt[1]) - dot_radius),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    # Draw label near corner 0
    cv2.putText(
        img, label, (int(pts[0][0]) + 4, int(pts[0][1]) - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


def render_sample(
    img: np.ndarray,
    gt_corners: np.ndarray | None,
    canny_corners: np.ndarray | None,
    neural1_corners: np.ndarray | None,
    neural2_corners: np.ndarray | None,
    title: str = "",
) -> np.ndarray:
    """Return an annotated copy of img with all four corner sets drawn."""
    out = img.copy()

    _draw_quad(out, gt_corners,      _COLORS["gt"],      _LABELS["gt"],      thickness=2)
    _draw_quad(out, canny_corners,   _COLORS["canny"],   _LABELS["canny"],   thickness=2)
    _draw_quad(out, neural1_corners, _COLORS["neural1"], _LABELS["neural1"], thickness=2)
    _draw_quad(out, neural2_corners, _COLORS["neural2"], _LABELS["neural2"], thickness=2)

    # Legend strip at top
    legend_h = 28
    legend = np.zeros((legend_h, out.shape[1], 3), dtype=np.uint8)
    x = 8
    for key in ("gt", "canny", "neural1", "neural2"):
        text = _LABELS[key]
        color = _COLORS[key]
        cv2.rectangle(legend, (x, 6), (x + 16, 22), color, -1)
        cv2.putText(legend, text, (x + 20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        x += 110

    if title:
        cv2.putText(legend, title, (x + 20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return np.vstack([legend, out])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _data_dir = cfg.data_dir
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir",   type=Path, default=_data_dir)
    p.add_argument("--corners-csv", type=Path,
                   default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/corners.csv")
    p.add_argument("--neg-dir", type=Path,
                   default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/bad")
    p.add_argument("--split", choices=["val", "all"], default="all")
    p.add_argument("--n", type=int, default=8, help="Number of positive frames to render")
    p.add_argument("--out-dir", type=Path, default=Path("debug_renders"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pad", type=float, default=0.25)
    p.add_argument("--max-dim", type=int, default=900,
                   help="Downsample longest edge to this before saving")
    args = p.parse_args()

    data_dir = args.data_dir

    train_rows, val_rows = load_dataset(args.corners_csv, args.neg_dir, data_dir)
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
        img_path = data_dir / row["img_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] {img_path}")
            continue
        h, w = img.shape[:2]

        gt_corners = row["corners"]

        canny_result   = canny.detect(img)
        neural1_result = neural.detect(img)
        neural2_result = detect_two_stage(neural, img, pad=args.pad)

        annotated = render_sample(
            img,
            gt_corners,
            canny_result.corners   if canny_result.card_present   else None,
            neural1_result.corners if neural1_result.card_present else None,
            neural2_result.corners if neural2_result.card_present else None,
            title=img_path.name,
        )

        # Optionally downscale for easier viewing
        ah, aw = annotated.shape[:2]
        if max(ah, aw) > args.max_dim:
            scale = args.max_dim / max(ah, aw)
            annotated = cv2.resize(annotated, (int(aw * scale), int(ah * scale)),
                                   interpolation=cv2.INTER_AREA)

        out_path = args.out_dir / f"{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  [{i+1}/{len(sample_rows)}] {out_path.name}")

    print(f"\nSaved {len(sample_rows)} renders to: {args.out_dir}/")


if __name__ == "__main__":
    main()
