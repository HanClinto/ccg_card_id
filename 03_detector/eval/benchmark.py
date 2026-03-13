#!/usr/bin/env python3
"""Evaluation harness for card corner detectors.

Loads labeled data from corners.csv + bad/ dir, runs each requested detector
on every image, and prints a summary table of CPE, PCK@5%, IoU, Detection Rate,
and False Positive Rate.

Usage:
  # Evaluate CannyPolyDetector on val split (default)
  python 03_detector/eval/benchmark.py

  # Evaluate both Canny and neural
  python 03_detector/eval/benchmark.py \\
      --detectors canny,neural \\
      --neural-checkpoint /path/to/corner_detector/last.pt

  # Evaluate all data (not just val split)
  python 03_detector/eval/benchmark.py --split all

  # Quick smoke-test: first 20 images only
  python 03_detector/eval/benchmark.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DETECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTOR_DIR))

from ccg_card_id.config import cfg  # noqa: E402
from base import CardDetector, DetectionResult  # noqa: E402
from detectors import CannyPolyDetector  # noqa: E402
from detectors.neural.dataset import load_dataset  # noqa: E402
from eval.metrics import corner_point_error, pck, quad_iou_exact  # noqa: E402


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------

class FrameResult(NamedTuple):
    true_present: bool
    pred_present: bool
    cpe: float | None        # None for negatives or false negatives
    pck_5: float | None
    iou: float | None
    img_w: int
    img_h: int


# ---------------------------------------------------------------------------
# Evaluate one detector over a list of rows
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector: CardDetector,
    rows: list[dict],
    data_dir: Path,
    limit: int | None = None,
) -> list[FrameResult]:
    results: list[FrameResult] = []
    eval_rows = rows[:limit] if limit else rows

    for row in tqdm(eval_rows, desc=detector.name, unit="img"):
        img_path = data_dir / row["img_path"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        true_present: bool = row["card_present"]
        true_corners: np.ndarray | None = row["corners"]

        result: DetectionResult = detector.detect(img)

        pred_present = result.card_present
        pred_corners = result.corners

        # Compute metrics only when both true and pred have corners
        if true_present and pred_present and true_corners is not None and pred_corners is not None:
            cpe_val   = corner_point_error(pred_corners, true_corners)
            pck_val   = pck(pred_corners, true_corners, threshold=0.05)
            iou_val   = quad_iou_exact(pred_corners, true_corners, w, h)
        else:
            cpe_val = pck_val = iou_val = None

        results.append(FrameResult(
            true_present=true_present,
            pred_present=pred_present,
            cpe=cpe_val,
            pck_5=pck_val,
            iou=iou_val,
            img_w=w,
            img_h=h,
        ))

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(results: list[FrameResult]) -> dict:
    """Compute summary metrics from a list of FrameResults."""
    positives = [r for r in results if r.true_present]
    negatives = [r for r in results if not r.true_present]

    # Detection rate: TP / (TP + FN) on positive frames
    if positives:
        det_rate = sum(1 for r in positives if r.pred_present) / len(positives)
    else:
        det_rate = float("nan")

    # False positive rate: FP / negatives
    if negatives:
        fpr = sum(1 for r in negatives if r.pred_present) / len(negatives)
    else:
        fpr = float("nan")

    # CPE, PCK, IoU: mean over frames where both true and pred are positive
    cpe_vals  = [r.cpe  for r in results if r.cpe  is not None]
    pck_vals  = [r.pck_5 for r in results if r.pck_5 is not None]
    iou_vals  = [r.iou  for r in results if r.iou  is not None]

    return {
        "n_total":    len(results),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "cpe":        float(np.mean(cpe_vals))  if cpe_vals  else float("nan"),
        "pck_5":      float(np.mean(pck_vals))  if pck_vals  else float("nan"),
        "iou":        float(np.mean(iou_vals))  if iou_vals  else float("nan"),
        "det_rate":   det_rate,
        "fpr":        fpr,
    }


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_table(rows: list[tuple[str, dict]]) -> None:
    """Print a formatted summary table.

    Args:
        rows: list of (detector_name, metrics_dict) pairs.
    """
    header = f"{'Detector':<28} | {'CPE↓':>7} | {'PCK@5%↑':>8} | {'IoU↑':>6} | {'Det.Rate↑':>10} | {'FPR↓':>6}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, m in rows:
        cpe_s      = f"{m['cpe']:.4f}"      if not _isnan(m['cpe'])      else "  N/A "
        pck_s      = f"{m['pck_5']*100:.1f}%"  if not _isnan(m['pck_5'])    else "  N/A "
        iou_s      = f"{m['iou']:.4f}"      if not _isnan(m['iou'])      else "  N/A "
        det_s      = f"{m['det_rate']*100:.1f}%"  if not _isnan(m['det_rate']) else "  N/A "
        fpr_s      = f"{m['fpr']*100:.1f}%"  if not _isnan(m['fpr'])      else "  N/A "
        print(f"{name:<28} | {cpe_s:>7} | {pck_s:>8} | {iou_s:>6} | {det_s:>10} | {fpr_s:>6}")
    print(sep)
    print()


def _isnan(v: float) -> bool:
    import math
    try:
        return math.isnan(v)
    except TypeError:
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    data_dir    = args.data_dir
    corners_csv = args.corners_csv
    neg_dir     = args.neg_dir

    if not corners_csv.exists():
        raise FileNotFoundError(f"corners.csv not found: {corners_csv}")

    train_rows, val_rows = load_dataset(corners_csv, neg_dir, data_dir)

    if args.split == "val":
        rows = val_rows
        print(f"Evaluating on val split: {len(rows)} images")
    else:
        rows = train_rows + val_rows
        print(f"Evaluating on all data: {len(rows)} images")

    # Build requested detectors
    detectors: list[CardDetector] = []
    requested = [d.strip().lower() for d in args.detectors.split(",")]

    for det_name in requested:
        if det_name == "canny":
            detectors.append(CannyPolyDetector())
        elif det_name == "neural":
            if args.neural_checkpoint is None:
                print("WARNING: --neural-checkpoint not provided; skipping neural detector.")
                continue
            if not args.neural_checkpoint.exists():
                print(f"WARNING: neural checkpoint not found: {args.neural_checkpoint}; skipping.")
                continue
            # Import here so neural deps are only needed when requested
            _neural_dir = Path(__file__).resolve().parents[1] / "detectors" / "neural"
            sys.path.insert(0, str(_neural_dir))
            from predict import NeuralCornerDetectorInference  # noqa: E402
            device = "mps" if _mps_available() else ("cuda" if _cuda_available() else "cpu")
            detectors.append(
                NeuralCornerDetectorInference(args.neural_checkpoint, device=device)
            )
        else:
            print(f"WARNING: unknown detector '{det_name}' — skipping. Valid: canny, neural")

    if not detectors:
        print("No detectors to evaluate. Use --detectors canny or --detectors canny,neural")
        return

    # Evaluate
    table_rows = []
    for detector in detectors:
        frame_results = evaluate_detector(detector, rows, data_dir, limit=args.limit)
        metrics = aggregate(frame_results)
        table_rows.append((detector.name, metrics))
        print(
            f"  {detector.name}: "
            f"n={metrics['n_total']} "
            f"(+{metrics['n_positive']}/-{metrics['n_negative']}) "
            f"CPE={metrics['cpe']:.4f} "
            f"det_rate={metrics['det_rate']*100:.1f}%"
        )

    print_table(table_rows)


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    _data_dir = cfg.data_dir
    p = argparse.ArgumentParser(description="Benchmark card corner detectors")
    p.add_argument(
        "--data-dir", type=Path, default=_data_dir,
        help="Root data directory (default: from cfg)",
    )
    p.add_argument(
        "--corners-csv", type=Path,
        default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/corners.csv",
        help="Path to corners.csv",
    )
    p.add_argument(
        "--neg-dir", type=Path,
        default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/bad",
        help="Directory of hard-negative frames",
    )
    p.add_argument(
        "--detectors", type=str, default="canny",
        help="Comma-separated list of detectors to evaluate: canny, neural (default: canny)",
    )
    p.add_argument(
        "--neural-checkpoint", type=Path, default=None,
        help="Path to trained neural detector checkpoint (required for --detectors neural)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate first N images per split (useful for quick smoke-tests)",
    )
    p.add_argument(
        "--split", choices=["val", "all"], default="val",
        help="Which data split to evaluate (default: val)",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
