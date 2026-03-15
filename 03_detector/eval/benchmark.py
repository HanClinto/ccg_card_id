#!/usr/bin/env python3
"""Evaluation harness for card corner detectors.

Loads labeled data from corners.csv + bad/ dir, runs each requested detector
on every image, and prints a summary table of CPE, PCK@5%, IoU, Detection Rate,
and False Positive Rate.

Usage:
  # Evaluate CannyPolyDetector on val split (default)
  python 03_detector/eval/benchmark.py

  # Evaluate both Canny and TinyCornerCNN
  python 03_detector/eval/benchmark.py \\
      --detectors canny,tinycornercnn \\
      --neural-checkpoint /path/to/corner_detector_tiny/last.pt

  # Evaluate all data (not just val split)
  python 03_detector/eval/benchmark.py --split all

  # Quick smoke-test: first 20 images only
  python 03_detector/eval/benchmark.py --limit 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from tqdm import tqdm

DETECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTOR_DIR))

from ccg_card_id.config import cfg  # noqa: E402
from base import CardDetector, DetectionResult  # noqa: E402
from detectors import CannyPolyDetector  # noqa: E402
from detectors.tiny_corner_cnn.dataset import load_dataset  # noqa: E402
from eval.metrics import corner_point_error, pck, quad_iou_exact, phash_distance  # noqa: E402

_PHASH_GOOD_THRESHOLD = 10  # Hamming distance ≤ 10 → close enough for real-world ID


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------

class FrameResult(NamedTuple):
    true_present: bool
    pred_present: bool
    cpe: float | None        # None for negatives or false negatives
    pck_5: float | None
    iou: float | None
    phash_dist: int | None   # Hamming distance between dewarped pred and ref; None if N/A
    img_w: int
    img_h: int


# ---------------------------------------------------------------------------
# Result cache (deterministic detectors like Canny are cached by dataset hash)
# ---------------------------------------------------------------------------

def _dataset_hash(corners_csv: Path, split: str, limit: int | None) -> str:
    key = f"{corners_csv}:{split}:{limit}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, detector_name: str, dataset_hash: str) -> Path:
    safe_name = detector_name.replace(" ", "_").replace("/", "_")
    return cache_dir / f"{safe_name}__{dataset_hash}.json"


def _save_cache(path: Path, results: list[FrameResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([list(r) for r in results], f)


def _load_cache(path: Path) -> list[FrameResult] | None:
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return [FrameResult(*r) for r in raw]


# ---------------------------------------------------------------------------
# Evaluate one detector over a list of rows
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector: CardDetector,
    rows: list[dict],
    data_dir: Path,
    catalog_root: Path,
    limit: int | None = None,
    cache_dir: Path | None = None,
    dataset_hash: str | None = None,
    cacheable: bool = False,
) -> list[FrameResult]:
    # Try cache first (only for deterministic detectors)
    if cacheable and cache_dir is not None and dataset_hash is not None:
        cp = _cache_path(cache_dir, detector.name, dataset_hash)
        cached = _load_cache(cp)
        if cached is not None:
            print(f"  [{detector.name}] loaded {len(cached)} results from cache")
            return cached

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
            # pHash: dewarp predicted crop and compare against Scryfall reference
            card_id   = row.get("card_id")
            if card_id:
                ref_path = catalog_root / card_id[0] / card_id[1] / f"{card_id}.png"
                phash_val = phash_distance(img, pred_corners, ref_path)
            else:
                phash_val = None
        else:
            cpe_val = pck_val = iou_val = phash_val = None

        results.append(FrameResult(
            true_present=true_present,
            pred_present=pred_present,
            cpe=cpe_val,
            pck_5=pck_val,
            iou=iou_val,
            phash_dist=phash_val,
            img_w=w,
            img_h=h,
        ))

    if cacheable and cache_dir is not None and dataset_hash is not None:
        cp = _cache_path(cache_dir, detector.name, dataset_hash)
        _save_cache(cp, results)
        print(f"  [{detector.name}] cached {len(results)} results → {cp.name}")

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

    # CPE, PCK, IoU, pHash: mean over frames where both true and pred are positive
    cpe_vals   = [r.cpe       for r in results if r.cpe       is not None]
    pck_vals   = [r.pck_5     for r in results if r.pck_5     is not None]
    iou_vals   = [r.iou       for r in results if r.iou       is not None]
    phash_vals = [r.phash_dist for r in results if r.phash_dist is not None]

    phash_good_rate = (
        float(sum(1 for d in phash_vals if d <= _PHASH_GOOD_THRESHOLD) / len(phash_vals))
        if phash_vals else float("nan")
    )

    return {
        "n_total":        len(results),
        "n_positive":     len(positives),
        "n_negative":     len(negatives),
        "cpe":            float(np.mean(cpe_vals))   if cpe_vals   else float("nan"),
        "pck_5":          float(np.mean(pck_vals))   if pck_vals   else float("nan"),
        "iou":            float(np.mean(iou_vals))   if iou_vals   else float("nan"),
        "phash_mean":     float(np.mean(phash_vals)) if phash_vals else float("nan"),
        "phash_good_rate": phash_good_rate,
        "det_rate":       det_rate,
        "fpr":            fpr,
    }


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_table(rows: list[tuple[str, dict]]) -> None:
    """Print a formatted summary table.

    Args:
        rows: list of (detector_name, metrics_dict) pairs.
    """
    header = (
        f"{'Detector':<28} | {'CPE↓':>7} | {'PCK@5%↑':>8} | {'IoU↑':>6} | "
        f"{'pHash↓':>7} | {'ID-able↑':>9} | {'Det.Rate↑':>10} | {'FPR↓':>6}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, m in rows:
        cpe_s    = f"{m['cpe']:.4f}"             if not _isnan(m['cpe'])             else "  N/A "
        pck_s    = f"{m['pck_5']*100:.1f}%"      if not _isnan(m['pck_5'])           else "  N/A "
        iou_s    = f"{m['iou']:.4f}"             if not _isnan(m['iou'])             else "  N/A "
        ph_s     = f"{m['phash_mean']:.1f}"      if not _isnan(m['phash_mean'])      else "  N/A "
        id_s     = f"{m['phash_good_rate']*100:.1f}%" if not _isnan(m['phash_good_rate']) else "  N/A "
        det_s    = f"{m['det_rate']*100:.1f}%"   if not _isnan(m['det_rate'])        else "  N/A "
        fpr_s    = f"{m['fpr']*100:.1f}%"        if not _isnan(m['fpr'])             else "  N/A "
        print(
            f"{name:<28} | {cpe_s:>7} | {pck_s:>8} | {iou_s:>6} | "
            f"{ph_s:>7} | {id_s:>9} | {det_s:>10} | {fpr_s:>6}"
        )
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
    data_dir     = args.data_dir
    corners_csv  = args.corners_csv
    neg_dir      = args.neg_dir
    catalog_root = data_dir / "catalog" / "scryfall" / "images" / "png" / "front"

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
        elif det_name == "tinycornercnn":
            if args.neural_checkpoint is None:
                print("WARNING: --neural-checkpoint not provided; skipping TinyCornerCNN.")
                continue
            if not args.neural_checkpoint.exists():
                print(f"WARNING: checkpoint not found: {args.neural_checkpoint}; skipping.")
                continue
            _tcc_dir = Path(__file__).resolve().parents[1] / "detectors" / "tiny_corner_cnn"
            sys.path.insert(0, str(_tcc_dir))
            from predict import NeuralCornerDetectorInference  # noqa: E402
            device = "mps" if _mps_available() else ("cuda" if _cuda_available() else "cpu")
            detectors.append(
                NeuralCornerDetectorInference(args.neural_checkpoint, device=device)
            )
        else:
            print(f"WARNING: unknown detector '{det_name}' — skipping. Valid: canny, tinycornercnn")

    if not detectors:
        print("No detectors to evaluate. Use --detectors canny or --detectors canny,tinycornercnn")
        return

    # Evaluate
    cache_dir = args.cache_dir
    ds_hash = _dataset_hash(corners_csv, args.split, args.limit)
    # Deterministic detectors whose results don't depend on a checkpoint
    _CACHEABLE = {"canny", "cannypolydetector"}  # class names lowercased, no spaces

    table_rows = []
    for detector in detectors:
        is_cacheable = detector.name.lower().replace(" ", "") in _CACHEABLE
        frame_results = evaluate_detector(
            detector, rows, data_dir, catalog_root,
            limit=args.limit,
            cache_dir=cache_dir,
            dataset_hash=ds_hash,
            cacheable=is_cacheable,
        )
        metrics = aggregate(frame_results)
        table_rows.append((detector.name, metrics))
        print(
            f"  {detector.name}: "
            f"n={metrics['n_total']} "
            f"(+{metrics['n_positive']}/-{metrics['n_negative']}) "
            f"CPE={metrics['cpe']:.4f} "
            f"pHash={metrics['phash_mean']:.1f} "
            f"ID-able={metrics['phash_good_rate']*100:.1f}% "
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
        help="Comma-separated list of detectors to evaluate: canny, tinycornercnn (default: canny)",
    )
    p.add_argument(
        "--neural-checkpoint", type=Path, default=None,
        help="Path to trained TinyCornerCNN checkpoint (required for --detectors tinycornercnn)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate first N images per split (useful for quick smoke-tests)",
    )
    p.add_argument(
        "--split", choices=["val", "all"], default="val",
        help="Which data split to evaluate (default: val)",
    )
    p.add_argument(
        "--cache-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "eval" / "cache",
        help="Directory for caching deterministic detector results (default: eval/cache/)",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
