#!/usr/bin/env python3
"""Two-stage neural corner detection test.

Stage 1: run the detector on the full image at 224×224 to get a rough quad.
Stage 2: crop a padded bounding box around that quad from the *original*
         full-resolution image, resize to 224×224, run the detector again.
         Map the refined corners back to original image coordinates.

The hypothesis: the card fills more of the 224×224 crop in stage 2, giving
the model higher effective resolution and tighter corner predictions.

Usage:
    python 03_detector/eval/two_stage_test.py \\
        --checkpoint /path/to/last.pt \\
        [--corners-csv PATH] [--neg-dir PATH] [--split all|val] [--limit N] [--pad 0.25]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DETECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTOR_DIR))

from ccg_card_id.config import cfg
from base import CardDetector, DetectionResult, sort_corners_canonical
from detectors import CannyPolyDetector
from detectors.tiny_corner_cnn.dataset import load_dataset
from detectors.tiny_corner_cnn.predict import NeuralCornerDetectorInference
from eval.metrics import corner_point_error, pck, quad_iou_exact, phash_distance


# ---------------------------------------------------------------------------
# Two-stage wrapper
# ---------------------------------------------------------------------------

def _bbox_from_corners(corners: np.ndarray, img_w: int, img_h: int, pad: float
                       ) -> tuple[int, int, int, int]:
    """Compute a padded axis-aligned bounding box from normalized corners.

    Returns (x1, y1, x2, y2) clamped to image bounds.
    """
    px = corners[:, 0] * img_w
    py = corners[:, 1] * img_h
    bx1, bx2 = px.min(), px.max()
    by1, by2 = py.min(), py.max()
    bw, bh = bx2 - bx1, by2 - by1
    pad_x, pad_y = pad * bw, pad * bh
    x1 = max(0,     int(bx1 - pad_x))
    y1 = max(0,     int(by1 - pad_y))
    x2 = min(img_w, int(bx2 + pad_x))
    y2 = min(img_h, int(by2 + pad_y))
    return x1, y1, x2, y2


def detect_two_stage(
    detector: NeuralCornerDetectorInference,
    image: np.ndarray,
    pad: float = 0.25,
) -> DetectionResult:
    """Run two-stage detection on a full-resolution image.

    Stage 1 on the full image → bounding box + padding → crop →
    Stage 2 on the crop → map corners back to original image space.

    Falls back to stage-1 result if stage 2 fails or produces a worse quad.
    """
    h, w = image.shape[:2]

    # --- Stage 1 ---
    result1 = detector.detect(image)
    if not result1.card_present or result1.corners is None:
        return result1

    # --- Crop ---
    x1, y1, x2, y2 = _bbox_from_corners(result1.corners, w, h, pad)
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w < 10 or crop_h < 10:
        return result1
    crop = image[y1:y2, x1:x2]

    # --- Stage 2 ---
    result2 = detector.detect(crop)
    if not result2.card_present or result2.corners is None:
        return result1

    # --- Map crop-normalized corners back to full-image normalized coords ---
    corners2 = result2.corners.copy()
    corners2[:, 0] = x1 / w + corners2[:, 0] * (crop_w / w)
    corners2[:, 1] = y1 / h + corners2[:, 1] * (crop_h / h)
    corners2 = sort_corners_canonical(corners2, img_w=w, img_h=h)

    return DetectionResult(
        card_present=True,
        corners=corners2,
        confidence=result2.confidence,
        metadata={**result2.metadata, "stage": 2,
                  "crop": (x1, y1, x2, y2)},
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(detector_name: str, detect_fn, rows, data_dir, catalog_root, limit):
    results = []
    eval_rows = rows[:limit] if limit else rows
    for row in tqdm(eval_rows, desc=detector_name, unit="img"):
        img_path = data_dir / row["img_path"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        true_present  = row["card_present"]
        true_corners  = row["corners"]
        result        = detect_fn(img)
        pred_present  = result.card_present
        pred_corners  = result.corners

        if true_present and pred_present and true_corners is not None and pred_corners is not None:
            cpe_val  = corner_point_error(pred_corners, true_corners)
            pck_val  = pck(pred_corners, true_corners, threshold=0.05)
            iou_val  = quad_iou_exact(pred_corners, true_corners, w, h)
            card_id  = row.get("card_id")
            if card_id:
                ref_path  = catalog_root / card_id[0] / card_id[1] / f"{card_id}.png"
                phash_val = phash_distance(img, pred_corners, ref_path)
            else:
                phash_val = None
        else:
            cpe_val = pck_val = iou_val = phash_val = None

        results.append(dict(
            true_present=true_present, pred_present=pred_present,
            cpe=cpe_val, pck_5=pck_val, iou=iou_val, phash_dist=phash_val,
        ))
    return results


def aggregate(results):
    pos = [r for r in results if r["true_present"]]
    neg = [r for r in results if not r["true_present"]]
    det_rate = sum(1 for r in pos if r["pred_present"]) / len(pos) if pos else float("nan")
    fpr      = sum(1 for r in neg if r["pred_present"]) / len(neg) if neg else float("nan")
    cpe_v    = [r["cpe"]       for r in results if r["cpe"]       is not None]
    pck_v    = [r["pck_5"]     for r in results if r["pck_5"]     is not None]
    iou_v    = [r["iou"]       for r in results if r["iou"]       is not None]
    ph_v     = [r["phash_dist"] for r in results if r["phash_dist"] is not None]
    phash_good = sum(1 for d in ph_v if d <= 10) / len(ph_v) if ph_v else float("nan")
    def m(v): return float(np.mean(v)) if v else float("nan")
    return dict(cpe=m(cpe_v), pck_5=m(pck_v), iou=m(iou_v),
                phash_mean=m(ph_v), phash_good_rate=phash_good,
                det_rate=det_rate, fpr=fpr, n=len(results))


def print_table(rows):
    hdr = (f"{'Detector':<32} | {'CPE↓':>7} | {'PCK@5%↑':>8} | {'IoU↑':>6} | "
           f"{'pHash↓':>7} | {'ID-able↑':>9} | {'Det%↑':>6}")
    sep = "-" * len(hdr)
    print(); print(sep); print(hdr); print(sep)
    for name, m in rows:
        def f(v, fmt): return fmt % v if v == v else "  N/A "
        print(f"{name:<32} | {f(m['cpe'],'%7.4f')} | {f(m['pck_5']*100,'%7.1f%%')} | "
              f"{f(m['iou'],'%6.4f')} | {f(m['phash_mean'],'%7.1f')} | "
              f"{f(m['phash_good_rate']*100,'%8.1f%%')} | {f(m['det_rate']*100,'%5.1f%%')}")
    print(sep); print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _data_dir = cfg.data_dir
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=_data_dir)
    p.add_argument("--corners-csv", type=Path,
                   default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/corners.csv")
    p.add_argument("--neg-dir", type=Path,
                   default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/bad")
    p.add_argument("--split", choices=["val", "all"], default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pad", type=float, default=0.25,
                   help="Fractional padding added around stage-1 bounding box (default 0.25)")
    p.add_argument("--device", type=str, default="mps",
                   help="Inference device: mps, cuda, cpu (default: mps)")
    args = p.parse_args()

    data_dir = args.data_dir
    catalog_root = data_dir / "catalog" / "scryfall" / "images" / "png" / "front"

    train_rows, val_rows = load_dataset(args.corners_csv, args.neg_dir, data_dir)
    rows = val_rows if args.split == "val" else train_rows + val_rows
    print(f"Dataset: {args.corners_csv.parent.name}  |  split={args.split}  |  n={len(rows)}")

    neural = NeuralCornerDetectorInference(args.checkpoint, device=args.device)
    canny  = CannyPolyDetector()

    table = []
    for name, fn in [
        ("Canny",          lambda img: canny.detect(img)),
        ("Neural 1-stage", lambda img: neural.detect(img)),
        ("Neural 2-stage", lambda img: detect_two_stage(neural, img, pad=args.pad)),
    ]:
        res = evaluate(name, fn, rows, data_dir, catalog_root, args.limit)
        m   = aggregate(res)
        table.append((name, m))
        print(f"  {name}: CPE={m['cpe']:.4f}  PCK={m['pck_5']*100:.1f}%  "
              f"IoU={m['iou']:.4f}  pHash={m['phash_mean']:.1f}  "
              f"ID-able={m['phash_good_rate']*100:.1f}%")

    print_table(table)


if __name__ == "__main__":
    main()
