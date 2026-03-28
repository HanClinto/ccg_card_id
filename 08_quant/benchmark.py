#!/usr/bin/env python3
"""Edge inference benchmarking tool for CCG Card ID.

Designed to run on Raspberry Pi with no PyTorch dependency.
Requires: pip install onnxruntime numpy opencv-python psutil

Runs the full pipeline (corner detection → dewarp → embedding → gallery lookup)
on each input image sequentially and reports per-image timing, memory, and
identification results.  Ground-truth card IDs are extracted automatically
from filenames that embed a Scryfall UUID.

Usage
-----
    # Full pipeline (raw frames — corner detect → dewarp → embed → lookup):
    python benchmark.py \\
        --detector   detector_e45.onnx \\
        --identifier identifier_e15.onnx \\
        --gallery    gallery.npz \\
        --images     /path/to/raw_frames/*.jpg \\
        [--dewarp-w 224] [--dewarp-h 312] \\
        [--top-k 3] \\
        [--csv results.csv]

    # Pre-cropped images (skip corner detection — embed → lookup directly):
    python benchmark.py \\
        --identifier identifier_e15.onnx \\
        --gallery    gallery.npz \\
        --images     /path/to/cropped/*.jpg \\
        --skip-detect \\
        [--csv results.csv]

    # Detector only (no identification):
    python benchmark.py \\
        --detector   detector_e45.onnx \\
        --images     /path/to/images/*.jpg

Gallery NPZ format (produced by the training pipeline)
------------------------------------------------------
    embeddings : (N, D) float32    L2-normalised gallery embeddings
    paths      : (N,)   str        image paths (UUID extracted from filename)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_UUID_RE       = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)

# ---------------------------------------------------------------------------
# Preprocessing helpers (no torchvision — pure numpy + cv2)
# ---------------------------------------------------------------------------

def _preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 → (1, 3, size, size) float32, ImageNet-normalised."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x   = rgb.astype(np.float32) / 255.0
    x   = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)   # (1,3,H,W)


def _dewarp(bgr: np.ndarray, corners_norm: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Perspective-warp card corners to a flat rectangle."""
    h, w = bgr.shape[:2]
    src  = corners_norm * np.array([w, h], dtype=np.float32)
    dst  = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h))


def _sort_corners(corners: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Sort (4,2) normalised corners to canonical CW order, shortest edge first."""
    corners = np.asarray(corners, dtype=np.float32)
    cx, cy  = corners.mean(axis=0)
    angles  = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
    corners = corners[np.argsort(angles)]
    scale   = np.array([img_w, img_h], dtype=np.float32)
    pts     = corners * scale
    edges   = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    return np.roll(corners, -int(np.argmin(edges)), axis=0)


def _extract_uuid(path_str: str) -> str | None:
    m = _UUID_RE.search(path_str)
    return m.group(0).lower() if m else None


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

class Gallery(NamedTuple):
    embeddings: np.ndarray   # (N, D) float32
    card_ids:   list[str]    # length N — Scryfall UUIDs

    @classmethod
    def load(cls, npz_path: Path) -> "Gallery":
        d    = np.load(str(npz_path), allow_pickle=True)
        embs = d["embeddings"].astype(np.float32)
        # Normalise in case the cached embeddings have drifted from unit norm
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(1e-8)
        embs  = embs / norms
        paths = d["paths"]
        ids   = [_extract_uuid(str(p)) or str(p) for p in paths]
        print(f"Gallery loaded: {len(ids):,} entries  (dim={embs.shape[1]})")
        return cls(embs, ids)

    def search(self, query_emb: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (card_id, cosine_similarity) pairs."""
        q    = query_emb.reshape(-1).astype(np.float32)
        sims = self.embeddings @ q                           # (N,)
        if k == 1:
            idx = [int(np.argmax(sims))]
        else:
            idx = np.argpartition(sims, -k)[-k:].tolist()
            idx = sorted(idx, key=lambda i: sims[i], reverse=True)
        return [(self.card_ids[i], float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# ONNX session wrappers
# ---------------------------------------------------------------------------

def _make_session(path: Path):
    import onnxruntime as ort  # noqa: PLC0415
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1   # single frame, no benefit from parallelism
    opts.intra_op_num_threads = 4   # use all Pi cores for within-op parallelism
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    return sess


class DetectorSession:
    def __init__(self, onnx_path: Path):
        self.sess       = _make_session(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        # Infer input size from model shape
        shape = self.sess.get_inputs()[0].shape
        self.input_size = int(shape[2]) if shape[2] != "unk" else 384
        out_names = [o.name for o in self.sess.get_outputs()]
        self._has_sharpness = "sharpness" in out_names
        print(f"Detector loaded: {onnx_path.name}  "
              f"input={self.input_size}×{self.input_size}  "
              f"sharpness={'yes' if self._has_sharpness else 'no'}")

    def run(self, bgr: np.ndarray) -> tuple[np.ndarray, float, float | None]:
        """Returns (corners (4,2), presence_prob, sharpness or None)."""
        x    = _preprocess(bgr, self.input_size)
        outs = self.sess.run(None, {self.input_name: x})
        corners_flat = outs[0].squeeze()           # (8,)
        presence     = float(1 / (1 + np.exp(-float(outs[1].squeeze()))))  # sigmoid
        sharpness    = float(outs[2].squeeze()) if self._has_sharpness else None
        corners      = np.clip(corners_flat, 0.0, 1.0).reshape(4, 2).astype(np.float32)
        corners      = _sort_corners(corners, bgr.shape[1], bgr.shape[0])
        return corners, presence, sharpness


class IdentifierSession:
    def __init__(self, onnx_path: Path):
        self.sess       = _make_session(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.input_size = int(shape[2]) if shape[2] != "unk" else 448
        print(f"Identifier loaded: {onnx_path.name}  input={self.input_size}×{self.input_size}")

    def run(self, bgr: np.ndarray) -> np.ndarray:
        """Returns L2-normalised embedding (D,)."""
        x   = _preprocess(bgr, self.input_size)
        out = self.sess.run(None, {self.input_name: x})[0]
        emb = out.squeeze().astype(np.float32)
        return emb / max(float(np.linalg.norm(emb)), 1e-8)


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def _run_image(
    path: Path,
    detector: DetectorSession | None,
    identifier: IdentifierSession | None,
    gallery: Gallery | None,
    dewarp_w: int,
    dewarp_h: int,
    top_k: int,
    skip_detect: bool = False,
) -> dict:
    result: dict = {"image": path.name}

    bgr = cv2.imread(str(path))
    if bgr is None:
        result["error"] = "unreadable"
        return result

    result["gt_card_id"] = _extract_uuid(path.name) or ""
    result["rss_mb"]     = round(_rss_mb(), 1)

    t0 = time.perf_counter()

    if skip_detect or detector is None:
        # Pre-cropped: no corner detection, pass full image to identifier
        card_bgr = bgr
        result["detect_ms"] = ""
        result["sharpness"] = ""
        result["presence"]  = ""
    else:
        corners, presence, sharpness = detector.run(bgr)
        t1 = time.perf_counter()
        result["detect_ms"] = round((t1 - t0) * 1000, 1)
        result["sharpness"] = round(sharpness, 4) if sharpness is not None else ""
        result["presence"]  = round(presence, 3)

        if identifier is None or gallery is None:
            result["total_ms"] = result["detect_ms"]
            return result

        # Dewarp
        td0    = time.perf_counter()
        card_bgr = _dewarp(bgr, corners, dewarp_w, dewarp_h)
        td1    = time.perf_counter()
        result["dewarp_ms"] = round((td1 - td0) * 1000, 1)

    if identifier is None or gallery is None:
        result["total_ms"] = result.get("detect_ms") or 0
        return result

    # Embed
    te0 = time.perf_counter()
    emb = identifier.run(card_bgr)
    te1 = time.perf_counter()
    result["identify_ms"] = round((te1 - te0) * 1000, 1)

    # Gallery lookup
    tl0     = time.perf_counter()
    matches = gallery.search(emb, k=top_k)
    tl1     = time.perf_counter()
    result["lookup_ms"] = round((tl1 - tl0) * 1000, 1)

    result["total_ms"]   = round((tl1 - t0) * 1000, 1)
    result["top1_card_id"] = matches[0][0] if matches else ""
    result["top1_score"]   = round(matches[0][1], 4) if matches else ""

    if top_k > 1:
        for rank, (cid, score) in enumerate(matches[1:], start=2):
            result[f"top{rank}_card_id"] = cid
            result[f"top{rank}_score"]   = round(score, 4)

    if result["gt_card_id"]:
        result["correct"] = int(result["top1_card_id"] == result["gt_card_id"])

    return result


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _summarise(rows: list[dict]) -> None:
    def _ms_stats(key: str) -> str:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return "–"
        arr = np.array(vals)
        return f"mean={arr.mean():.1f}  p50={np.percentile(arr,50):.1f}  p95={np.percentile(arr,95):.1f}"

    print("\n── Summary ───────────────────────────────────────────────")
    print(f"   images      : {len(rows)}")
    print(f"   detect_ms   : {_ms_stats('detect_ms')}")
    if any("dewarp_ms"  in r for r in rows):
        print(f"   dewarp_ms   : {_ms_stats('dewarp_ms')}")
    if any("identify_ms" in r for r in rows):
        print(f"   identify_ms : {_ms_stats('identify_ms')}")
    if any("lookup_ms"  in r for r in rows):
        print(f"   lookup_ms   : {_ms_stats('lookup_ms')}")
    if any("total_ms"   in r for r in rows):
        print(f"   total_ms    : {_ms_stats('total_ms')}")
    if any("rss_mb"     in r for r in rows):
        rss = [r["rss_mb"] for r in rows if isinstance(r.get("rss_mb"), float)]
        if rss:
            print(f"   RSS MB      : min={min(rss):.0f}  max={max(rss):.0f}")
    if any("correct"    in r for r in rows):
        correct = [r["correct"] for r in rows if "correct" in r]
        n = len(correct)
        top1 = sum(correct)
        print(f"   top-1 acc   : {top1}/{n}  ({100*top1/n:.1f}%)")
    if any("sharpness"  in r and r["sharpness"] != "" for r in rows):
        sh = [r["sharpness"] for r in rows if isinstance(r.get("sharpness"), float)]
        if sh:
            arr = np.array(sh)
            print(f"   sharpness   : mean={arr.mean()*100:.1f}%  "
                  f"p10={np.percentile(arr,10)*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CCG corner detection + identification on a set of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--detector",     type=Path, help="Detector ONNX model")
    parser.add_argument("--identifier",  type=Path, help="Identifier ONNX model (optional)")
    parser.add_argument("--gallery",     type=Path, help="Gallery NPZ (required with --identifier)")
    parser.add_argument("--images",      type=Path, nargs="+", required=True, help="Input image paths")
    parser.add_argument("--skip-detect", action="store_true",
                        help="Skip corner detection — embed pre-cropped images directly")
    parser.add_argument("--dewarp-w",    type=int, default=224,  help="Dewarp output width")
    parser.add_argument("--dewarp-h",    type=int, default=312,  help="Dewarp output height")
    parser.add_argument("--top-k",       type=int, default=3,    help="Gallery top-K results")
    parser.add_argument("--csv",         type=Path, help="Write per-image results to CSV")
    args = parser.parse_args()

    if not args.detector and not args.skip_detect:
        parser.error("--detector is required unless --skip-detect is given")
    if args.identifier and not args.gallery:
        parser.error("--gallery is required when --identifier is provided")

    # Expand globs if shell didn't (useful on Windows / explicit glob strings)
    import glob as _glob  # noqa: PLC0415
    image_paths: list[Path] = []
    for p in args.images:
        if "*" in str(p) or "?" in str(p):
            image_paths.extend(sorted(Path(x) for x in _glob.glob(str(p))))
        elif p.is_dir():
            for ext in ("jpg", "jpeg", "png", "JPG", "PNG"):
                image_paths.extend(sorted(p.glob(f"*.{ext}")))
        else:
            image_paths.append(p)

    if not image_paths:
        sys.exit("No images found.")

    print(f"\n── CCG Card ID Benchmark ─────────────────────────────────")
    print(f"   images      : {len(image_paths)}")

    detector   = DetectorSession(args.detector) if args.detector and not args.skip_detect else None
    if args.skip_detect:
        print("   (--skip-detect: embedding pre-cropped images directly)")
    identifier = IdentifierSession(args.identifier) if args.identifier else None
    gallery    = Gallery.load(args.gallery) if args.gallery else None

    # CSV header columns
    fixed_cols = ["image", "detect_ms", "sharpness", "presence",
                  "dewarp_ms", "identify_ms", "lookup_ms", "total_ms",
                  "rss_mb", "gt_card_id", "top1_card_id", "top1_score", "correct"]
    extra_cols = [f"top{k}_{s}" for k in range(2, args.top_k + 1)
                  for s in ("card_id", "score")]
    all_cols   = fixed_cols + extra_cols

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = csv.DictWriter(csv_file, fieldnames=all_cols, extrasaction="ignore") \
        if csv_file else None
    if csv_writer:
        csv_writer.writeheader()

    print(f"\n{'Image':<40} {'Det':>7} {'Shp':>6} {'ID':>7} {'Lkp':>6} {'Tot':>7}  {'Top-1 card_id':<36} {'Score':>6}")
    print("─" * 120)

    rows = []
    for path in image_paths:
        r = _run_image(path, detector, identifier, gallery,
                       args.dewarp_w, args.dewarp_h, args.top_k,
                       skip_detect=args.skip_detect)
        rows.append(r)

        det  = f"{r.get('detect_ms','–'):>7}"
        shp  = f"{r.get('sharpness', '–'):>6}"  if r.get('sharpness') != '' else f"{'–':>6}"
        ide  = f"{r.get('identify_ms','–'):>7}" if 'identify_ms' in r else f"{'–':>7}"
        lkp  = f"{r.get('lookup_ms','–'):>6}"   if 'lookup_ms'  in r else f"{'–':>6}"
        tot  = f"{r.get('total_ms','–'):>7}"
        cid  = r.get('top1_card_id','')
        sc   = f"{r.get('top1_score',''):>6}"
        ok   = " ✓" if r.get("correct") == 1 else (" ✗" if "correct" in r else "")

        print(f"{path.name:<40} {det} {shp} {ide} {lkp} {tot}  {cid:<36} {sc}{ok}")

        if csv_writer:
            csv_writer.writerow(r)

    if csv_file:
        csv_file.close()
        print(f"\n   CSV written → {args.csv}")

    _summarise(rows)


if __name__ == "__main__":
    main()
