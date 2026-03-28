#!/usr/bin/env python3
"""Apply static INT8 post-training quantization to exported ONNX models.

Run on the dev machine after export_onnx.py.
Requires: pip install onnxruntime onnx

Static quantization needs a calibration dataset to measure activation ranges.
Provide a directory of representative images (50–200 is usually sufficient).

Usage
-----
    # Quantize the detector
    python 08_quant/quantize_int8.py \\
        --model $DATA/models_onnx/detector_*.onnx \\
        --calib-images $DATA/datasets/packopening/frames_sample/ \\
        --input-size 384

    # Quantize the identifier
    python 08_quant/quantize_int8.py \\
        --model $DATA/models_onnx/identifier_*.onnx \\
        --calib-images $DATA/catalog/scryfall/images/png/front/0/ \\
        --input-size 448

The quantized model is written to <original>_int8.onnx in the same directory.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: Path, size: int) -> np.ndarray:
    """Load and preprocess a single image → (1, 3, size, size) float32."""
    import cv2  # noqa: PLC0415
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x   = rgb.astype(np.float32) / 255.0
    x   = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)


class _CalibrationReader:
    """onnxruntime.quantization calibration data reader."""

    def __init__(self, image_paths: list[Path], input_name: str, input_size: int, limit: int = 200):
        import random  # noqa: PLC0415
        paths = list(image_paths)
        if len(paths) > limit:
            random.shuffle(paths)
            paths = paths[:limit]
        self._paths     = paths
        self._input_name = input_name
        self._size      = input_size
        self._iter      = iter(self._paths)
        print(f"   calibration: {len(self._paths)} images at {input_size}×{input_size}")

    def get_next(self):
        try:
            path = next(self._iter)
        except StopIteration:
            return None
        arr = _load_image(path, self._size)
        if arr is None:
            return self.get_next()   # skip unreadable files
        return {self._input_name: arr}

    def rewind(self):
        self._iter = iter(self._paths)


def quantize(model_path: Path, calib_images: list[Path], input_size: int) -> Path:
    try:
        from onnxruntime.quantization import (  # noqa: PLC0415
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        raise SystemExit("onnxruntime not found. Install with: pip install onnxruntime") from e

    import onnxruntime as ort  # noqa: PLC0415

    out_path = model_path.with_name(model_path.stem + "_int8.onnx")

    # Discover input name
    sess      = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    reader = _CalibrationReader(calib_images, input_name, input_size)

    print(f"   quantizing  : {model_path.name}")
    print(f"   output      : {out_path.name}")

    quantize_static(
        model_input=str(model_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=False,   # per-channel is more accurate but slower on ARM
        reduce_range=False,
    )

    orig_mb  = model_path.stat().st_size / 1024 / 1024
    quant_mb = out_path.stat().st_size   / 1024 / 1024
    print(f"   size        : {orig_mb:.1f} MB → {quant_mb:.1f} MB  ({quant_mb/orig_mb*100:.0f}%)")
    return out_path


def _collect_images(root: Path, limit: int = 1000) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
        if len(paths) >= limit:
            break
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--model",        type=Path, required=True, help="FP32 .onnx model path")
    parser.add_argument("--calib-images", type=Path, required=True, help="Dir of calibration images")
    parser.add_argument("--input-size",   type=int,  default=384,   help="Model input resolution (default 384)")
    parser.add_argument("--calib-limit",  type=int,  default=200,   help="Max calibration images (default 200)")
    args = parser.parse_args()

    if not args.model.exists():
        # Allow glob expansion
        matches = sorted(glob.glob(str(args.model)))
        if not matches:
            raise SystemExit(f"Model not found: {args.model}")
        args.model = Path(matches[-1])

    images = _collect_images(args.calib_images, limit=max(args.calib_limit * 5, 1000))
    if not images:
        raise SystemExit(f"No images found under {args.calib_images}")

    print(f"\n── INT8 Static Quantization ──────────────────────────────")
    out = quantize(args.model, images, args.input_size)
    print(f"\n── Done ──────────────────────────────────────────────────")
    print(f"   {out}")


if __name__ == "__main__":
    main()
