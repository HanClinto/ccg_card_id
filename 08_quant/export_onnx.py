#!/usr/bin/env python3
"""Export PyTorch checkpoints to ONNX for edge / Pi deployment.

Run on the dev machine (requires PyTorch + timm).
The resulting .onnx files have no PyTorch dependency and run on Pi
via onnxruntime-cpu.

Usage
-----
    python 08_quant/export_onnx.py \\
        --detector-ckpt  $DATA/results/corner_detector/mvit_simcc_lc5_img384_ph10_seedin_blr10_fz2/epoch_0045.pt \\
        --identifier-ckpt $DATA/results/mobilevit_xxs/mobilevit_xxs_multitask_illustration_id+set_code_shared_128d+128d_mvitxxs_shared2h_arcface_v2light_img448_ph10/epoch_0015.pt \\
        --output-dir $DATA/models_onnx/

Outputs
-------
    detector_<run>_e<N>.onnx
        input  : (1, 3, 384, 384)  ImageNet-normalised float32
        outputs: corners   (1, 8)  normalised [0,1] TL/TR/BR/BL x0,y0…
                 presence  (1,)    raw presence logit
                 sharpness (1,)    mean peak of SimCC softmax distributions

    identifier_<run>_e<N>.onnx
        input  : (1, 3, 448, 448)  ImageNet-normalised float32
        output : embedding (1, 128) L2-normalised
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path bootstrap — make project packages importable
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([
    str(ROOT),
    str(ROOT / "03_detector"),
    str(ROOT / "03_detector" / "detectors"),
    str(ROOT / "03_detector" / "detectors" / "tiny_corner_cnn"),
    str(ROOT / "04_vectorize" / "mobilevit_xxs"),
])


# ---------------------------------------------------------------------------
# SimCC export wrapper
# ---------------------------------------------------------------------------

class _SimCCONNXWrapper(nn.Module):
    """Convert SimCCCornerDetector list outputs to plain tensors for ONNX.

    The raw model returns coord_logits as a Python list of 8 tensors, which
    ONNX cannot represent directly.  This wrapper stacks them and computes
    a single sharpness scalar (mean peak of the 8 softmax distributions)
    inside the graph so we get clean tensor outputs:
        corners   (1, 8)
        presence  (1,)
        sharpness (1,)
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        corners, presence, coord_logits = self.model(x)
        stacked = torch.stack(coord_logits, dim=1)          # (B, 8, num_bins)
        probs   = torch.softmax(stacked, dim=-1)
        mean_peak = probs.max(dim=-1).values.mean(dim=-1)   # (B,)
        return corners, presence, mean_peak


# ---------------------------------------------------------------------------
# Detector export
# ---------------------------------------------------------------------------

def export_detector(ckpt_path: Path, out_dir: Path) -> Path:
    from model import SimCCCornerDetector  # noqa: PLC0415

    print(f"\n── Detector ──────────────────────────────────────────────")
    print(f"   checkpoint : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "simcc")
    if arch != "simcc":
        raise ValueError(f"Expected 'simcc' arch, got '{arch}'. Only SimCC is supported.")

    base_model = SimCCCornerDetector(pretrained_backbone=False)
    base_model.load_state_dict(ckpt["model"])
    base_model.eval()

    model = _SimCCONNXWrapper(base_model)
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    run_name = ckpt_path.parent.name
    out_name = f"detector_{run_name}_e{epoch}.onnx"
    out_path = out_dir / out_name

    dummy  = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        corners, presence, sharpness = model(dummy)
    print(f"   dummy run  : corners={corners.shape}, presence={presence.shape}, sharpness={sharpness.shape}")

    print(f"   exporting  → {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=18,
        input_names=["image"],
        output_names=["corners", "presence", "sharpness"],
        dynamic_axes=None,   # batch=1 fixed for edge inference
    )

    _verify_onnx(out_path, dummy.numpy(), ["corners", "presence", "sharpness"])
    return out_path


# ---------------------------------------------------------------------------
# Identifier export
# ---------------------------------------------------------------------------

def export_identifier(ckpt_path: Path, out_dir: Path) -> Path:
    from retrieval import FineTunedEmbeddingModel  # noqa: PLC0415

    print(f"\n── Identifier ────────────────────────────────────────────")
    print(f"   checkpoint : {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    emb_dims  = cargs.get("embedding_dims") or [cargs.get("embedding_dim", 128)]
    emb_dim   = int(emb_dims[0])
    backbone  = cargs.get("backbone", "mobilevit_xxs")
    img_size  = int(cargs.get("image_size", 448))
    print(f"   backbone   : {backbone}  emb_dim={emb_dim}  img_size={img_size}")

    model = FineTunedEmbeddingModel(backbone, emb_dim)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"   WARNING: {len(missing)} missing keys (unexpected for shared-head model)")
    if unexpected:
        # Separate-head models store extra proj keys — fine to ignore
        print(f"   info: {len(unexpected)} extra keys in checkpoint (ignored: {unexpected[:3]}…)")
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    run_name = ckpt_path.parent.name
    out_name = f"identifier_{run_name}_e{epoch}.onnx"
    out_path = out_dir / out_name

    dummy = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        emb = model(dummy)
    print(f"   dummy run  : embedding={emb.shape}")

    print(f"   exporting  → {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes=None,
    )

    _verify_onnx(out_path, dummy.numpy(), ["embedding"])
    return out_path


# ---------------------------------------------------------------------------
# ONNX verification
# ---------------------------------------------------------------------------

def _verify_onnx(path: Path, dummy_input: np.ndarray, output_names: list[str]) -> None:
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError:
        print("   (onnxruntime not installed — skipping verification)")
        return

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: dummy_input})
    print(f"   verified   : {', '.join(f'{n}={o.shape}' for n, o in zip(output_names, outs))}")
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"   size       : {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX for edge deployment")
    parser.add_argument("--detector-ckpt",   type=Path, help="SimCC detector checkpoint (.pt)")
    parser.add_argument("--identifier-ckpt", type=Path, help="ArcFace identifier checkpoint (.pt)")
    parser.add_argument("--output-dir",      type=Path, required=True, help="Directory for .onnx files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    if args.detector_ckpt:
        exported.append(export_detector(args.detector_ckpt, args.output_dir))
    if args.identifier_ckpt:
        exported.append(export_identifier(args.identifier_ckpt, args.output_dir))

    if not exported:
        parser.error("Provide at least one of --detector-ckpt / --identifier-ckpt")

    print(f"\n── Done ──────────────────────────────────────────────────")
    for p in exported:
        print(f"   {p}")


if __name__ == "__main__":
    main()
