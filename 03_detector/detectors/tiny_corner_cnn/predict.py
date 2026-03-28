#!/usr/bin/env python3
"""Inference wrapper for the neural corner detector.

Implements the CardDetector ABC so NeuralCornerDetectorInference can be used
interchangeably with CannyPolyDetector or SIFTHomographyDetector.

Example:
    detector = NeuralCornerDetectorInference("path/to/last.pt")
    img_bgr = cv2.imread("card.jpg")
    result = detector.detect(img_bgr)
    if result.card_present:
        corners_px = result.corners_pixel(img_bgr.shape[1], img_bgr.shape[0])
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

_HERE        = Path(__file__).resolve().parent
_DETECTOR_DIR = Path(__file__).resolve().parents[2]  # 03_detector/
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_DETECTOR_DIR))

from base import CardDetector, DetectionResult, sort_corners_canonical
from model import TinyCornerCNN, MobileViTCornerDetector, SimCCCornerDetector

_ARCH_MAP = {
    "tiny":      TinyCornerCNN,
    "mobilevit": MobileViTCornerDetector,
    "simcc":     SimCCCornerDetector,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_DEFAULT_INPUT_SIZE = 448


def _make_preprocess(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class NeuralCornerDetectorInference(CardDetector):
    """Inference wrapper around NeuralCornerDetector.

    Loads a trained checkpoint and implements detect() for drop-in use
    alongside the classical detectors.

    Args:
        checkpoint_path:    Path to a checkpoint saved by tiny_corner_cnn/train.py.
        device:             "cpu", "cuda", "mps", or a torch.device.
        presence_threshold: Sigmoid threshold for card_present=True (default 0.5).
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        arch: str = "tiny",
        device: str | torch.device = "cpu",
        presence_threshold: float = 0.5,
        ignore_presence: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.presence_threshold = presence_threshold
        self.ignore_presence = ignore_presence

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        # Arch can be stored in checkpoint or passed explicitly
        arch = ckpt.get("arch", arch)
        model_cls = _ARCH_MAP.get(arch)
        if model_cls is None:
            raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(_ARCH_MAP)}")
        self.model = model_cls()
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()

        self._arch  = arch
        self._epoch = ckpt.get("epoch", "?")
        # SimCC: input size == num_bins (384); others default to 448
        self._input_size = getattr(self.model, "num_bins", _DEFAULT_INPUT_SIZE)
        self._preprocess = _make_preprocess(self._input_size)

    def detect(
        self,
        image: np.ndarray,
        gallery=None,
        return_heatmaps: bool = False,
    ) -> DetectionResult:
        """Detect card corners in a BGR image.

        Args:
            image:           HxWx3 uint8 BGR image (OpenCV convention).
            gallery:         Unused. Accepted for interface compatibility.
            return_heatmaps: If True and the model is SimCC, include per-corner
                             1D coordinate distributions in metadata under
                             "simcc_heatmaps" (dict with keys "heatmap_x" and
                             "heatmap_y", each a list of 4 float32 arrays of
                             length num_bins ordered TL, TR, BR, BL).

        Returns:
            DetectionResult with normalized corners in canonical order
            (CW, shortest edge at 0→1 — see base.py convention),
            or no_card() if the presence score is below threshold.
        """
        # Convert BGR → RGB PIL image
        img_rgb = Image.fromarray(image[:, :, ::-1])

        tensor = self._preprocess(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        # TinyCornerCNN: (corners, presence, heatmaps)
        # MobileViT:     (corners, presence)
        # SimCC:         (corners, presence, coord_logits)  — coord_logits is a list, not a tensor
        pred_corners          = out[0]
        pred_presence_logit   = out[1]
        raw_out2              = out[2] if len(out) == 3 else None
        # Only use out[2] as heatmaps if it's a single tensor (TinyCornerCNN); SimCC returns a list
        heatmaps              = raw_out2 if isinstance(raw_out2, torch.Tensor) else None

        presence_prob = float(torch.sigmoid(pred_presence_logit).squeeze())

        corners_flat = pred_corners.squeeze().cpu().numpy()  # (8,)
        corners = np.clip(corners_flat, 0.0, 1.0).reshape(4, 2).astype(np.float32)
        h, w = image.shape[:2]
        corners = sort_corners_canonical(corners, img_w=w, img_h=h)

        # Per-corner confidence from heatmap peak values (TinyCornerCNN only)
        corner_confidences = None
        if heatmaps is not None:
            peak_logits = heatmaps.squeeze(0).amax(dim=(-2, -1))  # (4,)
            corner_confidences = torch.sigmoid(peak_logits).cpu().numpy().tolist()

        # SimCC: extract 1D coordinate distributions for sharpness scoring + optional visualization
        simcc_heatmaps = None
        simcc_sharpness = None
        if raw_out2 is not None and not isinstance(raw_out2, torch.Tensor):
            coord_logits = raw_out2  # list of 8 tensors [x0,y0,x1,y1,x2,y2,x3,y3] → TL,TR,BR,BL
            hm_x, hm_y = [], []
            peaks = []
            for i in range(4):
                xp = torch.softmax(coord_logits[i * 2].squeeze(0), dim=0).cpu().numpy().astype(np.float32)
                yp = torch.softmax(coord_logits[i * 2 + 1].squeeze(0), dim=0).cpu().numpy().astype(np.float32)
                hm_x.append(xp)
                hm_y.append(yp)
                peaks.append(float(xp.max()))
                peaks.append(float(yp.max()))
            simcc_sharpness = {
                "mean_peak": float(np.mean(peaks)),
                "min_peak":  float(np.min(peaks)),
            }
            if return_heatmaps:
                simcc_heatmaps = {"heatmap_x": hm_x, "heatmap_y": hm_y}

        if not self.ignore_presence and presence_prob < self.presence_threshold:
            return DetectionResult(
                card_present=False,
                corners=None,
                confidence=presence_prob,
                metadata={"presence_prob": presence_prob, "corner_confidences": None},
            )

        return DetectionResult(
            card_present=True,
            corners=corners,
            confidence=presence_prob,
            metadata={
                "presence_prob":   presence_prob,
                "epoch":           self._epoch,
                "corner_confidences": corner_confidences,
                "simcc_sharpness": simcc_sharpness,
                "simcc_heatmaps":  simcc_heatmaps,
            },
        )

    def __repr__(self) -> str:
        return (
            f"NeuralCornerDetectorInference("
            f"ckpt={self.checkpoint_path.name}, "
            f"epoch={self._epoch}, "
            f"threshold={self.presence_threshold})"
        )
