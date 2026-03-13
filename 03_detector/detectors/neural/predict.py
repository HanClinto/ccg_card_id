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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from base import CardDetector, DetectionResult              # noqa: E402
from model import TinyCornerCNN, MobileViTCornerDetector    # noqa: E402

_ARCH_MAP = {"tiny": TinyCornerCNN, "mobilevit": MobileViTCornerDetector}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE     = 224

_PREPROCESS = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class NeuralCornerDetectorInference(CardDetector):
    """Inference wrapper around NeuralCornerDetector.

    Loads a trained checkpoint and implements detect() for drop-in use
    alongside the classical detectors.

    Args:
        checkpoint_path:    Path to a checkpoint saved by neural/train.py.
        device:             "cpu", "cuda", "mps", or a torch.device.
        presence_threshold: Sigmoid threshold for card_present=True (default 0.5).
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        arch: str = "tiny",
        device: str | torch.device = "cpu",
        presence_threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.presence_threshold = presence_threshold

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

    def detect(self, image: np.ndarray, gallery=None) -> DetectionResult:
        """Detect card corners in a BGR image.

        Args:
            image:   HxWx3 uint8 BGR image (OpenCV convention).
            gallery: Unused. Accepted for interface compatibility.

        Returns:
            DetectionResult with normalized corners in TL/TR/BR/BL order,
            or no_card() if the presence score is below threshold.
        """
        # Convert BGR → RGB PIL image
        img_rgb = Image.fromarray(image[:, :, ::-1])

        tensor = _PREPROCESS(img_rgb).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            pred_corners, pred_presence_logit = self.model(tensor)

        presence_prob = float(torch.sigmoid(pred_presence_logit).squeeze())
        if presence_prob < self.presence_threshold:
            return DetectionResult(
                card_present=False,
                corners=None,
                confidence=presence_prob,
                metadata={"presence_prob": presence_prob},
            )

        corners_flat = pred_corners.squeeze().cpu().numpy()  # (8,)
        corners = np.clip(corners_flat, 0.0, 1.0).reshape(4, 2).astype(np.float32)

        return DetectionResult(
            card_present=True,
            corners=corners,
            confidence=presence_prob,
            metadata={"presence_prob": presence_prob, "epoch": self._epoch},
        )

    def __repr__(self) -> str:
        return (
            f"NeuralCornerDetectorInference("
            f"ckpt={self.checkpoint_path.name}, "
            f"epoch={self._epoch}, "
            f"threshold={self.presence_threshold})"
        )
