#!/usr/bin/env python3
"""Base class and data types for card detection.

A CardDetector takes a raw image and returns the four corner points of the
most prominent card, plus a confidence score and presence flag.

Corner ordering (clockwise from top-left):
    corner 0 — Top-Left
    corner 1 — Top-Right
    corner 2 — Bottom-Right
    corner 3 — Bottom-Left

Coordinates are normalized to [0, 1] relative to image (width, height).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectionResult:
    """Result of a single card detection attempt."""

    card_present: bool
    """True if the detector believes a card is visible in the image."""

    corners: np.ndarray | None
    """Shape (4, 2) float32, normalized [0, 1] (x, y).  TL, TR, BR, BL order.
    None when card_present is False."""

    confidence: float = 0.0
    """Detector-specific confidence score in [0, 1]."""

    metadata: dict = field(default_factory=dict)
    """Optional detector-specific diagnostics (e.g. num_matches, area_pct)."""

    def corners_pixel(self, image_width: int, image_height: int) -> np.ndarray | None:
        """Return corners in pixel coordinates for a given image size."""
        if self.corners is None:
            return None
        scale = np.array([image_width, image_height], dtype=np.float32)
        return (self.corners * scale).astype(np.float32)

    @staticmethod
    def no_card() -> "DetectionResult":
        return DetectionResult(card_present=False, corners=None, confidence=0.0)


class CardDetector(ABC):
    """Abstract base class for all card detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray, gallery=None) -> DetectionResult:
        """Detect a card in the given image.

        Args:
            image:   HxWx3 uint8 BGR image (OpenCV convention).
            gallery: Optional detector-specific gallery object.
                     Only used by SIFTHomographyDetector.
                     Pass None for detectors that don't require one.

        Returns:
            DetectionResult with corners in normalized [0, 1] coordinates.
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
