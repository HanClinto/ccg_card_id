#!/usr/bin/env python3
"""Base class, data types, and shared utilities for card detection.

==========================================================================
Corner convention — ALL detectors MUST follow this contract
==========================================================================

Every DetectionResult.corners array is a float32 (4, 2) in normalized image
coordinates [0, 1] (x, y), ordered as follows:

    Winding:  CLOCKWISE
    Index 0:  start of the SHORTEST edge of the quad
    Index 1:  end   of the SHORTEST edge  (0 → 1 is the short "top" edge)
    Index 2:  far corner
    Index 3:  the remaining corner  (3 → 0 closes the short "bottom" edge)

Rationale
---------
MTG cards are portrait (63 × 88 mm). The two shortest edges of the detected
quad are the card's top and bottom; the two longest are the sides.  Placing
the shortest edge first in CW order gives a stable, orientation-robust
convention that does not depend on which direction is "up" in the image.

                    0 ─────── 1
                   /           \
                  3             2
                   \           /
                    (implied)

The 180° ambiguity (whether edge 0→1 is the physical card top or bottom) is
intentionally left unresolved here.  Code that needs a definitive orientation
(pHash matching, ArcFace embedding) should try BOTH the corners as-given AND
the same corners rolled by 2 (a 180° rotation), then keep whichever gives the
better match.

Use sort_corners_canonical() to produce corners in this convention.

==========================================================================
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Canonical corner sort
# ---------------------------------------------------------------------------

def sort_corners_canonical(corners: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Sort 4 detected corners into the canonical CW / shortest-edge-first order.

    Steps
    -----
    1. Compute the centroid; sort corners by atan2 angle ascending.
       In image coordinates (y increases downward), ascending atan2 gives
       clockwise order — TL→TR→BR→BL for an upright portrait card.
    2. Compute the four edge lengths in *pixel* space (not normalized, to
       avoid distortion from non-square images).
    3. Roll the sequence so the shortest edge is at position 0→1.

    The resulting array satisfies the corner convention documented at the
    top of this module.  The 0°/180° orientation ambiguity is NOT resolved;
    callers that need it should try both the returned corners and
    ``np.roll(corners, -2, axis=0)`` and pick the better match.

    Args:
        corners: shape (4, 2), normalized [0, 1], any order.
        img_w:   image width  in pixels — used for pixel-accurate edge lengths.
        img_h:   image height in pixels.

    Returns:
        shape (4, 2) float32, canonical CW order, shortest edge at 0→1.
    """
    corners = np.asarray(corners, dtype=np.float32)
    # Step 1 — clockwise sort via atan2
    cx, cy = corners.mean(axis=0)
    angles  = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
    corners = corners[np.argsort(angles)]   # ascending atan2 = CW in image space

    # Step 2/3 — find shortest edge in pixel space and roll it to front
    scale    = np.array([img_w, img_h], dtype=np.float32)
    pts      = corners * scale
    edge_len = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    shortest = int(np.argmin(edge_len))
    return np.roll(corners, -shortest, axis=0)


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of a single card detection attempt."""

    card_present: bool
    """True if the detector believes a card is visible in the image."""

    corners: np.ndarray | None
    """Shape (4, 2) float32, normalized [0, 1] (x, y).
    Ordered per the corner convention above: CW, shortest edge at 0→1.
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


# ---------------------------------------------------------------------------
# Abstract detector
# ---------------------------------------------------------------------------

class CardDetector(ABC):
    """Abstract base class for all card detectors.

    Subclasses MUST return corners in the canonical convention defined at the
    top of this module (CW winding, shortest edge at index 0→1).
    Use sort_corners_canonical() to produce conforming output.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, gallery=None) -> DetectionResult:
        """Detect a card in the given image.

        Args:
            image:   HxWx3 uint8 BGR image (OpenCV convention).
            gallery: Optional detector-specific gallery object.
                     Only used by SIFTHomographyDetector.
                     Pass None for detectors that don't require one.

        Returns:
            DetectionResult with corners in canonical normalized [0, 1]
            coordinates (see module docstring for ordering contract).
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
