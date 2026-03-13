#!/usr/bin/env python3
"""SIFT homography card detector.

Wraps the SIFT feature-matching pipeline from the packopening pipeline
(02_data_sets/packopening/code/pipeline/02_precompute_sift.py and
04_match_frames.py) to compute card corners via homography.

IMPORTANT: This detector requires a gallery to function. When gallery=None,
it immediately returns DetectionResult.no_card(). It is primarily used as a
ground-truth labeling tool, not as a deployment detector — because it requires
knowing the card identity in advance.

Requires: opencv-contrib-python or opencv-contrib-python-headless (for SIFT).

Gallery format
--------------
A dict mapping card_id (str) → {"kps": kp_array, "descs": descriptors}
where kp_array is shape (N, 7) float32 and descriptors is shape (N, 128) float32,
as produced by load_sift_features() from 02_precompute_sift.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from base import CardDetector, DetectionResult

# Standard Scryfall reference card dimensions (pixels)
REF_W: int = 745
REF_H: int = 1040

# Matching thresholds (mirror the packopening pipeline)
MIN_MATCHES: int = 20
LOWE_RATIO: float = 0.75


def _kp_array_to_kps(kp_array: np.ndarray):  # type: ignore[return]
    """Convert (N, 7) float32 array back to list[cv2.KeyPoint]."""
    import cv2  # lazy import
    return [
        cv2.KeyPoint(
            x=float(r[0]), y=float(r[1]), size=float(r[2]),
            angle=float(r[3]), response=float(r[4]),
            octave=int(r[5]), class_id=int(r[6]),
        )
        for r in kp_array
    ]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL order."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class SIFTHomographyDetector(CardDetector):
    """Card corner detector using SIFT feature matching + homography.

    When gallery is None (the common case), returns no_card() immediately.
    When a gallery is provided, matches the query image against all gallery
    entries and computes a homography for the best-matching card.

    Args:
        min_matches: Minimum number of good SIFT matches required (default 20).
        lowe_ratio:  Lowe's ratio test threshold (default 0.75).
    """

    def __init__(self, min_matches: int = MIN_MATCHES, lowe_ratio: float = LOWE_RATIO) -> None:
        self.min_matches = min_matches
        self.lowe_ratio = lowe_ratio

    def detect(self, image: np.ndarray, gallery: dict[str, Any] | None = None) -> DetectionResult:
        """Detect card using SIFT homography.

        Args:
            image:   HxWx3 uint8 BGR image (OpenCV convention).
            gallery: dict mapping card_id → {"kps": ndarray (N,7), "descs": ndarray (N,128)}.
                     If None, returns no_card() immediately.

        Returns:
            DetectionResult with normalized corners, or no_card() if matching fails.
        """
        if gallery is None:
            return DetectionResult(
                card_present=False,
                corners=None,
                confidence=0.0,
                metadata={"reason": "no gallery provided"},
            )

        # Lazy import cv2 — avoids hard dependency at import time
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-contrib-python is required for SIFTHomographyDetector. "
                "Install with: pip install opencv-contrib-python-headless"
            ) from exc

        h, w = image.shape[:2]

        # Compute SIFT features for query image
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        query_kps, query_descs = sift.detectAndCompute(gray, None)

        if query_descs is None or len(query_descs) < 2:
            return DetectionResult.no_card()

        query_descs = query_descs.astype(np.float32)

        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})

        best_card_id = None
        best_matches: list = []
        best_kps_ref: list = []

        for card_id, entry in gallery.items():
            kp_array = entry.get("kps")
            descs_ref = entry.get("descs")
            if descs_ref is None or len(descs_ref) < 2:
                continue

            descs_ref = descs_ref.astype(np.float32)

            try:
                raw_matches = flann.knnMatch(query_descs, descs_ref, k=2)
            except cv2.error:
                continue

            good = [
                m for pair in raw_matches if len(pair) == 2
                for m, n in [pair] if m.distance < self.lowe_ratio * n.distance
            ]

            if len(good) > len(best_matches):
                best_matches = good
                best_card_id = card_id
                best_kps_ref = _kp_array_to_kps(kp_array) if kp_array is not None else []

        if len(best_matches) < self.min_matches or best_card_id is None:
            return DetectionResult(
                card_present=False,
                corners=None,
                confidence=0.0,
                metadata={"reason": "insufficient matches", "best_matches": len(best_matches)},
            )

        # Full homography on best candidate
        src_pts = np.float32([query_kps[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([best_kps_ref[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return DetectionResult(
                card_present=False,
                corners=None,
                confidence=0.0,
                metadata={"reason": "homography failed", "card_id": best_card_id},
            )

        # Map reference card corners through the homography to get query-image corners
        ref_corners = np.float32([
            [0,     0    ],
            [REF_W, 0    ],
            [REF_W, REF_H],
            [0,     REF_H],
        ]).reshape(-1, 1, 2)

        query_corners = cv2.perspectiveTransform(ref_corners, H).reshape(4, 2)
        ordered = _order_corners(query_corners)

        # Check all corners are within the image (with a small margin)
        margin = 0.05
        if (
            ordered[:, 0].min() < -margin * w
            or ordered[:, 0].max() > (1 + margin) * w
            or ordered[:, 1].min() < -margin * h
            or ordered[:, 1].max() > (1 + margin) * h
        ):
            return DetectionResult(
                card_present=False,
                corners=None,
                confidence=0.0,
                metadata={"reason": "corners outside image", "card_id": best_card_id},
            )

        # Normalize
        scale = np.array([w, h], dtype=np.float32)
        normalized = np.clip(ordered / scale, 0.0, 1.0)

        n_inliers = int(mask.sum()) if mask is not None else len(best_matches)
        confidence = min(1.0, n_inliers / 100.0)

        return DetectionResult(
            card_present=True,
            corners=normalized,
            confidence=confidence,
            metadata={
                "card_id": best_card_id,
                "num_matches": len(best_matches),
                "num_inliers": n_inliers,
            },
        )

    def __repr__(self) -> str:
        return f"SIFTHomographyDetector(min_matches={self.min_matches}, lowe_ratio={self.lowe_ratio})"
