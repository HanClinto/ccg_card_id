#!/usr/bin/env python3
"""Classical Canny + polygon card detector.

Algorithm:
  1. BGR → grayscale
  2. Optional CLAHE contrast enhancement
  3. Gaussian blur (configurable kernel)
  4. Canny edge detection (configurable thresholds)
  5. Dilate edges to close small gaps
  6. findContours → sort by area descending
  7. For each contour: convex hull → approxPolyDP (epsilon = 2% of perimeter)
     Accept if 4 points and area in [min_area_frac, max_area_frac] of image area
  8. Sort corners into canonical order (CW, shortest edge at 0→1 — see base.py)
  9. Normalize to [0, 1] and return DetectionResult

Works well on high-contrast bordered cards against plain backgrounds.
Breaks on borderless art, foil glare, sleeves, and cluttered backgrounds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from base import CardDetector, DetectionResult, sort_corners_canonical


class CannyPolyDetector(CardDetector):
    """Card detector using Canny edges + contour quadrilateral fitting.

    Args:
        blur_kernel:    Size of Gaussian blur kernel (odd integer, default 5).
        canny_low:      Lower Canny hysteresis threshold (default 50).
        canny_high:     Upper Canny hysteresis threshold (default 150).
        min_area_frac:  Minimum quad area as fraction of image area (default 0.03).
        max_area_frac:  Maximum quad area as fraction of image area (default 0.95).
        use_clahe:      Apply CLAHE before blurring to boost local contrast (default True).
    """

    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area_frac: float = 0.03,
        max_area_frac: float = 0.95,
        use_clahe: bool = True,
    ) -> None:
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.use_clahe = use_clahe

        if use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, image: np.ndarray, gallery=None) -> DetectionResult:
        """Detect card corners using Canny edge detection.

        Args:
            image:   HxWx3 uint8 BGR image.
            gallery: Unused. Accepted for interface compatibility.

        Returns:
            DetectionResult with normalized corners in canonical order
            (CW, shortest edge at 0→1 — see base.py convention),
            or no_card() if no quad found.
        """
        h, w = image.shape[:2]
        image_area = float(h * w)

        # Step 1: grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: optional CLAHE
        if self.use_clahe:
            gray = self._clahe.apply(gray)

        # Step 3: Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # Step 4: Canny
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Step 5: dilate to close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Step 6: find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return DetectionResult.no_card()

        # Sort by area descending so we consider the largest shapes first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        min_area = self.min_area_frac * image_area
        max_area = self.max_area_frac * image_area

        # Step 7: find first acceptable quad
        for contour in contours:
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, closed=True)
            if perimeter < 1e-6:
                continue
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(hull, epsilon, closed=True)

            if len(approx) != 4:
                continue

            area = cv2.contourArea(approx)
            if area < min_area or area > max_area:
                continue

            # Found our quad — normalize then apply canonical sort
            pts = approx.reshape(4, 2).astype(np.float32)
            scale = np.array([w, h], dtype=np.float32)
            normalized = np.clip(pts / scale, 0.0, 1.0)
            normalized = sort_corners_canonical(normalized, img_w=w, img_h=h)

            confidence = min(1.0, area / image_area)

            return DetectionResult(
                card_present=True,
                corners=normalized,
                confidence=confidence,
                metadata={"quad_area_pct": area / image_area},
            )

        return DetectionResult.no_card()

    def __repr__(self) -> str:
        return (
            f"CannyPolyDetector("
            f"blur={self.blur_kernel}, canny={self.canny_low}/{self.canny_high}, "
            f"area={self.min_area_frac:.2f}-{self.max_area_frac:.2f}, "
            f"clahe={self.use_clahe})"
        )
