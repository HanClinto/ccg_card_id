"""Unit tests for coordinate conventions, scaling, and transformation logic.

These tests are deliberately paranoid about x/y ordering and the direction of
every scale factor.  The goal is to catch the class of bug where everything
*looks* plausible but coordinates are subtly transposed or scaled by the wrong
image dimension.

Run with:
    cd 03_detector && python -m pytest tests/test_geometry.py -v
or standalone:
    cd 03_detector && python tests/test_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure 03_detector is on sys.path regardless of where we're invoked from
_DETECTOR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DETECTOR_DIR))

from base import DetectionResult, sort_corners_canonical


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _portrait_quad(scale: float = 1.0) -> np.ndarray:
    """Axis-aligned portrait card in a square image.

    Returns TL, TR, BR, BL in that order (not yet canonical).
    Corners at 20%-80% horizontally, 10%-90% vertically.
    Short edges (top & bottom): width = 0.60
    Long edges (sides): height = 0.80
    """
    return np.array([
        [0.20, 0.10],   # TL
        [0.80, 0.10],   # TR
        [0.80, 0.90],   # BR
        [0.20, 0.90],   # BL
    ], dtype=np.float32) * scale


def _landscape_quad() -> np.ndarray:
    """Portrait card in a 1920×1080 landscape frame.

    The card occupies 30%-70% horizontally, 5%-95% vertically.
    In PIXEL space (1920×1080):
      top edge  : 0.40 * 1920 = 768 px  (card's short side)
      side edge : 0.90 * 1080 = 972 px  (card's long side)
    In NORMALIZED space:
      top edge  : 0.40 (looks short)
      side edge : 0.90 (looks long)
    Both measures agree here — tests below verify the pixel-space path.
    """
    return np.array([
        [0.30, 0.05],   # TL
        [0.70, 0.05],   # TR
        [0.70, 0.95],   # BR
        [0.30, 0.95],   # BL
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. sort_corners_canonical — winding direction
# ---------------------------------------------------------------------------

class TestSortCornersCanonicalWinding:
    """Verify that output is always CW in image-space."""

    def _winding(self, pts: np.ndarray) -> str:
        """Return 'CW' or 'CCW' based on signed polygon area in image coords.

        In standard math (y-up) the shoelace formula gives a positive area for
        CCW polygons.  In image coordinates (y increases *downward*) the y-axis
        is flipped, so a CW polygon has a *positive* shoelace area.
        """
        n = len(pts)
        area2 = sum(
            (pts[i, 0] * pts[(i + 1) % n, 1] - pts[(i + 1) % n, 0] * pts[i, 1])
            for i in range(n)
        )
        # In image coords (y-down), positive shoelace area = CW
        return "CW" if area2 > 0 else "CCW"

    def test_axis_aligned_portrait_square_image(self):
        """TL,TR,BR,BL → already CW; should come back in same order."""
        quad = _portrait_quad()
        out = sort_corners_canonical(quad, img_w=1000, img_h=1000)
        assert self._winding(out) == "CW"

    def test_input_ccw_is_flipped_to_cw(self):
        """Corners given in CCW (BL,BR,TR,TL) should be sorted to CW."""
        quad = _portrait_quad()[[3, 2, 1, 0]]   # reverse = CCW
        out = sort_corners_canonical(quad, img_w=1000, img_h=1000)
        assert self._winding(out) == "CW"

    def test_random_order_always_cw(self):
        rng = np.random.default_rng(0)
        quad = _portrait_quad()
        for _ in range(20):
            shuffled = quad[rng.permutation(4)]
            out = sort_corners_canonical(shuffled, img_w=800, img_h=600)
            assert self._winding(out) == "CW", "Shuffled input should produce CW output"

    def test_landscape_image_cw(self):
        quad = _landscape_quad()
        out = sort_corners_canonical(quad, img_w=1920, img_h=1080)
        assert self._winding(out) == "CW"


# ---------------------------------------------------------------------------
# 2. sort_corners_canonical — shortest-edge-first
# ---------------------------------------------------------------------------

class TestSortCornersCanonicalShortestEdge:
    """Verify edge 0→1 is the shortest edge of the quad in pixel space."""

    def _edge_lengths_px(self, corners: np.ndarray, img_w: int, img_h: int) -> list[float]:
        pts = corners * np.array([img_w, img_h], dtype=np.float32)
        return [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]

    def test_portrait_card_square_image_top_edge_is_shortest(self):
        quad = _portrait_quad()
        out = sort_corners_canonical(quad, img_w=1000, img_h=1000)
        lengths = self._edge_lengths_px(out, 1000, 1000)
        assert lengths[0] == min(lengths), (
            f"Edge 0 ({lengths[0]:.1f}) is not the shortest. All: {lengths}"
        )

    def test_portrait_card_landscape_frame_pixel_scale_matters(self):
        """In a 1920×1080 landscape frame, the card's short edge is still
        the top/bottom (horizontal) edge in pixel space — 768 px vs 972 px.

        If we mistakenly used normalized distances instead of pixel distances,
        we would compare 0.40 vs 0.90 and still get the right answer here —
        but in the critical test below the two methods diverge.
        """
        quad = _landscape_quad()
        out = sort_corners_canonical(quad, img_w=1920, img_h=1080)
        lengths = self._edge_lengths_px(out, 1920, 1080)
        assert lengths[0] == min(lengths), (
            f"Edge 0 ({lengths[0]:.1f}) is not the shortest. All: {lengths}"
        )

    def test_scale_swap_would_give_wrong_shortest_edge(self):
        """Demonstrate that swapping img_w/img_h produces a different result
        when the image is non-square.

        This test will FAIL if sort_corners_canonical uses scale=[img_h, img_w]
        instead of [img_w, img_h] — proving the x/y convention matters.

        Card in a 1920×1080 (wide) frame where NORMALIZED edges are equal but
        PIXEL edges are not: the horizontal extent (x) is wider in pixels
        than the normalized proportion implies when h < w.
        """
        # Craft a quad where normalized edges are equal but pixel edges differ
        # under wrong vs right scale assignment.
        # Card: 20%–80% horizontal, 10%–90% vertical in 1920×1080
        # Pixel horizontal edge: 0.60 * 1920 = 1152 px
        # Pixel vertical   edge: 0.80 * 1080 = 864 px
        # Correct shortest: vertical (864 px) → edge 1→2 (the sides)
        # Wrong scale [h,w]: horizontal = 0.60*1080=648, vertical=0.80*1920=1536
        #   → shortest is horizontal → wrong edge picked
        quad = np.array([
            [0.20, 0.10],  # TL
            [0.80, 0.10],  # TR
            [0.80, 0.90],  # BR
            [0.20, 0.90],  # BL
        ], dtype=np.float32)
        img_w, img_h = 1920, 1080

        correct = sort_corners_canonical(quad, img_w=img_w, img_h=img_h)
        swapped = sort_corners_canonical(quad, img_w=img_h, img_h=img_w)  # deliberate swap

        correct_lengths = self._edge_lengths_px(correct, img_w, img_h)
        swapped_lengths = self._edge_lengths_px(swapped, img_w, img_h)

        # The correctly-sorted result should have shortest edge first in pixel space
        assert correct_lengths[0] == min(correct_lengths), (
            f"Correct sort: edge 0 is not shortest. Lengths: {correct_lengths}"
        )
        # With a non-square image, correct and swapped should differ
        assert not np.allclose(correct, swapped), (
            "Correct and swapped results are identical — test is not discriminating"
        )

    def test_shortest_edge_after_random_shuffle(self):
        """No matter how corners arrive, edge 0→1 is always shortest."""
        rng = np.random.default_rng(7)
        quad = _landscape_quad()
        for _ in range(20):
            shuffled = quad[rng.permutation(4)]
            out = sort_corners_canonical(shuffled, img_w=1920, img_h=1080)
            lengths = self._edge_lengths_px(out, 1920, 1080)
            assert lengths[0] == min(lengths)


# ---------------------------------------------------------------------------
# 3. sort_corners_canonical — column 0 is x, column 1 is y
# ---------------------------------------------------------------------------

class TestSortCornersColumnOrder:
    """Verify that column 0 holds x (horizontal) and column 1 holds y (vertical)."""

    def test_top_left_corner_has_small_x_and_small_y(self):
        """In a portrait card, corner 0 (start of shortest edge = top-left in
        standard orientation) should have the smallest x among the top two
        and the smallest y overall.
        """
        quad = _portrait_quad()  # TL, TR, BR, BL
        out = sort_corners_canonical(quad, img_w=1000, img_h=1000)
        # After CW sort with shortest edge first, corner 0 should be TL
        # TL has x=0.20, TR has x=0.80
        assert out[0, 0] < out[1, 0], (
            f"Corner 0 x ({out[0,0]}) should be less than corner 1 x ({out[1,0]}) "
            "for a portrait card in standard orientation"
        )
        # Both top corners should have y < both bottom corners
        assert out[0, 1] < out[2, 1], "Top corners should have smaller y than bottom corners"
        assert out[1, 1] < out[3, 1], "Top corners should have smaller y than bottom corners"

    def test_horizontal_flip_changes_column_0_not_column_1(self):
        """Mirroring the card (x → 1-x) should only change column 0."""
        quad = _portrait_quad()
        flipped = quad.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]  # flip x

        out_orig = sort_corners_canonical(quad, img_w=1000, img_h=1000)
        out_flip = sort_corners_canonical(flipped, img_w=1000, img_h=1000)

        # y-values should be the same set (sorted may differ)
        assert np.allclose(np.sort(out_orig[:, 1]), np.sort(out_flip[:, 1])), (
            "Horizontal flip should not change y-values"
        )
        # x-values should all be mirrored
        orig_xs = np.sort(out_orig[:, 0])
        flip_xs = np.sort(1.0 - out_flip[:, 0])
        assert np.allclose(orig_xs, flip_xs), (
            "x-values after flip should be 1-x of the original x-values"
        )

    def test_centroid_computation_uses_correct_columns(self):
        """The centroid cx = mean of column 0, cy = mean of column 1.

        We verify this indirectly: shifting all corners by (dx, 0) should shift
        the output centroid in x but not y.
        """
        quad = _portrait_quad()
        shifted = quad.copy()
        shifted[:, 0] += 0.05   # shift x only

        out_orig = sort_corners_canonical(quad, img_w=1000, img_h=1000)
        out_shift = sort_corners_canonical(shifted, img_w=1000, img_h=1000)

        cx_orig  = out_orig[:, 0].mean()
        cx_shift = out_shift[:, 0].mean()
        cy_orig  = out_orig[:, 1].mean()
        cy_shift = out_shift[:, 1].mean()

        assert abs((cx_shift - cx_orig) - 0.05) < 1e-5, (
            f"x-shift of 0.05 should move centroid x by 0.05 (got {cx_shift - cx_orig:.5f})"
        )
        assert abs(cy_shift - cy_orig) < 1e-5, (
            f"Shifting x should not move centroid y (got delta {cy_shift - cy_orig:.5f})"
        )


# ---------------------------------------------------------------------------
# 4. DetectionResult.corners_pixel — scale order
# ---------------------------------------------------------------------------

class TestCornersPixel:
    """Verify corners_pixel multiplies column 0 by width and column 1 by height."""

    def test_pixel_scale_order(self):
        corners = np.array([
            [0.25, 0.10],  # x=25%, y=10%
            [0.75, 0.10],
            [0.75, 0.90],
            [0.25, 0.90],
        ], dtype=np.float32)
        result = DetectionResult(card_present=True, corners=corners, confidence=1.0)
        px = result.corners_pixel(image_width=1920, image_height=1080)

        assert px is not None
        # Column 0 should be x-pixels: 0.25 * 1920 = 480
        assert abs(px[0, 0] - 0.25 * 1920) < 0.5, f"x-pixel wrong: {px[0,0]} != {0.25*1920}"
        # Column 1 should be y-pixels: 0.10 * 1080 = 108
        assert abs(px[0, 1] - 0.10 * 1080) < 0.5, f"y-pixel wrong: {px[0,1]} != {0.10*1080}"

    def test_scale_non_square_image(self):
        """With a non-square image, swapping width/height would produce wrong results."""
        corners = np.array([[0.5, 0.5]], dtype=np.float32)
        result = DetectionResult(card_present=True, corners=corners, confidence=1.0)
        px = result.corners_pixel(image_width=1920, image_height=1080)
        assert abs(px[0, 0] - 960) < 0.5,  f"Center x should be 960, got {px[0,0]}"
        assert abs(px[0, 1] - 540) < 0.5,  f"Center y should be 540, got {px[0,1]}"


# ---------------------------------------------------------------------------
# 5. Two-stage coordinate mapping round-trip
# ---------------------------------------------------------------------------

class TestTwoStageCoordinateMapping:
    """Verify that crop-space corners map back to full-image-space correctly."""

    def _map_crop_to_full(
        self,
        crop_corners: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        full_w: int, full_h: int,
    ) -> np.ndarray:
        """Replicate the mapping used in detect_two_stage."""
        out = crop_corners.copy()
        crop_w = x2 - x1
        crop_h = y2 - y1
        out[:, 0] = x1 / full_w + out[:, 0] * (crop_w / full_w)
        out[:, 1] = y1 / full_h + out[:, 1] * (crop_h / full_h)
        return out

    def test_center_of_crop_maps_to_center_of_crop_region(self):
        """A corner at (0.5, 0.5) in crop space should map to the center
        of the crop region in full-image space."""
        x1, y1, x2, y2 = 480, 270, 1440, 810   # center half of 1920×1080
        full_w, full_h = 1920, 1080
        crop_w, crop_h = x2 - x1, y2 - y1

        crop_center = np.array([[0.5, 0.5]], dtype=np.float32)
        full = self._map_crop_to_full(crop_center, x1, y1, x2, y2, full_w, full_h)

        expected_x = (x1 + 0.5 * crop_w) / full_w
        expected_y = (y1 + 0.5 * crop_h) / full_h
        assert abs(full[0, 0] - expected_x) < 1e-6, f"x: {full[0,0]} != {expected_x}"
        assert abs(full[0, 1] - expected_y) < 1e-6, f"y: {full[0,1]} != {expected_y}"

    def test_top_left_of_crop_maps_to_crop_origin_in_full_image(self):
        """Corner at (0, 0) in crop space = (x1, y1) in full image."""
        x1, y1, x2, y2 = 200, 100, 900, 800
        full_w, full_h = 1920, 1080

        crop_tl = np.array([[0.0, 0.0]], dtype=np.float32)
        full = self._map_crop_to_full(crop_tl, x1, y1, x2, y2, full_w, full_h)

        assert abs(full[0, 0] - x1 / full_w) < 1e-6, f"x: {full[0,0]} != {x1/full_w}"
        assert abs(full[0, 1] - y1 / full_h) < 1e-6, f"y: {full[0,1]} != {y1/full_h}"

    def test_bottom_right_of_crop_maps_to_crop_corner_in_full_image(self):
        """Corner at (1, 1) in crop space = (x2, y2) in full image."""
        x1, y1, x2, y2 = 200, 100, 900, 800
        full_w, full_h = 1920, 1080

        crop_br = np.array([[1.0, 1.0]], dtype=np.float32)
        full = self._map_crop_to_full(crop_br, x1, y1, x2, y2, full_w, full_h)

        assert abs(full[0, 0] - x2 / full_w) < 1e-6, f"x: {full[0,0]} != {x2/full_w}"
        assert abs(full[0, 1] - y2 / full_h) < 1e-6, f"y: {full[0,1]} != {y2/full_h}"

    def test_x_and_y_axes_are_independent(self):
        """Verify column 0 is only affected by x1/x2/full_w, column 1 only by y1/y2/full_h."""
        crop_corners = np.array([[0.3, 0.7]], dtype=np.float32)
        x1, y1, x2, y2 = 100, 200, 500, 800
        full_w, full_h = 1920, 1080

        full = self._map_crop_to_full(crop_corners, x1, y1, x2, y2, full_w, full_h)

        expected_x = (x1 + 0.3 * (x2 - x1)) / full_w
        expected_y = (y1 + 0.7 * (y2 - y1)) / full_h

        assert abs(full[0, 0] - expected_x) < 1e-6
        assert abs(full[0, 1] - expected_y) < 1e-6

    def test_swapping_xy_in_mapping_gives_wrong_result(self):
        """If we accidentally used y-offsets for column 0, results would be wrong."""
        crop_corners = np.array([[0.5, 0.5]], dtype=np.float32)
        x1, y1, x2, y2 = 100, 400, 800, 900   # deliberately asymmetric crop
        full_w, full_h = 1920, 1080

        correct = self._map_crop_to_full(crop_corners, x1, y1, x2, y2, full_w, full_h)

        # Simulate bug: swap x1/y1 and x2/y2 in the mapping
        bugged = crop_corners.copy()
        bugged[:, 0] = y1 / full_h + bugged[:, 0] * ((y2 - y1) / full_h)  # using y for col0
        bugged[:, 1] = x1 / full_w + bugged[:, 1] * ((x2 - x1) / full_w)  # using x for col1

        assert not np.allclose(correct, bugged), (
            "The bug simulation should produce a different result from correct mapping"
        )


# ---------------------------------------------------------------------------
# 6. _bbox_from_corners — x/y consistency
# ---------------------------------------------------------------------------

class TestBboxFromCorners:
    """Verify _bbox_from_corners uses column 0 as x and column 1 as y."""

    def _bbox(self, corners, img_w, img_h, pad=0.0):
        from eval.two_stage_test import _bbox_from_corners
        return _bbox_from_corners(corners, img_w, img_h, pad)

    def test_known_quad_no_pad(self):
        """Card at 20%-80% x, 10%-90% y in a 1000×1000 image."""
        corners = _portrait_quad()
        x1, y1, x2, y2 = self._bbox(corners, 1000, 1000, pad=0.0)
        assert x1 == 200, f"x1={x1}, expected 200"
        assert y1 == 100, f"y1={y1}, expected 100"
        assert x2 == 800, f"x2={x2}, expected 800"
        assert y2 == 900, f"y2={y2}, expected 900"

    def test_column0_is_x_not_y(self):
        """If column 0 were treated as y, bbox would be wrong for non-square images."""
        corners = np.array([
            [0.3, 0.1],  # x=30%, y=10%
            [0.7, 0.1],
            [0.7, 0.9],
            [0.3, 0.9],
        ], dtype=np.float32)
        img_w, img_h = 1920, 1080

        x1, y1, x2, y2 = self._bbox(corners, img_w, img_h, pad=0.0)

        # x range: 0.3*1920=576 to 0.7*1920=1344
        assert x1 == 576,  f"x1={x1}, expected 576 (0.3 * 1920)"
        assert x2 == 1344, f"x2={x2}, expected 1344 (0.7 * 1920)"
        # y range: 0.1*1080=108 to 0.9*1080=972
        assert y1 == 108,  f"y1={y1}, expected 108 (0.1 * 1080)"
        assert y2 == 972,  f"y2={y2}, expected 972 (0.9 * 1080)"

    def test_padding_expands_symmetrically(self):
        corners = np.array([
            [0.4, 0.3],
            [0.6, 0.3],
            [0.6, 0.7],
            [0.4, 0.7],
        ], dtype=np.float32)
        x1, y1, x2, y2 = self._bbox(corners, 1000, 1000, pad=0.5)
        # Without pad: x=[400,600] w=200, y=[300,700] h=400
        # With 50% pad: x shrinks by 100 each side → [300, 700]
        #                y shrinks by 200 each side → [100, 900]
        assert x1 == 300
        assert x2 == 700
        assert y1 == 100
        assert y2 == 900


# ---------------------------------------------------------------------------
# 7. dewarp_card — corner order (TL/TR/BR/BL → correct perspective warp)
# ---------------------------------------------------------------------------

class TestDewarpCard:
    """Verify dewarp_card interprets corners as (x, y), col0=x, col1=y."""

    def test_dewarps_to_correct_region(self):
        """Create a synthetic image with coloured quadrants and verify
        that the dewarped output maps the expected quadrant to each corner.

        We use an axis-aligned crop as the 'card', so the dewarp should be
        a simple scale — no actual perspective.  If x/y were swapped the
        wrong quadrant would appear in the top-left of the output.

        Colour key (BGR):
            TL = Blue   (255,   0,   0) — only B channel non-zero
            TR = Green  (  0, 255,   0) — only G channel non-zero
            BR = Red    (  0,   0, 255) — only R channel non-zero
            BL = Yellow (  0, 255, 255) — G and R both high, B near zero
        We use a *per-channel threshold* check instead of argmax so that
        Yellow (two equal-high channels) is detected reliably.
        """
        import cv2
        from eval.metrics import dewarp_card

        # 100×100 image, four distinct quadrant colours (BGR)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0:50,  0:50]   = (255,   0,   0)   # TL → Blue
        img[0:50,  50:100] = (  0, 255,   0)   # TR → Green
        img[50:100, 50:100] = (  0,   0, 255)  # BR → Red
        img[50:100,  0:50]  = (  0, 255, 255)  # BL → Yellow

        # Card = full image, corners in (x, y) normalized order: TL, TR, BR, BL
        corners = np.array([
            [0.0, 0.0],   # TL
            [1.0, 0.0],   # TR
            [1.0, 1.0],   # BR
            [0.0, 1.0],   # BL
        ], dtype=np.float32)

        dewarped = dewarp_card(img, corners)
        assert dewarped is not None, "dewarp_card returned None"

        h, w = dewarped.shape[:2]

        def mean_bgr(region):
            return region.mean(axis=(0, 1))  # shape (3,): [B, G, R]

        tl_bgr = mean_bgr(dewarped[:h//2,  :w//2])
        tr_bgr = mean_bgr(dewarped[:h//2,  w//2:])
        br_bgr = mean_bgr(dewarped[h//2:, w//2:])
        bl_bgr = mean_bgr(dewarped[h//2:,  :w//2])

        thresh = 100  # channel must exceed this to be considered "present"

        # TL → Blue: B high, G low, R low
        assert tl_bgr[0] > thresh,  f"TL Blue channel low: {tl_bgr}"
        assert tl_bgr[1] < thresh,  f"TL Green channel high (unexpected): {tl_bgr}"
        assert tl_bgr[2] < thresh,  f"TL Red channel high (unexpected): {tl_bgr}"

        # TR → Green: G high, B low, R low
        assert tr_bgr[1] > thresh,  f"TR Green channel low: {tr_bgr}"
        assert tr_bgr[0] < thresh,  f"TR Blue channel high (unexpected): {tr_bgr}"
        assert tr_bgr[2] < thresh,  f"TR Red channel high (unexpected): {tr_bgr}"

        # BR → Red: R high, B low, G low
        assert br_bgr[2] > thresh,  f"BR Red channel low: {br_bgr}"
        assert br_bgr[0] < thresh,  f"BR Blue channel high (unexpected): {br_bgr}"
        assert br_bgr[1] < thresh,  f"BR Green channel high (unexpected): {br_bgr}"

        # BL → Yellow: G high AND R high, B low
        assert bl_bgr[1] > thresh,  f"BL Green channel low (expected Yellow): {bl_bgr}"
        assert bl_bgr[2] > thresh,  f"BL Red channel low (expected Yellow): {bl_bgr}"
        assert bl_bgr[0] < thresh,  f"BL Blue channel high (unexpected): {bl_bgr}"

    def test_identity_corners_returns_full_image(self):
        """Corners at the four image corners should return the full image."""
        import cv2
        from eval.metrics import dewarp_card, _REF_W, _REF_H

        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (100, 150, 3), dtype=np.uint8)

        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        dewarped = dewarp_card(img, corners)
        assert dewarped is not None
        assert dewarped.shape == (_REF_H, _REF_W, 3), (
            f"Expected ({_REF_H}, {_REF_W}, 3), got {dewarped.shape}"
        )


# ---------------------------------------------------------------------------
# 8. corner_point_error — permutation invariance
# ---------------------------------------------------------------------------

class TestCornerPointError:
    """Verify CPE is invariant to cyclic rotation of the prediction."""

    def test_cyclic_rotations_give_same_cpe(self):
        from eval.metrics import corner_point_error

        true = np.array([
            [0.2, 0.1],
            [0.8, 0.1],
            [0.8, 0.9],
            [0.2, 0.9],
        ], dtype=np.float32)

        # Roll pred by 1 step — CPE should be the same
        pred = np.roll(true, 1, axis=0)
        cpe1 = corner_point_error(true, pred)
        cpe2 = corner_point_error(pred, true)
        assert abs(cpe1 - cpe2) < 1e-6

    def test_perfect_prediction_gives_zero_cpe(self):
        from eval.metrics import corner_point_error

        quad = _portrait_quad()
        assert corner_point_error(quad, quad) < 1e-7

    def test_cyclic_rotation_of_perfect_pred_gives_zero_cpe(self):
        """If pred == roll(true, k) for any k, CPE should still be ~0."""
        from eval.metrics import corner_point_error

        true = _portrait_quad()
        for k in range(4):
            pred = np.roll(true, k, axis=0)
            cpe = corner_point_error(pred, true)
            assert cpe < 1e-6, f"Roll by {k}: CPE should be ~0, got {cpe}"

    def test_non_zero_error_detected(self):
        from eval.metrics import corner_point_error

        true = _portrait_quad()
        pred = true + 0.05  # shift all corners
        cpe = corner_point_error(pred, true)
        assert cpe > 0.01, f"Expected non-zero CPE, got {cpe}"


# ---------------------------------------------------------------------------
# 9. _rotate_corners — round-trip and direction
# ---------------------------------------------------------------------------

class TestRotateCorners:
    """Verify the augmentation rotation transform (dataset.py) is consistent."""

    def _rotate(self, corners, angle_deg):
        """Import and call the private function from dataset.py."""
        sys.path.insert(0, str(_DETECTOR_DIR / "detectors" / "tiny_corner_cnn"))
        from dataset import _rotate_corners
        return _rotate_corners(corners, angle_deg, cx=0.5, cy=0.5)

    def test_360_degree_rotation_is_identity(self):
        """Rotating by 360° should recover the original corners."""
        quad = _portrait_quad()
        rotated = self._rotate(quad, 360.0)
        assert np.allclose(quad, rotated, atol=1e-5), (
            f"360° rotation should be identity.\nExpected:\n{quad}\nGot:\n{rotated}"
        )

    def test_180_degree_rotation_mirrors_through_center(self):
        """180° rotation: each corner should move to (1-x, 1-y)."""
        quad = _portrait_quad()
        rotated = self._rotate(quad, 180.0)
        expected = np.clip(1.0 - quad, 0.0, 1.0)
        assert np.allclose(rotated, expected, atol=1e-5), (
            f"180° rotation should flip through centre.\nExpected:\n{expected}\nGot:\n{rotated}"
        )

    def test_90_degree_ccw_moves_top_left_to_bottom_left(self):
        """PIL TF.rotate(90) is CCW in screen space.

        In image coordinates (y-down), a CCW 90° rotation maps:
            (x, y)  →  (y,  1-x)   [relative to centre (0.5, 0.5)]

        Physically: the top of the image swings left, so the top-left corner
        (x small, y small) ends up at the bottom-left (x small, y large).

        Verified against the actual _rotate_corners implementation.
        """
        corner = np.array([[0.1, 0.1]], dtype=np.float32)
        rotated = self._rotate(corner, 90.0)
        # CCW 90° in image space: (x, y) → (y, 1-x)
        expected = np.array([[0.1, 1.0 - 0.1]], dtype=np.float32)  # (0.1, 0.9)
        assert np.allclose(rotated, expected, atol=1e-5), (
            f"CCW 90°: (0.1, 0.1) → expected {expected}, got {rotated}"
        )

    def test_two_90_degree_rotations_equal_180(self):
        quad = _portrait_quad()
        twice = self._rotate(self._rotate(quad, 90.0), 90.0)
        once180 = self._rotate(quad, 180.0)
        assert np.allclose(twice, once180, atol=1e-4)


# ---------------------------------------------------------------------------
# 10. Visualization drawing — ensure corners * [w, h] uses correct column order
# ---------------------------------------------------------------------------

class TestVisualizationScaling:
    """Verify that the visualization correctly maps col0→x-pixels, col1→y-pixels."""

    def test_corner_to_pixel_mapping(self):
        """Replicate the mapping in visualize_corners.py and verify correctness."""
        corners = np.array([[0.3, 0.7]], dtype=np.float32)
        h, w = 1080, 1920  # as returned by img.shape[:2]

        # This is what visualize_corners.py does:
        pts = (corners * np.array([w, h])).astype(np.int32)

        # col0 * w should give x-pixels
        expected_x = int(0.3 * 1920)  # = 576
        expected_y = int(0.7 * 1080)  # = 756

        # Allow ±1 pixel for float32 truncation (np.float32(0.7)*1080 ≈ 755.999)
        assert abs(pts[0, 0] - expected_x) <= 1, f"x-pixel: {pts[0,0]} != {expected_x}"
        assert abs(pts[0, 1] - expected_y) <= 1, f"y-pixel: {pts[0,1]} != {expected_y}"

    def test_shape_convention_h_w_not_w_h(self):
        """img.shape[:2] returns (H, W) — ensure we unpack in the right order."""
        fake_shape = (1080, 1920, 3)  # H=1080, W=1920
        h, w = fake_shape[:2]
        assert h == 1080, f"h should be 1080 (height), got {h}"
        assert w == 1920, f"w should be 1920 (width), got {w}"

        # Then np.array([w, h]) should be [1920, 1080]
        scale = np.array([w, h])
        assert scale[0] == 1920, f"scale[0] should be width=1920, got {scale[0]}"
        assert scale[1] == 1080, f"scale[1] should be height=1080, got {scale[1]}"

    def test_swapped_scale_gives_different_result_for_non_square(self):
        """If we wrote np.array([h, w]) instead of np.array([w, h]) the pixels
        would be transposed for non-square images."""
        corners = np.array([[0.3, 0.7]], dtype=np.float32)
        h, w = 1080, 1920

        correct = (corners * np.array([w, h])).astype(int)   # [1920, 1080]
        swapped = (corners * np.array([h, w])).astype(int)   # [1080, 1920] — BUG

        assert correct[0, 0] != swapped[0, 0], (
            "Correct and swapped x-pixel should differ for non-square images"
        )
        assert correct[0, 1] != swapped[0, 1], (
            "Correct and swapped y-pixel should differ for non-square images"
        )
        # Correct: x=0.3*1920=576, y=0.7*1080=756
        assert correct[0, 0] == 576
        assert abs(correct[0, 1] - 756) <= 1   # float32: 0.7*1080=755.999 → 755
        # Swapped: x=0.3*1080=324, y=0.7*1920=1344
        assert abs(swapped[0, 0] - 324) <= 1
        assert abs(swapped[0, 1] - 1344) <= 1


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestSortCornersCanonicalWinding,
        TestSortCornersCanonicalShortestEdge,
        TestSortCornersColumnOrder,
        TestCornersPixel,
        TestTwoStageCoordinateMapping,
        TestBboxFromCorners,
        TestDewarpCard,
        TestCornerPointError,
        TestRotateCorners,
        TestVisualizationScaling,
    ]

    passed = failed = 0
    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for method in methods:
            name = f"{cls.__name__}.{method}"
            try:
                getattr(obj, method)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
