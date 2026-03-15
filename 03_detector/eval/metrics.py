"""Evaluation metrics for card corner detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Standard Scryfall card dimensions used as dewarp target
_REF_W = 745
_REF_H = 1040


def dewarp_card(image: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Perspective-warp the detected card region to a flat _REF_W × _REF_H rectangle.

    Args:
        image:   HxWx3 uint8 BGR image.
        corners: shape (4, 2), normalized [0, 1], order TL/TR/BR/BL.

    Returns:
        Dewarped uint8 BGR image of shape (_REF_H, _REF_W, 3), or None if warp fails.
    """
    import cv2
    h, w = image.shape[:2]
    src = (corners * np.array([w, h], dtype=np.float32)).astype(np.float32)
    dst = np.array([
        [0,       0      ],
        [_REF_W,  0      ],
        [_REF_W,  _REF_H ],
        [0,       _REF_H ],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (_REF_W, _REF_H))


def phash_distance(
    image: np.ndarray,
    pred_corners: np.ndarray,
    ref_img_path: Path,
    hash_size: int = 8,
) -> int | None:
    """Compute pHash Hamming distance between dewarped prediction and Scryfall reference.

    Dewarp the predicted card region and compare its perceptual hash against
    the reference PNG's hash.  Both the normal orientation and the 180° rotation
    are tried; the minimum distance is returned to handle upside-down cards.

    Args:
        image:         Original HxWx3 uint8 BGR image.
        pred_corners:  shape (4, 2), normalized [0, 1], CW order.
        ref_img_path:  Path to Scryfall reference PNG for this card.
        hash_size:     pHash grid size (default 8 → 64-bit hash).

    Returns:
        Hamming distance in [0, 64], or None if the reference is missing or warp fails.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return None

    if not ref_img_path.exists():
        return None

    import cv2
    ref_pil  = Image.open(ref_img_path).convert("RGB")
    ref_hash = imagehash.phash(ref_pil, hash_size=hash_size)

    # Corners arrive in canonical form (CW, shortest edge at 0→1 — see base.py).
    # Try both 0° and 180° (roll by 2) to resolve the top-vs-bottom ambiguity.
    best_dist = None
    for rot in [0, 2]:
        corners = np.roll(pred_corners, -rot, axis=0)
        dewarped = dewarp_card(image, corners)
        if dewarped is None:
            continue
        pred_pil  = Image.fromarray(cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB))
        pred_hash = imagehash.phash(pred_pil, hash_size=hash_size)
        dist = int(pred_hash - ref_hash)
        if best_dist is None or dist < best_dist:
            best_dist = dist

    return best_dist


def corner_point_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Permutation-invariant mean corner error (minimum over 4 cyclic rotations).

    Detectors output corners in canonical CW order with the shortest edge at
    index 0→1 (see base.py).  Small prediction errors can cause the canonical
    sort to pick a different starting corner than the ground truth, inflating a
    naïve element-wise CPE.  Taking the minimum over all 4 cyclic rotations of
    pred makes the metric robust to this one-position shift.

    Args:
        pred: shape (4, 2), normalized [0, 1] corner coordinates.
        true: shape (4, 2), normalized [0, 1] corner coordinates.

    Returns:
        Minimum mean Euclidean distance across all 4 cyclic rotations of pred.
    """
    return float(min(
        np.mean(np.linalg.norm(np.roll(pred, -i, axis=0) - true, axis=1))
        for i in range(4)
    ))


def pck(pred: np.ndarray, true: np.ndarray, threshold: float = 0.05) -> float:
    """Percentage of Correct Keypoints within threshold (permutation-invariant).

    Uses the same best-cyclic-rotation logic as corner_point_error to avoid
    inflating error when canonical sort picks a different starting corner.

    threshold=0.05 means within 5% of the image diagonal.

    Args:
        pred:      shape (4, 2), normalized [0, 1].
        true:      shape (4, 2), normalized [0, 1].
        threshold: Fraction of image diagonal allowed (default 0.05 = 5%).

    Returns:
        Fraction in [0, 1] of corners within the threshold distance.
    """
    diag = np.sqrt(2.0)   # normalized diagonal of a unit square
    best = 0.0
    for i in range(4):
        dists = np.linalg.norm(np.roll(pred, -i, axis=0) - true, axis=1)
        best = max(best, float(np.mean(dists < threshold * diag)))
    return best


def quad_iou(pred_corners: np.ndarray, true_corners: np.ndarray, img_w: int, img_h: int) -> float:
    """IoU between predicted and true quadrilateral (using pixel coords).

    Approximated via bounding-box IoU for efficiency; exact polygon IoU
    requires shapely and is implemented in quad_iou_exact().

    Args:
        pred_corners: shape (4, 2), normalized [0, 1].
        true_corners: shape (4, 2), normalized [0, 1].
        img_w:        Image width in pixels.
        img_h:        Image height in pixels.

    Returns:
        Bounding-box IoU in [0, 1].
    """
    def bbox(corners, w, h):
        px = corners * np.array([w, h])
        return px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()

    px1, py1, px2, py2 = bbox(pred_corners, img_w, img_h)
    tx1, ty1, tx2, ty2 = bbox(true_corners, img_w, img_h)

    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = (px2 - px1) * (py2 - py1)
    area_t = (tx2 - tx1) * (ty2 - ty1)
    union = area_p + area_t - inter
    return float(inter / union) if union > 0 else 0.0


def quad_iou_exact(pred_corners: np.ndarray, true_corners: np.ndarray, img_w: int, img_h: int) -> float:
    """Exact polygon IoU using shapely (optional dependency).

    Falls back to quad_iou() if shapely is not installed.

    Args:
        pred_corners: shape (4, 2), normalized [0, 1].
        true_corners: shape (4, 2), normalized [0, 1].
        img_w:        Image width in pixels.
        img_h:        Image height in pixels.

    Returns:
        Exact polygon IoU in [0, 1].
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return quad_iou(pred_corners, true_corners, img_w, img_h)
    scale = np.array([img_w, img_h])
    p = Polygon((pred_corners * scale).tolist())
    t = Polygon((true_corners * scale).tolist())
    if not p.is_valid or not t.is_valid:
        return 0.0
    inter = p.intersection(t).area
    union = p.union(t).area
    return float(inter / union) if union > 0 else 0.0
