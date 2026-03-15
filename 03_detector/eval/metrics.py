"""Evaluation metrics for card corner detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Standard Scryfall card dimensions used as dewarp target
_REF_W = 745
_REF_H = 1040


def _orient_short_edge_top(corners: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Roll CW corners so the shortest edge (card width) is at position 0→1.

    MTG cards are portrait (63×88mm); the short edges are top and bottom.
    Placing the shortest edge first ensures portrait orientation regardless
    of how the card is held in the frame.

    Args:
        corners: shape (4, 2), normalized [0, 1], CW order, any starting corner.
        img_w:   image width in pixels (for correct aspect-aware lengths).
        img_h:   image height in pixels.

    Returns:
        Reordered corners, shape (4, 2), shortest edge at index 0→1.
    """
    scale = np.array([img_w, img_h], dtype=np.float32)
    pts = corners * scale  # pixel coords
    edge_lens = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    shortest = int(np.argmin(edge_lens))
    return np.roll(corners, -shortest, axis=0)


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
    h, w = image.shape[:2]
    ref_pil  = Image.open(ref_img_path).convert("RGB")
    ref_hash = imagehash.phash(ref_pil, hash_size=hash_size)

    # Orient so the shortest edge (card top/bottom) is at position 0→1, then
    # try both 0° and 180° to resolve the top-vs-bottom ambiguity.
    oriented = _orient_short_edge_top(pred_corners, w, h)
    best_dist = None
    for rot in [0, 2]:  # 0° and 180°
        corners = np.roll(oriented, -rot, axis=0)
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
    """Mean Euclidean distance between predicted and true corners (normalized units).

    Args:
        pred: shape (4, 2), normalized [0, 1] corner coordinates.
        true: shape (4, 2), normalized [0, 1] corner coordinates.

    Returns:
        Mean distance across all 4 corners, in normalized units.
    """
    # pred, true: shape (4, 2), normalized [0,1]
    return float(np.mean(np.linalg.norm(pred - true, axis=1)))


def pck(pred: np.ndarray, true: np.ndarray, threshold: float = 0.05) -> float:
    """Percentage of Correct Keypoints within threshold of image diagonal (normalized).

    threshold=0.05 means within 5% of the image diagonal.

    Args:
        pred:      shape (4, 2), normalized [0, 1].
        true:      shape (4, 2), normalized [0, 1].
        threshold: Fraction of image diagonal allowed (default 0.05 = 5%).

    Returns:
        Fraction in [0, 1] of corners within the threshold distance.
    """
    dists = np.linalg.norm(pred - true, axis=1)  # (4,)
    diag = np.sqrt(2.0)   # normalized diagonal of a unit square
    return float(np.mean(dists < threshold * diag))


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
