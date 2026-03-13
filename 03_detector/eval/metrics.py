"""Evaluation metrics for card corner detection."""
import numpy as np


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
