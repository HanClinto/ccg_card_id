#!/usr/bin/env python3
"""Train the neural card corner detector.

Training sources
----------------
  packopening (default, recommended)
    380k+ SIFT-verified labeled frames from pack-opening videos.
    Negatives sampled from unmatched extracted frames.
    Validation: 5% holdout from packopening.
    Test: clint_cards_with_backgrounds (separate domain — domain-generalization eval).

  clint (ablation)
    ~1267 frames from clint_cards_with_backgrounds, 15% val split.
    Use only to measure the effect of training data size/domain.

Usage:
  # Train on packopening (default), seed backbone from card-ID checkpoint
  python 03_detector/detectors/tiny_corner_cnn/train.py \\
      --seed-checkpoint $DATA/results/mobilevit_xxs/mobilevit_xxs_illustration_id_128d/last.pt

  # Ablation: train on clint only
  python 03_detector/detectors/tiny_corner_cnn/train.py --train-source clint

  # Resume from last.pt
  python 03_detector/detectors/tiny_corner_cnn/train.py --epochs 10

  # Start from scratch
  python 03_detector/detectors/tiny_corner_cnn/train.py --rebuild

Checkpoints: <results-dir>/epoch_XXXX.pt  and  last.pt
Format: {"model": state_dict, "optimizer": state_dict, "epoch": N, "val_loss": v}
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccg_card_id.config import cfg

# Local imports (relative to this file's directory)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model import TinyCornerCNN, TinyCornerCNNDirect, MobileViTCornerDetector, make_gaussian_heatmaps   # noqa: E402
from dataset import (  # noqa: E402
    CornerDataset,
    load_from_packopening_db,
    load_from_clint_csv,
    load_clint_as_test,
)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def detection_loss(
    pred_corners: torch.Tensor,
    pred_presence_logit: torch.Tensor,
    pred_heatmaps: torch.Tensor | None,
    true_corners: torch.Tensor,
    true_presence: torch.Tensor,
    lambda_corners: float = 5.0,
    lambda_heatmap: float = 10.0,
    heatmap_sigma: float = 2.0,
) -> torch.Tensor:
    """Combined presence + corner regression + heatmap auxiliary loss.

    Corners are supervised in canonical TL→TR→BR→BL order (no permutation
    invariance needed — the dataset ensures canonical GT order, and
    augmentation re-sorts after flips/rotations).

    Args:
        pred_corners:        (B, 8) soft-argmax corner coords from heatmaps.
        pred_presence_logit: (B,)   raw presence logit.
        pred_heatmaps:       (B, 4, H, W) raw corner heatmap logits, or None
                             for MobileViTCornerDetector which uses direct regression.
        true_corners:        (B, 8) GT corners in canonical order (zeros for negatives).
        true_presence:       (B,)   binary float label (1.0 = card present).
        lambda_corners:      Weight on coordinate regression term (default 5.0).
        lambda_heatmap:      Weight on heatmap auxiliary term (default 10.0).
        heatmap_sigma:       Gaussian sigma for heatmap targets in heatmap pixels (default 2.0).

    Returns:
        Scalar loss tensor.
    """
    presence_loss = F.binary_cross_entropy_with_logits(pred_presence_logit, true_presence)
    pos_mask = true_presence.bool()

    corner_loss  = torch.tensor(0.0, device=pred_corners.device)
    heatmap_loss = torch.tensor(0.0, device=pred_corners.device)

    if pos_mask.any():
        p = pred_corners[pos_mask]   # (N, 8)
        t = true_corners[pos_mask]   # (N, 8)
        corner_loss = F.smooth_l1_loss(p, t)

        if pred_heatmaps is not None:
            hm_pred = pred_heatmaps[pos_mask]                        # (N, 4, H, W)
            t_4x2   = t.view(-1, 4, 2)
            hm_gt   = make_gaussian_heatmaps(
                t_4x2,
                H=hm_pred.shape[2],
                W=hm_pred.shape[3],
                sigma=heatmap_sigma,
            ).to(pred_corners.device)                                # (N, 4, H, W)
            heatmap_loss = F.binary_cross_entropy_with_logits(hm_pred, hm_gt)

    return presence_loss + lambda_corners * corner_loss + lambda_heatmap * heatmap_loss


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _val_cpe(pred_corners: torch.Tensor, true_corners: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean Corner Point Error (normalized units) on positive examples."""
    if not mask.any():
        return 0.0
    pc = pred_corners[mask].detach().cpu().numpy().reshape(-1, 4, 2)
    tc = true_corners[mask].detach().cpu().numpy().reshape(-1, 4, 2)
    import numpy as np
    dists = np.linalg.norm(pc - tc, axis=2)  # (N, 4)
    return float(dists.mean())


def _val_presence_acc(pred_logit: torch.Tensor, true_presence: torch.Tensor) -> float:
    """Presence classification accuracy."""
    pred_prob = torch.sigmoid(pred_logit).detach()
    pred_label = (pred_prob > 0.5).float()
    return float((pred_label == true_presence).float().mean())


def _quad_iou(pred: "np.ndarray", true: "np.ndarray") -> float:
    """IoU between two convex quadrilaterals (each shape (4, 2), normalized coords).

    Uses cv2.intersectConvexConvex.  Returns 0.0 if either polygon has zero area.
    """
    import cv2
    import numpy as np
    # Take convex hull of predicted corners so IoU is order-invariant.
    # Without this, a bowtie/cross ordering (common in early training) gives
    # near-zero contourArea even when corners are spatially correct.
    p_hull = cv2.convexHull(pred.astype(np.float32)).reshape(-1, 2)
    t = true.astype(np.float32)
    retval, inter_pts = cv2.intersectConvexConvex(p_hull, t)
    if retval == 0 or inter_pts is None or len(inter_pts) == 0:
        return 0.0
    inter = float(cv2.contourArea(inter_pts))
    area_p = float(cv2.contourArea(p_hull))
    area_t = float(cv2.contourArea(t))
    union = area_p + area_t - inter
    return inter / union if union > 0 else 0.0


def _val_iou(pred_corners: torch.Tensor, true_corners: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean quadrilateral IoU on positive examples (normalized coords, so scale-invariant)."""
    if not mask.any():
        return 0.0
    import numpy as np
    pc = pred_corners[mask].detach().cpu().numpy().reshape(-1, 4, 2)
    tc = true_corners[mask].detach().cpu().numpy().reshape(-1, 4, 2)
    ious = [_quad_iou(p, t) for p, t in zip(pc, tc)]
    return float(np.mean(ious))


# ---------------------------------------------------------------------------
# pHash-based dewarp quality metric
# ---------------------------------------------------------------------------

_IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

import re as _re
_UUID_RE = _re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', _re.I
)


def _ref_image_path(card_id: str, data_dir: Path) -> Path:
    """Return the reference image path for a card_id (MTG UUID or Pokemon TCG id)."""
    if _UUID_RE.match(card_id):
        return (data_dir / "catalog" / "scryfall" / "images" / "png" / "front"
                / card_id[0] / card_id[1] / f"{card_id}.png")
    else:
        # Pokemon TCG id, e.g. "base1-4", "sv10-123"
        return (data_dir / "catalog" / "pokemontcg" / "images" / "large"
                / card_id[0] / card_id[1] / f"{card_id}.png")


def build_ref_phash_dict(card_ids: set[str], data_dir: Path, cache_dir: Path | None = None) -> dict:
    """Precompute pHash for each unique card_id from reference images.

    Supports both Scryfall (UUID format) and Pokemon TCG card IDs.
    Returns {card_id: imagehash.ImageHash} for every card whose reference image
    exists on disk.  Cards not found are silently omitted.

    Results are cached to {cache_dir}/ref_phash_cache.pkl so subsequent runs
    skip the slow PNG reads.  New card_ids not in the cache are computed and
    merged in incrementally.
    """
    import pickle

    try:
        import imagehash
        from PIL import Image as _PILImage
    except ImportError:
        return {}

    # Load existing cache
    cache_path = (cache_dir or data_dir) / "ref_phash_cache.pkl"
    cached: dict = {}
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
        except Exception:
            cached = {}

    result = {}
    new_entries = 0
    for card_id in card_ids:
        if not card_id:
            continue
        if card_id in cached:
            result[card_id] = cached[card_id]
            continue
        img_path = _ref_image_path(card_id, data_dir)
        if img_path.exists():
            try:
                result[card_id] = imagehash.phash(_PILImage.open(img_path).convert("RGB"))
                cached[card_id] = result[card_id]
                new_entries += 1
            except Exception:
                pass

    # Persist any newly computed entries
    if new_entries > 0:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(cached, f)
        except Exception:
            pass

    return result


def batch_phash_dists(
    pred_corners: torch.Tensor,   # (B, 8) normalized [0, 1]
    images: torch.Tensor,         # (B, 3, H, W) ImageNet-normalized, on any device
    card_ids: list[str],          # length B
    presence_mask: torch.Tensor,  # (B,) bool — skip negatives
    ref_phash_dict: dict,
) -> tuple[int, int]:
    """Dewarp each frame using predicted corners, compute pHash distance to reference.

    Returns (sum_dist, n_evaluated).  Only frames where card_id is in
    ref_phash_dict and presence_mask is True are evaluated.  Divide to get
    mean pHash distance — lower is better (0 = perfect match, ~32 = random).

    The input images are denormalized from ImageNet stats and the dewarp is
    applied in the input pixel space, outputting a 224×224 crop that is hashed.
    """
    try:
        import cv2
        import imagehash
        import numpy as np
        from PIL import Image as _PILImage
    except ImportError:
        return 0, 0

    if not ref_phash_dict:
        return 0, 0

    B, _, H, W = images.shape
    # Denormalize to RGB uint8
    imgs_cpu = images.cpu()
    imgs_np = (imgs_cpu * _IMAGENET_STD_T + _IMAGENET_MEAN_T).clamp(0, 1)
    imgs_np = (imgs_np.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)  # (B,H,W,3) RGB

    corners_np = pred_corners.detach().cpu().numpy().reshape(B, 4, 2)  # normalized
    corners_px = corners_np * np.array([W, H], dtype=np.float32)       # pixel coords

    dst_pts = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])

    sum_dist = n_total = 0
    pres = presence_mask.cpu().numpy()
    for i, card_id in enumerate(card_ids):
        if not pres[i]:
            continue
        ref_ph = ref_phash_dict.get(card_id)
        if ref_ph is None:
            continue
        n_total += 1
        src_pts = corners_px[i].astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        dewarped = cv2.warpPerspective(imgs_np[i], M, (224, 224))
        try:
            pred_ph = imagehash.phash(_PILImage.fromarray(dewarped))
            sum_dist += int(pred_ph - ref_ph)
        except Exception:
            pass

    return sum_dist, n_total


# Keep old name as alias for eval_checkpoints.py compatibility
def batch_phash_hits(
    pred_corners, images, card_ids, presence_mask, ref_phash_dict, threshold=10,
):
    """Legacy wrapper — returns (n_hits, n_evaluated) at given threshold."""
    sum_dist, n_total = batch_phash_dists(
        pred_corners, images, card_ids, presence_mask, ref_phash_dict
    )
    if n_total == 0:
        return 0, 0
    # Re-compute hits from individual distances requires the loop, so approximate
    # by re-running (or just keep for backward compat of eval_checkpoints).
    # For new code use batch_phash_dists directly.
    try:
        import cv2
        import imagehash
        import numpy as np
        from PIL import Image as _PILImage
    except ImportError:
        return 0, 0
    B, _, H, W = images.shape
    imgs_cpu = images.cpu()
    imgs_np = (imgs_cpu * _IMAGENET_STD_T + _IMAGENET_MEAN_T).clamp(0, 1)
    imgs_np = (imgs_np.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    corners_np = pred_corners.detach().cpu().numpy().reshape(B, 4, 2)
    corners_px = corners_np * np.array([W, H], dtype=np.float32)
    dst_pts = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])
    n_ok = n_total = 0
    pres = presence_mask.cpu().numpy()
    for i, card_id in enumerate(card_ids):
        if not pres[i]:
            continue
        ref_ph = ref_phash_dict.get(card_id)
        if ref_ph is None:
            continue
        n_total += 1
        src_pts = corners_px[i].astype(np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        dewarped = cv2.warpPerspective(imgs_np[i], M, (224, 224))
        try:
            pred_ph = imagehash.phash(_PILImage.fromarray(dewarped))
            if (pred_ph - ref_ph) <= threshold:
                n_ok += 1
        except Exception:
            pass
    return n_ok, n_total


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def pick_device(cpu: bool = False) -> torch.device:
    if cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Run naming and history logging
# ---------------------------------------------------------------------------

def _make_run_name(args: argparse.Namespace) -> str:
    """Build a self-documenting run name from training args.

    Convention: {backbone}_{head}_{loss_cfg}_{input}_{data_filter}[_{seed}{_lr}{_fzN}]
      backbone   : tcnn  (TinyCornerCNN/TinyCornerCNNDirect) | mvit (MobileViTCornerDetector)
      head       : softargmax | directreg | spatialreg4 (4×4 pool)
      loss_cfg   : lc{N}lh{N}  e.g. lc5lh0; direct/mvit omit lh
      input      : img{size}   e.g. img448
      data_filter: ph{N}       (packopening, phash_dist≤N) | clint
      seed_tag   : _seedin (ImageNet) | _seedcid (ArcFace card-ID) [mobilevit only]
      lr_tag     : _blrNN (backbone LR scale) [mobilevit only]
      freeze_tag : _fzN   (N frozen backbone epochs) [mobilevit only]

    Examples:
      tcnn_softargmax_lc5lh0_img448_ph10
      tcnn_directreg_lc5_img448_ph10
      mvit_spatialreg4_lc5_img448_ph10_seedin_blr10_fz2
    """
    if args.arch == "tiny":
        backbone, head = "tcnn", "softargmax"
    elif args.arch == "tiny-direct":
        backbone, head = "tcnn", "directreg"
    else:
        backbone, head = "mvit", "spatialreg4"

    lc = int(args.lambda_corners) if args.lambda_corners == int(args.lambda_corners) else args.lambda_corners
    loss_cfg = f"lc{lc}"
    if args.arch == "tiny":
        lh = int(args.lambda_heatmap) if args.lambda_heatmap == int(args.lambda_heatmap) else args.lambda_heatmap
        loss_cfg += f"lh{lh}"

    from dataset import INPUT_SIZE
    input_tag   = f"img{INPUT_SIZE}"
    data_filter = f"ph{args.max_phash_dist}" if args.train_source == "packopening" else "clint"

    # For mobilevit, encode seed source, backbone LR scale, and freeze epochs
    seed_tag = ""
    if args.arch == "mobilevit":
        seed_tag = "_seedcid" if args.seed_checkpoint is not None else "_seedin"
        if args.backbone_lr_scale != 1.0:
            scale_str = f"{args.backbone_lr_scale:.2f}".replace("0.", "").replace(".", "")
            seed_tag += f"_blr{scale_str}"
        if args.freeze_backbone_epochs > 0:
            seed_tag += f"_fz{args.freeze_backbone_epochs}"

    return f"{backbone}_{head}_{loss_cfg}_{input_tag}_{data_filter}{seed_tag}"


_HISTORY_COLS = [
    "run_name", "epoch", "timestamp",
    "arch", "lambda_corners", "lambda_heatmap", "batch_size", "lr_start",
    "train_source", "max_phash_dist",
    "train_loss", "val_loss", "val_cpe", "val_iou", "val_pres_acc",
    "val_mean_phash_dist",
    "test_cpe", "test_iou", "test_pres_acc", "test_mean_phash_dist",
    "lr_end", "checkpoint_saved",
]


def _append_history_row(
    history_csv: Path,
    run_name: str,
    epoch: int,
    args: argparse.Namespace,
    train_loss: float,
    val_loss: float,
    val_cpe: float,
    val_iou: float,
    val_pres_acc: float,
    val_mean_phash_dist: float | None,
    test_cpe: float | None,
    test_iou: float | None,
    test_pres_acc: float | None,
    test_mean_phash_dist: float | None,
    lr_end: float,
    checkpoint_saved: bool,
) -> None:
    """Append one epoch row to the shared training history CSV."""
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_csv.exists()
    with history_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_name":              run_name,
            "epoch":                 epoch,
            "timestamp":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "arch":                  args.arch,
            "lambda_corners":        args.lambda_corners,
            "lambda_heatmap":        args.lambda_heatmap,
            "batch_size":            args.batch_size,
            "lr_start":              args.lr,
            "train_source":          args.train_source,
            "max_phash_dist":        args.max_phash_dist if args.train_source == "packopening" else "",
            "train_loss":            f"{train_loss:.6f}",
            "val_loss":              f"{val_loss:.6f}",
            "val_cpe":               f"{val_cpe:.6f}",
            "val_iou":               f"{val_iou:.6f}",
            "val_pres_acc":          f"{val_pres_acc:.6f}",
            "val_mean_phash_dist":   f"{val_mean_phash_dist:.2f}" if val_mean_phash_dist is not None else "",
            "test_cpe":              f"{test_cpe:.6f}" if test_cpe is not None else "",
            "test_iou":              f"{test_iou:.6f}" if test_iou is not None else "",
            "test_pres_acc":         f"{test_pres_acc:.6f}" if test_pres_acc is not None else "",
            "test_mean_phash_dist":  f"{test_mean_phash_dist:.2f}" if test_mean_phash_dist is not None else "",
            "lr_end":                f"{lr_end:.6e}",
            "checkpoint_saved":      int(checkpoint_saved),
        })


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    data_dir      = args.data_dir
    fast_data_dir = args.fast_data_dir
    run_name      = _make_run_name(args)
    results_dir   = args.results_dir or (data_dir / "results" / "corner_detector" / run_name)
    history_csv   = data_dir / "results" / "corner_detector" / "training_history.csv"

    if fast_data_dir is not None:
        print(f"fast_data_dir: {fast_data_dir}  (cache)")
    else:
        print("fast_data_dir: (none — loading from data_dir)")

    print(f"run_name     : {run_name}")
    print(f"data_dir     : {data_dir}")
    print(f"train_source : {args.train_source}")
    print(f"results_dir  : {results_dir}")
    print(f"history_csv  : {history_csv}")

    clint_csv = args.clint_corners_csv
    clint_neg = args.clint_neg_dir

    if args.train_source == "packopening":
        db_path = args.packopening_db
        if not db_path.exists():
            raise FileNotFoundError(f"packopening DB not found: {db_path}")
        train_rows, val_rows = load_from_packopening_db(
            db_path, data_dir,
            neg_sample_n=0 if args.positives_only else args.neg_sample_n,
            val_frac=0.05,
            max_phash_dist=args.max_phash_dist,
        )
        # Always evaluate on clint as the held-out test domain
        test_rows = load_clint_as_test(clint_csv, clint_neg, data_dir) if clint_csv.exists() else []
        print(f"clint test   : {len(test_rows)} frames (domain-generalization eval)")
    else:
        # clint ablation: train/val split within clint, no separate test set
        if not clint_csv.exists():
            raise FileNotFoundError(f"corners.csv not found: {clint_csv}")
        train_rows, val_rows = load_from_clint_csv(clint_csv, clint_neg, data_dir)
        test_rows = []

    train_ds = CornerDataset(train_rows, data_dir, augment=True,  fast_data_dir=fast_data_dir)
    val_ds   = CornerDataset(val_rows,   data_dir, augment=False, fast_data_dir=fast_data_dir)
    test_ds  = CornerDataset(test_rows,  data_dir, augment=False) if test_rows else None

    # Precompute reference pHashes for all unique card_ids in val + test sets.
    # Used each epoch to measure dewarp quality independently of SIFT label noise.
    _all_eval_card_ids = {r.get("card_id", "") for r in val_rows + test_rows if r.get("card_id")}
    print(f"building ref pHash dict: {len(_all_eval_card_ids)} unique cards...", end=" ", flush=True)
    ref_phash_dict = build_ref_phash_dict(_all_eval_card_ids, data_dir, cache_dir=args.fast_data_dir)
    print(f"{len(ref_phash_dict)} found")

    _dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=False,          # MPS tensors can't use shared memory
        persistent_workers=(args.num_workers > 0),
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **_dl_kwargs,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **_dl_kwargs,
    )
    test_dl = (DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **_dl_kwargs)
               if test_ds else None)

    device = pick_device(args.cpu)
    print(f"device      : {device}")
    print(f"arch        : {args.arch}")

    if args.arch == "tiny":
        model = TinyCornerCNN().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {params:,} parameters ({params*4/1024:.0f} KB fp32)")
        if args.seed_checkpoint is not None:
            print("WARNING: --seed-checkpoint ignored for tiny arch (no compatible backbone)")
    elif args.arch == "tiny-direct":
        model = TinyCornerCNNDirect().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {params:,} parameters ({params*4/1024:.0f} KB fp32)")
        if args.seed_checkpoint is not None:
            print("WARNING: --seed-checkpoint ignored for tiny-direct arch (no compatible backbone)")
    else:
        model = MobileViTCornerDetector(pretrained_backbone=True).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {params:,} parameters ({params*4/1024**2:.1f} MB fp32)")
        if args.seed_checkpoint is not None:
            print(f"seeding backbone from {args.seed_checkpoint}")
            model.load_card_id_checkpoint(args.seed_checkpoint)
        if args.freeze_backbone_epochs > 0:
            print(f"backbone frozen for first {args.freeze_backbone_epochs} epoch(s)")

    # Differential LR: for mobilevit, use a lower LR for the pretrained backbone
    # to avoid catastrophic forgetting while the head learns quickly.
    backbone_lr = args.lr * args.backbone_lr_scale
    if args.arch == "mobilevit" and args.backbone_lr_scale != 1.0:
        head_params     = list(model.spatial_pool.parameters()) + list(model.head.parameters())
        backbone_params = list(model.backbone.parameters())
        optim = torch.optim.AdamW([
            {"params": backbone_params, "lr": backbone_lr,  "name": "backbone"},
            {"params": head_params,     "lr": args.lr,      "name": "head"},
        ], weight_decay=1e-4)
        print(f"lr          : backbone={backbone_lr:.2e}  head={args.lr:.2e}")
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"lr          : {args.lr:.2e}")

    results_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = results_dir / "last.pt"

    start_epoch = 1

    # Auto-resume from last.pt unless --rebuild
    if not args.rebuild and last_ckpt.exists():
        print(f"auto-resume: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        # Reset LR to args.lr so the new cosine cycle starts correctly.
        for pg in optim.param_groups:
            if pg.get("name") == "backbone":
                pg["lr"] = backbone_lr
            else:
                pg["lr"] = args.lr
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"resuming from epoch {start_epoch - 1}")

    end_epoch = start_epoch + args.epochs - 1
    total_steps = (end_epoch - start_epoch + 1) * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps, eta_min=args.lr * 0.01,
    )
    print(f"cosine LR   : {args.lr:.2e} → {args.lr * 0.01:.2e} over {total_steps} steps")

    for epoch in range(start_epoch, end_epoch + 1):
        # Staged backbone freeze for mobilevit: keep backbone frozen for the
        # first N epochs so the randomly-initialised head can anchor onto
        # pretrained features before we start adapting the backbone.
        if args.arch == "mobilevit" and args.freeze_backbone_epochs > 0:
            freeze = epoch <= args.freeze_backbone_epochs
            for p in model.backbone.parameters():
                p.requires_grad_(not freeze)
            if epoch == args.freeze_backbone_epochs + 1:
                print(f"epoch {epoch}: backbone unfrozen, switching to differential LR")

        # ---- train ----
        model.train()
        total_loss = 0.0
        n_train = 0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{end_epoch}", unit="batch"):
            images   = batch["image"].to(device)
            presence = batch["card_present"].to(device)
            corners  = batch["corners"].to(device)

            out = model(images)
            pred_corners, pred_presence = out[0], out[1]
            pred_heatmaps = out[2] if len(out) == 3 else None
            loss = detection_loss(
                pred_corners, pred_presence, pred_heatmaps, corners, presence,
                args.lambda_corners, args.lambda_heatmap,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            scheduler.step()

            total_loss += loss.item() * images.shape[0]
            n_train    += images.shape[0]

        train_loss = total_loss / max(1, n_train)

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        all_pred_corners:  list[torch.Tensor] = []
        all_true_corners:  list[torch.Tensor] = []
        all_pred_presence: list[torch.Tensor] = []
        all_true_presence: list[torch.Tensor] = []

        val_ph_sum = val_ph_total = 0
        with torch.no_grad():
            for batch in val_dl:
                images   = batch["image"].to(device)
                presence = batch["card_present"].to(device)
                corners  = batch["corners"].to(device)
                card_ids = batch.get("card_id", [])

                out = model(images)
                pred_corners, pred_presence = out[0], out[1]
                pred_heatmaps = out[2] if len(out) == 3 else None
                loss = detection_loss(
                    pred_corners, pred_presence, pred_heatmaps, corners, presence,
                    args.lambda_corners, args.lambda_heatmap,
                )
                val_loss_sum += loss.item() * images.shape[0]
                n_val        += images.shape[0]

                all_pred_corners.append(pred_corners.cpu())
                all_true_corners.append(corners.cpu())
                all_pred_presence.append(pred_presence.cpu())
                all_true_presence.append(presence.cpu())

                if ref_phash_dict and card_ids:
                    s, n = batch_phash_dists(
                        pred_corners, images, card_ids, presence.bool(),
                        ref_phash_dict,
                    )
                    val_ph_sum += s; val_ph_total += n

        val_loss = val_loss_sum / max(1, n_val)

        pc_all = torch.cat(all_pred_corners)
        tc_all = torch.cat(all_true_corners)
        pp_all = torch.cat(all_pred_presence)
        tp_all = torch.cat(all_true_presence)

        cpe                 = _val_cpe(pc_all, tc_all, tp_all.bool())
        iou                 = _val_iou(pc_all, tc_all, tp_all.bool())
        pres_acc            = _val_presence_acc(pp_all, tp_all)
        val_mean_phash_dist = val_ph_sum / val_ph_total if val_ph_total > 0 else None
        cur_lr              = optim.param_groups[0]["lr"]

        # Optionally evaluate on clint test set (domain generalization)
        test_cpe = test_iou = test_pres_acc = test_mean_phash_dist = None
        if test_dl is not None:
            model.eval()
            t_pc, t_tc, t_pp, t_tp = [], [], [], []
            t_ph_sum = t_ph_total = 0
            with torch.no_grad():
                for batch in test_dl:
                    images   = batch["image"].to(device)
                    presence = batch["card_present"].to(device)
                    corners  = batch["corners"].to(device)
                    card_ids = batch.get("card_id", [])
                    t_out = model(images)
                    pc, pp = t_out[0], t_out[1]
                    t_pc.append(pc.cpu()); t_tc.append(corners.cpu())
                    t_pp.append(pp.cpu()); t_tp.append(presence.cpu())
                    if ref_phash_dict and card_ids:
                        s, n = batch_phash_dists(
                            pc, images, card_ids, presence.bool(),
                            ref_phash_dict,
                        )
                        t_ph_sum += s; t_ph_total += n
            t_pc_all = torch.cat(t_pc); t_tc_all = torch.cat(t_tc)
            t_pp_all = torch.cat(t_pp); t_tp_all = torch.cat(t_tp)
            test_cpe            = _val_cpe(t_pc_all, t_tc_all, t_tp_all.bool())
            test_iou            = _val_iou(t_pc_all, t_tc_all, t_tp_all.bool())
            test_pres_acc       = _val_presence_acc(t_pp_all, t_tp_all)
            test_mean_phash_dist = t_ph_sum / t_ph_total if t_ph_total > 0 else None

        val_ph_str  = f"  val_phash_dist={val_mean_phash_dist:.1f}" if val_mean_phash_dist is not None else ""
        test_str    = ""
        if test_cpe is not None:
            test_str = f"  test_cpe={test_cpe:.4f}  test_iou={test_iou:.3f}  test_pres_acc={test_pres_acc:.3f}"
            if test_mean_phash_dist is not None:
                test_str += f"  test_phash_dist={test_mean_phash_dist:.1f}"
        print(
            f"epoch={epoch:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_cpe={cpe:.4f}  "
            f"val_iou={iou:.3f}  "
            f"val_pres_acc={pres_acc:.3f}"
            f"{val_ph_str}"
            f"{test_str}  "
            f"lr={cur_lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "arch": args.arch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(ckpt, last_ckpt)
        saved_epoch_ckpt = args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0
        if saved_epoch_ckpt:
            torch.save(ckpt, results_dir / f"epoch_{epoch:04d}.pt")

        _append_history_row(
            history_csv, run_name, epoch, args,
            train_loss, val_loss, cpe, iou, pres_acc, val_mean_phash_dist,
            test_cpe, test_iou, test_pres_acc, test_mean_phash_dist,
            lr_end=cur_lr,
            checkpoint_saved=saved_epoch_ckpt,
        )

    print(f"Training complete. Checkpoints in: {results_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    _data_dir = cfg.data_dir
    p = argparse.ArgumentParser(
        description="Train neural card corner detector (MobileViT-XXS + regression head)"
    )
    p.add_argument(
        "--data-dir", type=Path, default=_data_dir,
        help="Root data directory (default: from cfg)",
    )
    p.add_argument(
        "--fast-data-dir", type=Path, default=cfg.fast_data_dir if cfg.fast_data_dir.exists() else None,
        help="Fast local cache directory for pre-resized images (default: cfg.fast_data_dir if it exists)",
    )
    p.add_argument(
        "--train-source", choices=["packopening", "clint"], default="packopening",
        help="Training data source. 'packopening' (default): 380k+ SIFT-verified frames "
             "from the packopening DB, with clint as the held-out test domain. "
             "'clint': ablation using only the 1267-frame clint dataset.",
    )
    p.add_argument(
        "--packopening-db", type=Path,
        default=_data_dir / "datasets/packopening/packopening.db",
        help="Path to packopening SQLite DB (used with --train-source packopening)",
    )
    p.add_argument(
        "--neg-sample-n", type=int, default=10_000,
        help="Number of negative frames to sample from unmatched packopening videos (default: 10000)",
    )
    p.add_argument(
        "--positives-only", action="store_true", default=True,
        help="Train on positive (card-visible) frames only; sets neg-sample-n=0 (default: True)",
    )
    p.add_argument(
        "--no-positives-only", dest="positives_only", action="store_false",
        help="Include negative frames in training (overrides --positives-only default)",
    )
    p.add_argument(
        "--max-phash-dist", type=int, default=20,
        help="Exclude packopening frames with pHash distance > this threshold (default: 20). "
             "Filters ~11%% of frames where SIFT may have matched the wrong card.",
    )
    p.add_argument(
        "--clint-corners-csv", type=Path,
        default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/corners.csv",
        help="Path to clint corners.csv (test set when --train-source packopening; "
             "train/val when --train-source clint)",
    )
    p.add_argument(
        "--clint-neg-dir", type=Path,
        default=_data_dir / "datasets/clint_cards_with_backgrounds/data/04_data/bad",
        help="Directory of clint hard-negative frames",
    )
    p.add_argument(
        "--arch", choices=["tiny", "tiny-direct", "mobilevit"], default="tiny",
        help="Model architecture. "
             "'tiny': TinyCornerCNN with soft-argmax heatmap head (CONCLUSIVELY STUCK — do not use). "
             "'tiny-direct': TinyCornerCNNDirect — same encoder, direct spatial regression head. "
             "Diagnostic run to isolate whether soft-argmax was the only failure mode. "
             "'mobilevit': MobileViT-XXS backbone (ImageNet seed), 4×4 spatial pool, direct regression. "
             "Primary path forward.",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Directory for checkpoints (default: results/corner_detector/{run_name})",
    )
    p.add_argument(
        "--seed-checkpoint", type=Path, default=None,
        help="Card-ID ArcFace checkpoint to seed backbone weights from",
    )
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--lambda-corners", type=float, default=5.0,
        help="Weight on corner regression loss (default: 5.0)",
    )
    p.add_argument(
        "--lambda-heatmap", type=float, default=10.0,
        help="Weight on heatmap auxiliary loss (default: 10.0)",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Save epoch_XXXX.pt every N epochs (0 disables, default: 5)",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (use 0 on MPS if you hit issues)",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Ignore last.pt and train from scratch",
    )
    p.add_argument(
        "--backbone-lr-scale", type=float, default=0.1,
        help="LR multiplier for pretrained backbone vs head (mobilevit only). "
             "Default 0.1 — backbone trains at lr*0.1 to avoid catastrophic forgetting. "
             "Set to 1.0 to use the same LR for all parameters.",
    )
    p.add_argument(
        "--freeze-backbone-epochs", type=int, default=0,
        help="Number of epochs to keep the backbone fully frozen while the head warms up "
             "(mobilevit only). After this many epochs, backbone unfreezes and trains at "
             "backbone_lr_scale * lr. Default 0 (no freeze phase). Recommended: 2.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
