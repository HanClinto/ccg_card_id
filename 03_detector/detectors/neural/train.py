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
  python 03_detector/detectors/neural/train.py \\
      --seed-checkpoint $DATA/results/mobilevit_xxs/mobilevit_xxs_illustration_id_128d/last.pt

  # Ablation: train on clint only
  python 03_detector/detectors/neural/train.py --train-source clint

  # Resume from last.pt
  python 03_detector/detectors/neural/train.py --epochs 10

  # Start from scratch
  python 03_detector/detectors/neural/train.py --rebuild

Checkpoints: <results-dir>/epoch_XXXX.pt  and  last.pt
Format: {"model": state_dict, "optimizer": state_dict, "epoch": N, "val_loss": v}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg  # noqa: E402

# Local imports (relative to this file's directory)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model import NeuralCornerDetector   # noqa: E402
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
    true_corners: torch.Tensor,
    true_presence: torch.Tensor,
    lambda_corners: float = 5.0,
) -> torch.Tensor:
    """Combined presence classification + corner regression loss.

    Args:
        pred_corners:        (B, 8) sigmoid-activated corner predictions.
        pred_presence_logit: (B,)   raw presence logit (before sigmoid).
        true_corners:        (B, 8) ground-truth corners (zeros for negatives).
        true_presence:       (B,)   binary float label (1.0 = card present).
        lambda_corners:      Weight on corner regression term.

    Returns:
        Scalar loss tensor.
    """
    presence_loss = F.binary_cross_entropy_with_logits(pred_presence_logit, true_presence)
    # Corner loss only on positive examples
    pos_mask = true_presence.bool()
    if pos_mask.any():
        corner_loss = F.smooth_l1_loss(pred_corners[pos_mask], true_corners[pos_mask])
    else:
        corner_loss = torch.tensor(0.0, device=pred_corners.device)
    return presence_loss + lambda_corners * corner_loss


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
# Training loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    data_dir    = args.data_dir
    results_dir = args.results_dir

    print(f"data_dir     : {data_dir}")
    print(f"train_source : {args.train_source}")
    print(f"results_dir  : {results_dir}")

    clint_csv = args.clint_corners_csv
    clint_neg = args.clint_neg_dir

    if args.train_source == "packopening":
        db_path = args.packopening_db
        if not db_path.exists():
            raise FileNotFoundError(f"packopening DB not found: {db_path}")
        train_rows, val_rows = load_from_packopening_db(
            db_path, data_dir,
            neg_sample_n=args.neg_sample_n,
            val_frac=0.05,
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

    train_ds = CornerDataset(train_rows, data_dir, augment=True)
    val_ds   = CornerDataset(val_rows,   data_dir, augment=False)
    test_ds  = CornerDataset(test_rows,  data_dir, augment=False) if test_rows else None

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_dl = (DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
               if test_ds else None)

    device = pick_device(args.cpu)
    print(f"device      : {device}")

    model = NeuralCornerDetector(pretrained_backbone=True).to(device)

    # Seed backbone from card-ID checkpoint before optimizer setup
    if args.seed_checkpoint is not None:
        print(f"seeding backbone from {args.seed_checkpoint}")
        model.load_card_id_checkpoint(args.seed_checkpoint)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

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
        # Without this the scheduler would inherit the near-zero end-of-cycle
        # LR from the previous run and the new cycle would be useless.
        for pg in optim.param_groups:
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
        # ---- train ----
        model.train()
        total_loss = 0.0
        n_train = 0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{end_epoch}", unit="batch"):
            images   = batch["image"].to(device)
            presence = batch["card_present"].to(device)
            corners  = batch["corners"].to(device)

            pred_corners, pred_presence = model(images)
            loss = detection_loss(
                pred_corners, pred_presence, corners, presence, args.lambda_corners
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

        with torch.no_grad():
            for batch in val_dl:
                images   = batch["image"].to(device)
                presence = batch["card_present"].to(device)
                corners  = batch["corners"].to(device)

                pred_corners, pred_presence = model(images)
                loss = detection_loss(
                    pred_corners, pred_presence, corners, presence, args.lambda_corners
                )
                val_loss_sum += loss.item() * images.shape[0]
                n_val        += images.shape[0]

                all_pred_corners.append(pred_corners.cpu())
                all_true_corners.append(corners.cpu())
                all_pred_presence.append(pred_presence.cpu())
                all_true_presence.append(presence.cpu())

        val_loss = val_loss_sum / max(1, n_val)

        pc_all = torch.cat(all_pred_corners)
        tc_all = torch.cat(all_true_corners)
        pp_all = torch.cat(all_pred_presence)
        tp_all = torch.cat(all_true_presence)

        cpe      = _val_cpe(pc_all, tc_all, tp_all.bool())
        pres_acc = _val_presence_acc(pp_all, tp_all)
        cur_lr   = optim.param_groups[0]["lr"]

        # Optionally evaluate on clint test set (domain generalization)
        test_cpe = test_pres_acc = None
        if test_dl is not None:
            model.eval()
            t_pc, t_tc, t_pp, t_tp = [], [], [], []
            with torch.no_grad():
                for batch in test_dl:
                    images   = batch["image"].to(device)
                    presence = batch["card_present"].to(device)
                    corners  = batch["corners"].to(device)
                    pc, pp = model(images)
                    t_pc.append(pc.cpu()); t_tc.append(corners.cpu())
                    t_pp.append(pp.cpu()); t_tp.append(presence.cpu())
            t_pc_all = torch.cat(t_pc); t_tc_all = torch.cat(t_tc)
            t_pp_all = torch.cat(t_pp); t_tp_all = torch.cat(t_tp)
            test_cpe      = _val_cpe(t_pc_all, t_tc_all, t_tp_all.bool())
            test_pres_acc = _val_presence_acc(t_pp_all, t_tp_all)

        test_str = (f"  test_cpe={test_cpe:.4f}  test_pres_acc={test_pres_acc:.3f}"
                    if test_cpe is not None else "")
        print(
            f"epoch={epoch:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_cpe={cpe:.4f}  "
            f"val_pres_acc={pres_acc:.3f}"
            f"{test_str}  "
            f"lr={cur_lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(ckpt, last_ckpt)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(ckpt, results_dir / f"epoch_{epoch:04d}.pt")

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
        "--results-dir", type=Path,
        default=_data_dir / "results/corner_detector",
        help="Directory for checkpoints",
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
    p.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
