#!/usr/bin/env python3
"""Train MobileViT-XXS with ArcFace loss for CCG card identification.

Stages:
  artwork_id   --label-field illustration_id   (default)
               Groups all cards sharing the same artwork.

  printing_id  --label-field card_id
               Distinguishes different printings of the same card.

Usage:
  python 02_train.py --lr-find                    # LR range test (~2 min, then exit)
  python 02_train.py                              # artwork_id, 10 epochs
  python 02_train.py --label-field card_id        # printing_id stage
  python 02_train.py --epochs 5                   # run 5 more epochs (resumes)
  python 02_train.py --rebuild                    # ignore last.pt, train from scratch

Checkpoints and training history are written under:
  <output-dir>/<backbone>_<label_field>_<embedding_dim>d/
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
from data import load_manifest
from models import ArcFaceLoss, EmbeddingNet

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _train_transform(image_size: int, rotate_180_p: float = 0.0) -> transforms.Compose:
    """Light augmentation profile for real-camera aligned card crops.

    Policy:
    - No random crop/affine/perspective (already present in captured data)
    - Color jitter only
    - Optional exact 180° rotation (p configurable)
    """
    aug = [
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    if rotate_180_p > 0:
        aug.append(transforms.RandomApply([transforms.RandomRotation((180, 180))], p=rotate_180_p))
    aug.extend([
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    return transforms.Compose(aug)


class ImageLabelDataset(Dataset):
    def __init__(
        self,
        rows: list,
        label_to_idx: dict[str, int],
        label_field: str,
        image_size: int,
        rotate_180_p: float,
    ):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.label_field = label_field
        self.transform = _train_transform(image_size, rotate_180_p=rotate_180_p)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        r = self.rows[idx]
        label_val = getattr(r, self.label_field)
        y = self.label_to_idx[label_val]
        img = Image.open(r.image_path).convert("RGB")
        return self.transform(img), y


# ---------------------------------------------------------------------------
# LR range test
# ---------------------------------------------------------------------------

def lr_find(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1e-1,
    n_steps: int = 100,
    smoothing: float = 0.9,
) -> float:
    """LR range test (fastai-style). Returns suggested LR."""
    model_state = copy.deepcopy(model.state_dict())
    criterion_state = copy.deepcopy(criterion.state_dict())

    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(params, lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1.0 / n_steps)
    lr = start_lr

    best_loss = float("inf")
    avg_loss = 0.0
    lrs: list[float] = []
    losses: list[float] = []

    data_iter = iter(dl)
    model.train()
    criterion.train()

    print(f"LR finder: sweeping {start_lr:.0e} → {end_lr:.0e} over {n_steps} steps")
    for step in range(n_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, y = next(data_iter)

        for pg in optim.param_groups:
            pg["lr"] = lr

        x, y = x.to(device), y.to(device)
        z = model(x)
        loss = criterion(z, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optim.step()

        loss_val = loss.item()
        avg_loss = smoothing * avg_loss + (1 - smoothing) * loss_val
        smooth = avg_loss / (1 - smoothing ** (step + 1))  # bias correction
        lrs.append(lr)
        losses.append(smooth)

        if smooth < best_loss:
            best_loss = smooth
        elif step > 10 and smooth > 4 * best_loss:
            print(f"  diverging at lr={lr:.2e}, stopping early")
            break

        lr *= lr_mult

    # Restore weights
    model.load_state_dict(model_state)
    criterion.load_state_dict(criterion_state)

    min_idx = losses.index(min(losses))
    # Suggest LR slightly before minimum (steepest descent region)
    suggested_idx = max(0, min_idx - max(1, len(lrs) // 10))
    suggested_lr = lrs[suggested_idx]

    print(f"\n{'LR':>10}  {'Loss':>10}")
    step_size = max(1, len(lrs) // 20)
    for i in range(0, len(lrs), step_size):
        marker = " <-- suggested" if i == suggested_idx else ""
        print(f"  {lrs[i]:8.2e}  {losses[i]:10.4f}{marker}")
    print(f"\nSuggested LR: {suggested_lr:.2e}  (min loss {min(losses):.4f} at {lrs[min_idx]:.2e})")
    return suggested_lr


# ---------------------------------------------------------------------------
# Device
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows_train = load_manifest(args.manifest, split="train")
    if not rows_train:
        raise ValueError(f"No training rows found in {args.manifest}")

    # Build label index from training split
    labels = sorted({
        getattr(r, args.label_field)
        for r in rows_train
        if getattr(r, args.label_field)
    })
    if not labels:
        raise ValueError(f"No non-empty values for --label-field={args.label_field!r}")
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)
    print(f"label_field={args.label_field}  n_classes={n_classes}  n_train={len(rows_train)}")

    ds = ImageLabelDataset(
        rows_train,
        label_to_idx=label_to_idx,
        label_field=args.label_field,
        image_size=args.image_size,
        rotate_180_p=args.rotate_180_p,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    device = pick_device(args.cpu)
    model = EmbeddingNet(
        args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = ArcFaceLoss(
        num_classes=n_classes,
        embedding_dim=args.embedding_dim,
        margin=args.arcface_margin,
        scale=args.arcface_scale,
    ).to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_find:
        lr_find(model, criterion, dl, device)
        return

    run_dir = args.output_dir / f"{args.backbone}_{args.label_field}_{args.embedding_dim}d"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "train_history.json"

    start_epoch = 1
    history: list[dict] = []

    # Auto-resume from last checkpoint unless --rebuild
    resume = args.resume_checkpoint
    auto_ckpt = run_dir / "last.pt"
    if resume is None and not args.rebuild and auto_ckpt.exists():
        resume = auto_ckpt
        print(f"auto-resume: {auto_ckpt}")

    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        # Reset LR to args.lr so the new cosine cycle starts from the right point.
        # Without this, the scheduler would inherit the final (very low) LR from
        # the previous run and schedule from there to near-zero — useless.
        for pg in optim.param_groups:
            pg["lr"] = args.lr
        prior_epoch = int(ckpt.get("epoch", 0))
        start_epoch = prior_epoch + 1
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        print(f"resuming from epoch {prior_epoch}")

    end_epoch = start_epoch + args.epochs - 1
    total_steps = (end_epoch - start_epoch + 1) * len(dl)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=args.lr * 0.01,
        )
        print(f"cosine LR schedule: {args.lr:.2e} → {args.lr * 0.01:.2e} over {total_steps} steps")

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        criterion.train()
        total_loss = 0.0
        n = 0
        for x, y in tqdm(dl, desc=f"epoch {epoch}/{end_epoch}", unit="batch"):
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = criterion(z, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optim.step()
            if scheduler is not None:
                scheduler.step()
            bs = x.shape[0]
            total_loss += loss.item() * bs
            n += bs

        avg_loss = total_loss / max(1, n)
        cur_lr = optim.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": avg_loss, "lr": cur_lr})
        print(f"epoch={epoch} loss={avg_loss:.4f} lr={cur_lr:.2e}")

        # Write history after every epoch so partial runs are preserved
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optim.state_dict(),
            "args": vars(args),
            "labels": labels,
            "label_field": args.label_field,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(ckpt, run_dir / f"epoch_{epoch:04d}.pt")

    print(f"Training complete. Run dir: {run_dir}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    data_dir = cfg.data_dir
    p = argparse.ArgumentParser(description="ArcFace metric learning for CCG card identification")
    p.add_argument(
        "--manifest",
        type=Path,
        default=data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv",
        help="Training manifest CSV (default: artwork_id_manifest.csv)",
    )
    p.add_argument(
        "--label-field",
        default="illustration_id",
        choices=["illustration_id", "card_id", "oracle_id"],
        help="Manifest column to use as class label (default: illustration_id for artwork_id stage)",
    )
    p.add_argument("--output-dir", type=Path, default=data_dir / "results" / "mobilevit_xxs")
    p.add_argument("--backbone", default="mobilevit_xxs",
                   choices=["mobilevit_xxs", "tinyvit", "resnet50"])
    p.add_argument("--embedding-dim", type=int, default=128, choices=[128, 256, 384, 512])
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of epochs to train (resumes add on top of prior epochs)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--arcface-margin", type=float, default=0.3)
    p.add_argument("--arcface-scale", type=float, default=32.0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--rotate-180-p", type=float, default=0.0,
                   help="Probability of applying exact 180° rotation during training (default: 0.0)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (default: 4; use 0 on MPS if you hit issues)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-checkpoint", type=Path, default=None,
                   help="Explicit checkpoint to resume from")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore last.pt auto-resume and train from scratch")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save epoch_XXXX.pt every N epochs (0 disables)")
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"],
                   help="LR scheduler (default: cosine)")
    p.add_argument("--lr-find", action="store_true",
                   help="Run LR range test and exit (does not save checkpoints)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
