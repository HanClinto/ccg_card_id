#!/usr/bin/env python3
"""Multi-task ArcFace training for CCG card identification.

Trains a shared MobileViT-XXS backbone with one ArcFace head per task.
Default: illustration_id + set_code trained simultaneously from scratch.

Two embedding modes (--separate-heads):
  shared (default)  one projection layer, one 128-d embedding, N ArcFace heads.
                    The embedding must serve all tasks — good for retrieval where
                    you want a single vector at query time.

  separate          one projection layer per task, independent embeddings.
                    Better task specialisation; at query time you concatenate or
                    use the appropriate head.  Useful when tasks have conflicting
                    feature requirements.

Usage:
  python 03_train_multitask.py --lr-find
  python 03_train_multitask.py                              # shared, 10 epochs
  python 03_train_multitask.py --separate-heads             # separate projections
  python 03_train_multitask.py --epochs 5                   # resumes from last.pt
  python 03_train_multitask.py --rebuild                    # ignore last.pt

Checkpoints written under:
  <output-dir>/<backbone>_multitask_<label_fields>_<mode>_<embedding_dim>d/
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
from data import load_manifest
from models import ArcFaceLoss, build_backbone

# ---------------------------------------------------------------------------
# Augmentation (same as single-task)
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """Returns (image_tensor, [label_idx_task0, label_idx_task1, ...])."""

    def __init__(
        self,
        rows: list,
        label_to_idx_per_task: list[dict[str, int]],
        label_fields: list[str],
        image_size: int,
        rotate_180_p: float,
    ):
        # Filter to rows that have a valid label for every task
        self.label_fields = label_fields
        self.label_to_idx = label_to_idx_per_task
        self.transform = _train_transform(image_size, rotate_180_p=rotate_180_p)

        self.rows = [
            r for r in rows
            if all(getattr(r, f) and getattr(r, f) in label_to_idx_per_task[i]
                   for i, f in enumerate(label_fields))
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[int]]:
        r = self.rows[idx]
        labels = [self.label_to_idx[i][getattr(r, f)]
                  for i, f in enumerate(self.label_fields)]
        img = Image.open(r.image_path).convert("RGB")
        return self.transform(img), labels


def multitask_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    n_tasks = len(batch[0][1])
    labels = [torch.tensor([b[1][t] for b in batch], dtype=torch.long) for t in range(n_tasks)]
    return imgs, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTaskEmbeddingNet(nn.Module):
    """Shared backbone with per-task projection layers.

    separate_heads=False  →  one projection shared across tasks (single embedding).
                             All embedding_dims must be equal; only the first is used.
    separate_heads=True   →  one projection per task (independent embeddings).
                             Each task may have a different embedding_dim.

    forward() always returns a list of normalised embeddings, one per task.
    In shared mode the list contains the same tensor repeated.
    """

    def __init__(
        self,
        backbone_name: str,
        embedding_dims: list[int],
        n_tasks: int,
        separate_heads: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        if len(embedding_dims) == 1:
            embedding_dims = embedding_dims * n_tasks
        if len(embedding_dims) != n_tasks:
            raise ValueError(
                f"embedding_dims must have 1 value or {n_tasks} values, got {len(embedding_dims)}"
            )
        if not separate_heads and len(set(embedding_dims)) > 1:
            raise ValueError(
                "Shared projection (--separate-heads not set) requires all embedding dims to be equal. "
                f"Got {embedding_dims}. Use --separate-heads for per-task dims."
            )

        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.separate_heads = separate_heads
        self.n_tasks = n_tasks
        self.embedding_dims = embedding_dims

        if separate_heads:
            self.projs = nn.ModuleList(
                [nn.Linear(feat_dim, dim) for dim in embedding_dims]
            )
        else:
            self.proj = nn.Linear(feat_dim, embedding_dims[0])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f = self.backbone(x)
        if self.separate_heads:
            return [F.normalize(proj(f), dim=1) for proj in self.projs]
        z = F.normalize(self.proj(f), dim=1)
        return [z] * self.n_tasks  # same embedding, evaluated by each head


# ---------------------------------------------------------------------------
# LR finder
# ---------------------------------------------------------------------------

def lr_find(
    model: nn.Module,
    criterions: list[nn.Module],
    task_weights: list[float],
    dl: DataLoader,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1e-1,
    n_steps: int = 100,
    smoothing: float = 0.9,
) -> float:
    model_state = copy.deepcopy(model.state_dict())
    crit_states = [copy.deepcopy(c.state_dict()) for c in criterions]

    params = list(model.parameters()) + [p for c in criterions for p in c.parameters()]
    optim = torch.optim.AdamW(params, lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1.0 / n_steps)
    lr = start_lr

    best_loss = float("inf")
    avg_loss = 0.0
    lrs: list[float] = []
    losses: list[float] = []

    data_iter = iter(dl)
    model.train()
    for c in criterions:
        c.train()

    print(f"LR finder: sweeping {start_lr:.0e} → {end_lr:.0e} over {n_steps} steps")
    for step in range(n_steps):
        try:
            x, ys = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, ys = next(data_iter)

        for pg in optim.param_groups:
            pg["lr"] = lr

        x = x.to(device)
        ys = [y.to(device) for y in ys]
        z_list = model(x)
        loss = sum(w * c(z, y) for w, c, z, y in zip(task_weights, criterions, z_list, ys))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optim.step()

        loss_val = loss.item()
        avg_loss = smoothing * avg_loss + (1 - smoothing) * loss_val
        smooth = avg_loss / (1 - smoothing ** (step + 1))
        lrs.append(lr)
        losses.append(smooth)

        if smooth < best_loss:
            best_loss = smooth
        elif step > 10 and smooth > 4 * best_loss:
            print(f"  diverging at lr={lr:.2e}, stopping early")
            break

        lr *= lr_mult

    model.load_state_dict(model_state)
    for c, s in zip(criterions, crit_states):
        c.load_state_dict(s)

    min_idx = losses.index(min(losses))
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

    label_fields: list[str] = args.label_fields
    task_weights: list[float] = args.task_weights
    if len(task_weights) == 1:
        task_weights = task_weights * len(label_fields)
    if len(task_weights) != len(label_fields):
        raise ValueError(
            f"--task-weights must have 1 value or {len(label_fields)} values, "
            f"got {len(task_weights)}"
        )

    rows_train = load_manifest(args.manifest, split="train")
    if not rows_train:
        raise ValueError(f"No training rows in {args.manifest}")

    # Build per-task label indices from training split
    label_to_idx_per_task: list[dict[str, int]] = []
    all_labels_per_task: list[list[str]] = []
    for field in label_fields:
        labels = sorted({getattr(r, field) for r in rows_train if getattr(r, field)})
        if not labels:
            raise ValueError(f"No non-empty values for label field {field!r}")
        label_to_idx_per_task.append({lab: i for i, lab in enumerate(labels)})
        all_labels_per_task.append(labels)
        print(f"  {field}: {len(labels)} classes")

    ds = MultiTaskDataset(
        rows_train,
        label_to_idx_per_task=label_to_idx_per_task,
        label_fields=label_fields,
        image_size=args.image_size,
        rotate_180_p=args.rotate_180_p,
    )
    print(f"Training rows after filtering: {len(ds)} / {len(rows_train)}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=multitask_collate,
    )

    device = pick_device(args.cpu)
    n_tasks = len(label_fields)

    # Resolve per-task embedding dims: broadcast single value, or use one per task
    embedding_dims: list[int] = args.embedding_dims
    if len(embedding_dims) == 1:
        embedding_dims = embedding_dims * n_tasks
    if len(embedding_dims) != n_tasks:
        raise ValueError(
            f"--embedding-dims must have 1 value or {n_tasks} values, got {len(embedding_dims)}"
        )

    model = MultiTaskEmbeddingNet(
        args.backbone,
        embedding_dims=embedding_dims,
        n_tasks=n_tasks,
        separate_heads=args.separate_heads,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterions = [
        ArcFaceLoss(
            num_classes=len(all_labels_per_task[i]),
            embedding_dim=embedding_dims[i],
            margin=args.arcface_margin,
            scale=args.arcface_scale,
        ).to(device)
        for i in range(n_tasks)
    ]

    params = list(model.parameters()) + [p for c in criterions for p in c.parameters()]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_find:
        lr_find(model, criterions, task_weights, dl, device)
        return

    mode_tag = "sep" if args.separate_heads else "shared"
    fields_tag = "+".join(label_fields)
    dims_tag = "+".join(f"{d}d" for d in embedding_dims)
    base_name = f"{args.backbone}_multitask_{fields_tag}_{mode_tag}_{dims_tag}"
    if args.run_tag:
        base_name = f"{base_name}_{args.run_tag}"
    run_dir = args.output_dir / base_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "train_history.json"

    start_epoch = 1
    history: list[dict] = []

    if args.seed_checkpoint is not None:
        seed_ckpt = torch.load(args.seed_checkpoint, map_location="cpu", weights_only=False)
        backbone_sd = {k: v for k, v in seed_ckpt["model"].items() if k.startswith("backbone.")}
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        print(f"Seeded backbone from {args.seed_checkpoint}")
        print(f"  Loaded {len(backbone_sd)} backbone keys — missing {len(missing)}, unexpected {len(unexpected)}")

    resume = args.resume_checkpoint
    auto_ckpt = run_dir / "last.pt"
    if resume is None and not args.rebuild and auto_ckpt.exists():
        resume = auto_ckpt
        print(f"auto-resume: {auto_ckpt}")

    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        for i, c in enumerate(criterions):
            key = f"criterion_{i}"
            if key in ckpt:
                c.load_state_dict(ckpt[key])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        # Reset LR so new cosine cycle starts from args.lr, not the final LR of last run
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

    task_w_str = " + ".join(f"{w}×{f}" for w, f in zip(task_weights, label_fields))
    print(f"Task loss weights: {task_w_str}")
    print(f"Embedding mode: {'separate projections' if args.separate_heads else 'shared projection'}")

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        for c in criterions:
            c.train()

        total_loss = 0.0
        task_losses = [0.0] * n_tasks
        n = 0

        for x, ys in tqdm(dl, desc=f"epoch {epoch}/{end_epoch}", unit="batch"):
            x = x.to(device)
            ys = [y.to(device) for y in ys]
            z_list = model(x)

            per_task = [criterions[i](z_list[i], ys[i]) for i in range(n_tasks)]
            loss = sum(task_weights[i] * per_task[i] for i in range(n_tasks))

            if not loss.isfinite():
                optim.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            bs = x.shape[0]
            total_loss += loss.item() * bs
            for i in range(n_tasks):
                task_losses[i] += per_task[i].item() * bs
            n += bs

        avg_loss = total_loss / max(1, n)
        avg_task = [task_losses[i] / max(1, n) for i in range(n_tasks)]
        cur_lr = optim.param_groups[0]["lr"]

        task_str = "  ".join(f"{f}={avg_task[i]:.4f}" for i, f in enumerate(label_fields))
        print(f"epoch={epoch} loss={avg_loss:.4f} lr={cur_lr:.2e}  [{task_str}]")

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "lr": cur_lr,
            **{f"loss_{f}": avg_task[i] for i, f in enumerate(label_fields)},
        })
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            **{f"criterion_{i}": criterions[i].state_dict() for i in range(n_tasks)},
            "optimizer": optim.state_dict(),
            "args": vars(args),
            "label_fields": label_fields,
            "labels_per_task": all_labels_per_task,
            "embedding_dims": embedding_dims,
            "separate_heads": args.separate_heads,
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
    p = argparse.ArgumentParser(
        description="Multi-task ArcFace metric learning for CCG card identification"
    )
    p.add_argument(
        "--manifest", type=Path,
        default=data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv",
    )
    p.add_argument(
        "--label-fields", nargs="+",
        default=["illustration_id", "set_code"],
        choices=["illustration_id", "card_id", "oracle_id", "set_code", "lang"],
        help="Manifest columns to use as ArcFace class labels, one per task (default: illustration_id set_code)",
    )
    p.add_argument(
        "--task-weights", nargs="+", type=float, default=[1.0],
        help="Loss weight per task (one value = same weight for all). Default: 1.0",
    )
    p.add_argument(
        "--separate-heads", action="store_true",
        help="Use a separate projection layer per task instead of a shared one",
    )
    p.add_argument("--output-dir", type=Path, default=data_dir / "results" / "mobilevit_xxs")
    p.add_argument("--run-tag", type=str, default="",
                   help="Optional suffix to append to run directory name (for bakeoff naming)")
    p.add_argument("--backbone", default="mobilevit_xxs",
                   choices=["mobilevit_xxs", "tinyvit", "resnet50"])
    p.add_argument(
        "--embedding-dims", nargs="+", type=int, default=[128],
        help="Embedding dim(s). One value = same for all tasks. "
             "Multiple values (--separate-heads only) = one per task. "
             "E.g. --embedding-dims 128 64  gives 128-d for task 0, 64-d for task 1.",
    )
    p.add_argument("--epochs", type=int, default=15)
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
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--seed-checkpoint", type=Path, default=None,
                   help="Load backbone weights only from a single-task checkpoint (ignores ArcFace heads and optimizer)")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore last.pt auto-resume and train from scratch")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    p.add_argument("--lr-find", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
