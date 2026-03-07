#!/usr/bin/env python3
"""Train MobileViT-XXS with ArcFace loss for CCG card identification.

Stages:
  artwork_id   --label-field illustration_id   (default)
               Groups all cards sharing the same artwork.

  printing_id  --label-field card_id
               Distinguishes different printings of the same card.

Usage:
  python 02_train.py                              # artwork_id, 10 epochs
  python 02_train.py --label-field card_id        # printing_id stage
  python 02_train.py --epochs 5                   # run 5 more epochs (resumes)
  python 02_train.py --rebuild                    # ignore last.pt, train from scratch

Checkpoints and training history are written under:
  <output-dir>/<backbone>_<label_field>_<embedding_dim>d/
"""
from __future__ import annotations

import argparse
import json
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


def _train_transform(image_size: int) -> transforms.Compose:
    """Heavy augmentation to simulate real-world scan/photo conditions."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.05),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


class ImageLabelDataset(Dataset):
    def __init__(
        self,
        rows: list,
        label_to_idx: dict[str, int],
        label_field: str,
        image_size: int,
    ):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.label_field = label_field
        self.transform = _train_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        r = self.rows[idx]
        label_val = getattr(r, self.label_field)
        y = self.label_to_idx[label_val]
        img = Image.open(r.image_path).convert("RGB")
        return self.transform(img), y


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
        prior_epoch = int(ckpt.get("epoch", 0))
        start_epoch = prior_epoch + 1
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        print(f"resuming from epoch {prior_epoch}")

    end_epoch = start_epoch + args.epochs - 1
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
            bs = x.shape[0]
            total_loss += loss.item() * bs
            n += bs

        avg_loss = total_loss / max(1, n)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"epoch={epoch} loss={avg_loss:.4f}")

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
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-checkpoint", type=Path, default=None,
                   help="Explicit checkpoint to resume from")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore last.pt auto-resume and train from scratch")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save epoch_XXXX.pt every N epochs (0 disables)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
