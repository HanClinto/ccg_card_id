from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .data import ManifestRow, load_manifest, read_hard_negatives
from .eval_retrieval import eval_solring_retrieval
from .mining import TripletMiner
from .models import ArcFaceLoss, EmbeddingNet


@dataclass
class TrainExample:
    anchor: str
    positive: str
    label: int


class TripletClassificationDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], card_to_idx: dict[str, int], miner: TripletMiner, image_size: int = 224):
        self.rows = rows
        self.card_to_idx = card_to_idx
        self.miner = miner
        self.t_anchor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.t_positive = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        anchor_row = self.rows[idx]
        tri = self.miner.sample_for_anchor(anchor_row)

        a_img = Image.open(tri.anchor.image_path).convert("RGB")
        if tri.positive is not None:
            p_img = Image.open(tri.positive.image_path).convert("RGB")
            x = self.t_anchor(a_img)
            p = self.t_anchor(p_img)
        else:
            x = self.t_anchor(a_img)
            p = self.t_positive(a_img)

        # Merge same-label views to stabilize low-positive classes.
        out = 0.5 * x + 0.5 * p
        y = self.card_to_idx[anchor_row.card_id]
        return out, y


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows_train = load_manifest(args.manifest, split="train")
    rows_all = load_manifest(args.manifest, split=None)
    card_ids = sorted({r.card_id for r in rows_train})
    card_to_idx = {cid: i for i, cid in enumerate(card_ids)}

    hard_negs = read_hard_negatives(args.hard_negatives_jsonl)
    miner = TripletMiner(rows_train, hard_negatives=hard_negs, seed=args.seed)
    ds = TripletClassificationDataset(rows_train, card_to_idx=card_to_idx, miner=miner, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = pick_device(force_cpu=args.cpu)
    model = EmbeddingNet(args.backbone, embedding_dim=args.embedding_dim, pretrained=not args.no_pretrained).to(device)
    criterion = ArcFaceLoss(num_classes=len(card_ids), embedding_dim=args.embedding_dim, margin=args.arcface_margin, scale=args.arcface_scale).to(device)

    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    run_dir = args.output_dir / f"{args.backbone}_arcface_{args.embedding_dim}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for x, y in tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", unit="batch"):
            x = x.to(device)
            y = y.to(device)
            z = model(x)
            loss = criterion(z, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = x.shape[0]
            epoch_loss += loss.item() * bs
            n += bs

        avg_loss = epoch_loss / max(1, n)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"epoch={epoch} loss={avg_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "card_ids": card_ids,
        }
        torch.save(ckpt, run_dir / "last.pt")

    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.eval_solring:
        metrics = eval_solring_retrieval(
            model=model,
            manifest_rows=rows_all,
            solring_aligned_dir=args.solring_dir,
            out_dir=run_dir / "eval_solring",
            device=device,
            batch_size=args.eval_batch_size,
            image_size=args.image_size,
        )
        print("solring retrieval:", metrics)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-2 ArcFace metric training scaffold")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--hard-negatives-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backbone", choices=["tinyvit", "resnet50"], default="tinyvit")
    p.add_argument("--embedding-dim", type=int, default=128, choices=[128, 256, 384, 512])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--arcface-margin", type=float, default=0.3)
    p.add_argument("--arcface-scale", type=float, default=30.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--eval-solring", action="store_true")
    p.add_argument("--solring-dir", type=Path, default=Path("~/claw/data/ccg_card_id/datasets/solring/04_data/aligned").expanduser())
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
