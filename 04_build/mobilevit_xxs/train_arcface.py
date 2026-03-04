from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from data import ManifestRow, load_manifest, read_hard_negatives
from eval_retrieval import eval_solring_retrieval
from mining import TripletMiner
from models import ArcFaceLoss, EmbeddingNet


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
        self.t_train_aug = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
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
            p = self.t_train_aug(p_img)
        else:
            p = self.t_train_aug(a_img)

        n_img = Image.open(tri.negative.image_path).convert("RGB")

        a = self.t_anchor(a_img)
        n = self.t_train_aug(n_img)
        y = self.card_to_idx[anchor_row.card_id]
        return a, p, n, y


class PrebuiltTripletDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], card_to_idx: dict[str, int], image_size: int = 224):
        self.rows = rows
        self.card_to_idx = card_to_idx
        self.t_anchor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.t_train_aug = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        a_img = Image.open(r["anchor_path"]).convert("RGB")
        p_img = Image.open(r["positive_path"]).convert("RGB")
        n_img = Image.open(r["negative_path"]).convert("RGB")

        a = self.t_anchor(a_img)
        p = self.t_train_aug(p_img)
        n = self.t_train_aug(n_img)
        y = self.card_to_idx[r["anchor_id"]]
        return a, p, n, y


def _load_triplets_csv(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({k: str(v) for k, v in row.items()})
    return out


def _parse_task_weights(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


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

    train_id_set = {r.card_id for r in rows_train}

    sampler = None
    if args.triplets_csv is not None:
        triplets = _load_triplets_csv(args.triplets_csv)
        triplets = [t for t in triplets if t.get("anchor_id") in train_id_set]
        ds = PrebuiltTripletDataset(triplets, card_to_idx=card_to_idx, image_size=args.image_size)

        if args.task_weights:
            tw = _parse_task_weights(args.task_weights)
            weights = [float(tw.get(t.get("task", ""), 1.0)) for t in triplets]
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            print(f"task-balanced sampling enabled: {tw}")
    else:
        hard_negs = read_hard_negatives(args.hard_negatives_jsonl)
        miner = TripletMiner(rows_train, hard_negatives=hard_negs, seed=args.seed)
        ds = TripletClassificationDataset(rows_train, card_to_idx=card_to_idx, miner=miner, image_size=args.image_size)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
    )

    device = pick_device(force_cpu=args.cpu)
    model = EmbeddingNet(args.backbone, embedding_dim=args.embedding_dim, pretrained=not args.no_pretrained).to(device)
    criterion = ArcFaceLoss(num_classes=len(card_ids), embedding_dim=args.embedding_dim, margin=args.arcface_margin, scale=args.arcface_scale).to(device)
    triplet_criterion = torch.nn.TripletMarginLoss(margin=args.triplet_margin, p=2)

    params = list(model.parameters()) + list(criterion.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    run_dir = args.output_dir / f"{args.backbone}_arcface_{args.embedding_dim}"
    run_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    history: list[dict] = []

    resume_checkpoint = args.resume_checkpoint
    auto_ckpt = run_dir / "last.pt"
    if resume_checkpoint is None and (not args.rebuild_cache) and auto_ckpt.exists():
        resume_checkpoint = auto_ckpt
        print(f"auto-resume: found {auto_ckpt}")
        print("hint: pass --rebuild-cache to ignore prior checkpoint and train from scratch")

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        print(f"loaded checkpoint: {resume_checkpoint}")
        model.load_state_dict(ckpt["model"])
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])

        ckpt_card_ids = ckpt.get("card_ids")
        if ckpt_card_ids is not None and list(ckpt_card_ids) != card_ids:
            raise ValueError("resume checkpoint card_ids mismatch with current manifest train split")

        prior_epoch = int(ckpt.get("epoch", 0))
        start_epoch = prior_epoch + 1

        history_path = run_dir / "train_history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []

        print(f"resuming from {resume_checkpoint} at epoch={prior_epoch}; running {args.epochs} additional epoch(s)")

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for a, p, n_img, y in tqdm(dl, desc=f"epoch {epoch}/{end_epoch}", unit="batch"):
            a = a.to(device)
            p = p.to(device)
            n_img = n_img.to(device)
            y = y.to(device)

            za = model(a)
            zp = model(p)
            zn = model(n_img)

            loss_arc = criterion(za, y)
            loss_tri = triplet_criterion(za, zp, zn)
            loss = loss_arc + (args.triplet_weight * loss_tri)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bs = a.shape[0]
            epoch_loss += loss.item() * bs
            n += bs

        avg_loss = epoch_loss / max(1, n)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"epoch={epoch} loss={avg_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optim.state_dict(),
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
    p.add_argument("--triplets-csv", type=Path, default=None)
    p.add_argument("--task-weights", type=str, default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backbone", choices=["mobilevit_xxs", "tinyvit", "resnet50"], default="mobilevit_xxs")
    p.add_argument("--embedding-dim", type=int, default=128, choices=[128, 256, 384, 512])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--arcface-margin", type=float, default=0.4)
    p.add_argument("--arcface-scale", type=float, default=32.0)
    p.add_argument("--triplet-margin", type=float, default=0.2)
    p.add_argument("--triplet-weight", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-checkpoint", type=Path, default=None, help="Resume from a prior last.pt and run --epochs additional epochs")
    p.add_argument("--rebuild-cache", action="store_true", help="Ignore last.pt auto-resume and start from scratch")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--eval-solring", action="store_true")
    p.add_argument("--solring-dir", type=Path, default=Path("~/claw/data/ccg_card_id/datasets/solring/04_data/aligned").expanduser())
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
