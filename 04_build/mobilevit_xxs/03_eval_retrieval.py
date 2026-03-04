from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import load_manifest
from eval_retrieval import eval_solring_retrieval
from models import EmbeddingNet
from train_arcface import pick_device


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MobileViT-XXS fine-tuned embedding model on Sol Ring retrieval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--solring-dir", type=Path, default=Path("~/claw/data/ccg_card_id/datasets/solring/04_data/aligned").expanduser())
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true", help="Recompute even if retrieval_summary.json exists")
    args = p.parse_args()

    summary_path = args.out_dir / "retrieval_summary.json"
    if summary_path.exists() and not args.rebuild_cache:
        print(summary_path.read_text(encoding="utf-8"))
        print("(cached) use --rebuild-cache to recompute")
        return

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    model = EmbeddingNet(cargs["backbone"], embedding_dim=int(cargs["embedding_dim"]), pretrained=False)
    model.load_state_dict(ckpt["model"])

    device = pick_device(force_cpu=args.cpu)
    model = model.to(device)
    rows = load_manifest(args.manifest)
    metrics = eval_solring_retrieval(
        model=model,
        manifest_rows=rows,
        solring_aligned_dir=args.solring_dir,
        out_dir=args.out_dir,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    print(metrics)


if __name__ == "__main__":
    main()
