from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.data import load_manifest
from phase2.eval_retrieval import eval_solring_retrieval
from phase2.models import EmbeddingNet
from phase2.train_arcface import pick_device


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate phase2 embedding model on solring retrieval")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--solring-dir", type=Path, default=Path("~/claw/data/ccg_card_id/datasets/solring/04_data/aligned").expanduser())
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

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
