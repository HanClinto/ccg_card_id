from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.data import load_manifest
from phase2.eval_retrieval import eval_solring_retrieval
from phase2.models import EmbeddingNet, build_backbone
from phase2.train_arcface import pick_device


class BackboneFeatureModel(torch.nn.Module):
    """Normalized backbone features (no train-time projection head)."""

    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone, _ = build_backbone(backbone_name, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return torch.nn.functional.normalize(f, dim=1)


def eval_base(
    *,
    backbone: str,
    rows,
    solring_dir: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> dict:
    model = BackboneFeatureModel(backbone, pretrained=True).to(device)
    metrics = eval_solring_retrieval(
        model=model,
        manifest_rows=rows,
        solring_aligned_dir=solring_dir,
        out_dir=out_dir,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    metrics["model"] = f"base:{backbone}"
    return metrics


def eval_checkpoint(
    *,
    checkpoint: Path,
    rows,
    solring_dir: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cargs = ckpt["args"]
    model = EmbeddingNet(cargs["backbone"], embedding_dim=int(cargs["embedding_dim"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    metrics = eval_solring_retrieval(
        model=model,
        manifest_rows=rows,
        solring_aligned_dir=solring_dir,
        out_dir=out_dir,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    metrics["model"] = f"ft:{checkpoint.parent.name}"
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Compare base vs fine-tuned retrieval on Sol Ring set")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--solring-dir", type=Path, default=Path("~/claw/data/ccg_card_id/datasets/solring/04_data/aligned").expanduser())
    p.add_argument("--base-backbone", action="append", default=["mobilevit_xxs"], help="Backbone baseline to evaluate (repeatable)")
    p.add_argument("--checkpoint", type=Path, action="append", default=[], help="Fine-tuned checkpoint(s) to compare")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = pick_device(force_cpu=args.cpu)
    rows = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []

    for bb in args.base_backbone:
        out = args.out_dir / f"base_{bb}"
        all_metrics.append(
            eval_base(
                backbone=bb,
                rows=rows,
                solring_dir=args.solring_dir,
                out_dir=out,
                device=device,
                batch_size=args.batch_size,
                image_size=args.image_size,
            )
        )

    for ckpt in args.checkpoint:
        out = args.out_dir / ckpt.parent.name
        all_metrics.append(
            eval_checkpoint(
                checkpoint=ckpt,
                rows=rows,
                solring_dir=args.solring_dir,
                out_dir=out,
                device=device,
                batch_size=args.batch_size,
                image_size=args.image_size,
            )
        )

    with (args.out_dir / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    fields = ["model", "top1", "top3", "top10", "n_queries", "evaluated_at", "checkpoint"]
    with (args.out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_metrics:
            row = {k: m.get(k, "") for k in fields}
            w.writerow(row)

    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
