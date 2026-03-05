#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_build" / "mobilevit_xxs"))

from ccg_card_id.config import cfg
from retrieval import (  # type: ignore
    BackboneFeatureModel,
    embed_paths,
    load_finetuned_model,
    load_manifest_gallery,
    load_solring_queries,
)


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _discover_latest_checkpoint(results_root: Path) -> Path | None:
    cands = sorted(results_root.glob("mobilevit_xxs_arcface_*/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute MobileViT-XXS gallery/query vectors for retrieval")
    p.add_argument("--manifest", type=Path, default=cfg.data_dir / "mobilevit_xxs" / "manifest.csv")
    p.add_argument("--dataset", type=Path, default=cfg.data_dir / "datasets" / "solring")
    p.add_argument("--results-root", type=Path, default=cfg.data_dir / "results" / "mobilevit_xxs")
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--vectors-root", type=Path, default=cfg.data_dir / "vectors" / "mobilevit_xxs")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    if not args.checkpoint:
        auto_ckpt = _discover_latest_checkpoint(args.results_root)
        if auto_ckpt is not None:
            args.checkpoint = [auto_ckpt]
            print(f"Auto-discovered checkpoint: {auto_ckpt}")

    gallery_paths, _ = load_manifest_gallery(args.manifest)
    query_paths, _ = load_solring_queries(args.dataset)
    device = pick_device(force_cpu=args.cpu)

    gallery_root = args.vectors_root / f"img{args.image_size}" / f"gallery_manifest_{args.manifest.stem}"
    query_root = args.dataset / "cache" / "mobilevit_xxs" / f"img{args.image_size}"
    gallery_root.mkdir(parents=True, exist_ok=True)
    query_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Gallery (from manifest): {len(gallery_paths)}")
    print(f"Queries (from dataset={args.dataset.name}): {len(query_paths)}")
    print(f"Gallery vectors root: {gallery_root}")
    print(f"Query vectors root:   {query_root}")

    if not args.skip_base:
        base = BackboneFeatureModel("mobilevit_xxs").to(device).eval()
        embed_paths(
            base,
            gallery_paths,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            desc="base: gallery",
            cache_path=gallery_root / "mobilevit_xxs_base_320d_gallery.npz",
            rebuild_cache=args.rebuild_cache,
        )
        embed_paths(
            base,
            query_paths,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            desc="base: query",
            cache_path=query_root / "mobilevit_xxs_base_320d_query_solring.npz",
            rebuild_cache=args.rebuild_cache,
        )

    for ckpt in args.checkpoint:
        model, meta = load_finetuned_model(ckpt, device)
        emb_dim = int(meta.get("args", {}).get("embedding_dim", 128))
        epoch = int(meta.get("epoch", 0))
        tag = f"mobilevit_xxs_ft_e{epoch}_{emb_dim}d"
        embed_paths(
            model,
            gallery_paths,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            desc=f"{tag}: gallery",
            cache_path=gallery_root / f"{tag}_gallery.npz",
            rebuild_cache=args.rebuild_cache,
        )
        embed_paths(
            model,
            query_paths,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            desc=f"{tag}: query",
            cache_path=query_root / f"{tag}_query_solring.npz",
            rebuild_cache=args.rebuild_cache,
        )


if __name__ == "__main__":
    main()
