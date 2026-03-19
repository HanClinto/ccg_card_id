#!/usr/bin/env python3
"""Update gallery embedding NPZs for the web scanner.

Computes (or refreshes) gallery vectors for all configured identifiers:
  - pHash at 8×8, 16×16, 32×32
  - ArcFace for the latest checkpoint (or a specified one)

Uses incremental cache updates by default: if the manifest path list changes,
existing gallery caches reuse vectors for unchanged paths, compute vectors only
for new paths, and drop removed paths so the output stays aligned to the current
manifest order. Use --rebuild to force a full recompute.

Run after:
  - 01_data_sources/scryfall/03_sync_images.py   (new card images available)
  - 04_vectorize/mobilevit_xxs/01_build_manifest.py  (manifest updated)
  - A new training checkpoint has been saved

Usage (run from project root):
    python 05_lookup_db/02_update_gallery_vectors.py
    python 05_lookup_db/02_update_gallery_vectors.py --checkpoint path/to/epoch_0090.pt
    python 05_lookup_db/02_update_gallery_vectors.py --skip-arcface
    python 05_lookup_db/02_update_gallery_vectors.py --rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_vectorize" / "mobilevit_xxs"))

from ccg_card_id.config import cfg  # noqa: E402
from retrieval import (  # noqa: E402  type: ignore
    compute_phash_embeddings,
    embed_paths,
    load_finetuned_model,
    load_manifest_gallery,
)


PHASH_SIZES = [8, 16, 32]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _discover_latest_checkpoint(results_root: Path) -> Path | None:
    """Return the most recently modified last.pt under results_root."""
    cands = sorted(
        results_root.glob("mobilevit_xxs_*/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _checkpoint_variant(ckpt_path: Path) -> str:
    """Build the gallery NPZ stem from a checkpoint, matching the web scanner's naming."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cargs = ckpt.get("args", {})
        epoch = int(ckpt.get("epoch", 0))
        emb_dim = int(
            cargs.get("embedding_dim")
            or (cargs.get("embedding_dims") or [128])[0]
        )
        raw_field = cargs.get("label_field")
        if raw_field is None:
            fields = cargs.get("label_fields") or ["card_id"]
            raw_field = "+".join(str(f) for f in fields)
        return f"mobilevit_xxs_ft_{raw_field}_e{epoch}_{emb_dim}d"
    except Exception as exc:
        print(f"  Warning: could not read checkpoint metadata ({exc}); using filename as fallback")
        return ckpt_path.stem


def main() -> None:
    p = argparse.ArgumentParser(
        description="Update web-scanner gallery NPZs with incremental reuse; use --rebuild for a full recompute"
    )
    p.add_argument(
        "--manifest", type=Path,
        default=cfg.data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv",
        help="Gallery manifest CSV; cache order follows this manifest (default: artwork_id_manifest.csv)",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="ArcFace checkpoint to use. Default: auto-discover latest last.pt",
    )
    p.add_argument(
        "--results-root", type=Path,
        default=cfg.data_dir / "results" / "mobilevit_xxs",
        help="Root directory for training runs (used for auto-discovery)",
    )
    p.add_argument(
        "--phash-gallery-dir", type=Path,
        default=None,
        help="Override pHash gallery output dir for incremental gallery caches",
    )
    p.add_argument(
        "--arcface-gallery-dir", type=Path,
        default=None,
        help="Override ArcFace gallery output dir for incremental gallery caches",
    )
    p.add_argument("--skip-phash",   action="store_true", help="Skip pHash gallery update")
    p.add_argument("--skip-arcface", action="store_true", help="Skip ArcFace gallery update")
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force full recompute instead of incrementally reusing unchanged cache entries",
    )
    p.add_argument("--batch-size",   type=int, default=64)
    args = p.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    # Derived gallery dirs (match GallerySearchManager defaults in search.py)
    phash_dir = args.phash_gallery_dir or (
        cfg.data_dir / "vectors" / "phash" / f"gallery_manifest_{args.manifest.stem}"
    )
    arcface_dir = args.arcface_gallery_dir or (
        cfg.data_dir / "vectors" / "mobilevit_xxs" / "img224"
        / f"gallery_manifest_{args.manifest.stem}"
    )

    print(f"Manifest      : {args.manifest}")
    print(f"pHash dir     : {phash_dir}")
    print(f"ArcFace dir   : {arcface_dir}")

    # Load gallery paths (existence-checked inside load_manifest_gallery)
    print("\nLoading gallery from manifest …")
    gallery_paths, _, _ = load_manifest_gallery(args.manifest)
    print(f"  {len(gallery_paths):,} gallery images")

    # ------------------------------------------------------------------ pHash
    if not args.skip_phash:
        print("\n--- pHash galleries ---")
        phash_dir.mkdir(parents=True, exist_ok=True)
        for hs in PHASH_SIZES:
            bits = hs * hs
            cache_path = phash_dir / f"phash_{hs}x{hs}_{bits}bit_gallery.npz"
            compute_phash_embeddings(
                gallery_paths,
                hash_size=hs,
                desc=f"phash_{hs}x{hs}: gallery",
                cache_path=cache_path,
                rebuild_cache=args.rebuild,
            )

    # ---------------------------------------------------------------- ArcFace
    if not args.skip_arcface:
        print("\n--- ArcFace gallery ---")
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = _discover_latest_checkpoint(args.results_root)
        if ckpt is None:
            print("  No checkpoint found — skipping ArcFace gallery update.")
            print(f"  (Searched: {args.results_root})")
        else:
            print(f"  Checkpoint: {ckpt}")
            variant = _checkpoint_variant(ckpt)
            print(f"  Variant   : {variant}")
            arcface_dir.mkdir(parents=True, exist_ok=True)
            cache_path = arcface_dir / f"{variant}_gallery.npz"

            device = pick_device()
            print(f"  Device    : {device}")
            model, _ = load_finetuned_model(ckpt, device)
            embed_paths(
                model,
                gallery_paths,
                device=device,
                batch_size=args.batch_size,
                image_size=224,
                desc=f"{variant}: gallery",
                cache_path=cache_path,
                rebuild_cache=args.rebuild,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
