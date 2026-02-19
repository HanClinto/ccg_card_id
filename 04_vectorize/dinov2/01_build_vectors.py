#!/usr/bin/env python3
"""
Build DINOv2 embedding vectors for the full Scryfall front-image corpus.

Resumable + cache-aware behavior:
- Periodically writes chunk files during processing
- Re-runs resume from cached chunks by default
- --rebuild-cache forces a clean rebuild

Output:
  ~/claw/data/ccg_card_id/default_cards_dinov2_{variant}.npz
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

tqdm.monitor_interval = 0

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccg_card_id.config import cfg

DEFAULT_VARIANT = "small"
DEFAULT_BATCH_SIZE = 64
DEFAULT_CACHE_EVERY = 512  # cards

_HUB_NAMES: dict[str, str] = {
    "small": "dinov2_vits14",
    "base": "dinov2_vitb14",
    "large": "dinov2_vitl14",
    "giant": "dinov2_vitg14",
}

_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def resolve_model(arg: str) -> tuple[str, str]:
    if arg in _HUB_NAMES:
        return _HUB_NAMES[arg], arg
    for k, v in _HUB_NAMES.items():
        if arg == v:
            return v, k
    key = arg.replace("dinov2_vit", "").rstrip("14").lower()
    return arg, key or arg


def get_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(hub_name: str, device: torch.device) -> torch.nn.Module:
    print(f"Loading {hub_name} via torch.hub on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", hub_name, verbose=False)
    return model.to(device).eval()


def embed_batch(images: list[Image.Image], model: torch.nn.Module, device: torch.device) -> np.ndarray:
    tensors = torch.stack([_TRANSFORM(img) for img in images]).to(device)
    with torch.inference_mode():
        features = model(tensors)
    return features.cpu().float().numpy()


def load_cached_ids(cache_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not cache_dir.exists():
        return ids
    for part in sorted(cache_dir.glob("part_*.npz")):
        try:
            arr = np.load(part, allow_pickle=True)["card_ids"]
            ids.update(arr.tolist())
        except Exception as e:
            print(f"Warning: skipping unreadable cache part {part.name}: {e}")
    return ids


def count_final_output(output_file: Path) -> int:
    if not output_file.exists():
        return 0
    try:
        return int(len(np.load(output_file, allow_pickle=True)["card_ids"]))
    except Exception:
        return 0


def write_part(cache_dir: Path, idx: int, ids: list[str], embs: list[np.ndarray]) -> int:
    if not ids:
        return 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    part_file = cache_dir / f"part_{idx:06d}.npz"
    merged = np.vstack(embs).astype(np.float32)
    np.savez(part_file, embeddings=merged, card_ids=np.array(ids))
    return len(ids)


def merge_parts_to_output(cache_dir: Path, output_file: Path) -> int:
    part_files = sorted(cache_dir.glob("part_*.npz"))
    if not part_files:
        return 0

    all_ids: list[np.ndarray] = []
    all_embs: list[np.ndarray] = []
    for p in part_files:
        d = np.load(p, allow_pickle=True)
        all_ids.append(d["card_ids"])
        all_embs.append(d["embeddings"])

    card_ids = np.concatenate(all_ids)
    embeddings = np.vstack(all_embs).astype(np.float32)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, embeddings=embeddings, card_ids=card_ids)
    return len(card_ids)


def build_vectors(
    image_dir: Path,
    output_file: Path,
    hub_name: str,
    batch_size: int,
    cache_every: int,
    rebuild_cache: bool,
    device_override: str | None = None,
) -> None:
    cache_dir = output_file.parent / f"{output_file.stem}.cache"

    if rebuild_cache:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if output_file.exists():
            output_file.unlink()
        print("Rebuild mode: cleared existing cache/output.")

    cached_final = count_final_output(output_file)
    cached_ids = load_cached_ids(cache_dir)
    cached_total = max(cached_final, len(cached_ids))

    if cached_total > 0:
        print(f"Loaded {cached_total:,} cached items.")
    else:
        print("Loaded 0 cached items.")
    print("Tip: rerun with --rebuild-cache to ignore cache and rebuild from scratch.\n")

    exts = {".png", ".jpg", ".jpeg"}
    image_files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    if not image_files:
        sys.exit(f"No images found in {image_dir}")
    print(f"Found {len(image_files):,} images in {image_dir}")

    # If final output already covers everything, we're done.
    if output_file.exists() and cached_final >= len(image_files) and not rebuild_cache:
        print("Final output already complete; nothing to do.")
        return

    # Build fast skip set from cache parts (or final file if no parts).
    if not cached_ids and output_file.exists() and not rebuild_cache:
        try:
            cached_ids = set(np.load(output_file, allow_pickle=True)["card_ids"].tolist())
        except Exception:
            cached_ids = set()

    device = get_device(device_override)
    model = load_model(hub_name, device)

    part_idx = 0
    existing_parts = sorted(cache_dir.glob("part_*.npz")) if cache_dir.exists() else []
    if existing_parts:
        part_idx = int(existing_parts[-1].stem.split("_")[-1]) + 1

    pending_ids: list[str] = []
    pending_embs: list[np.ndarray] = []

    processed_now = 0
    skipped_cached = 0

    with tqdm(total=len(image_files), desc="Embedding", unit="img") as pbar:
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i : i + batch_size]

            batch_images: list[Image.Image] = []
            batch_ids: list[str] = []

            for p in batch_paths:
                cid = p.stem
                if cid in cached_ids:
                    skipped_cached += 1
                    continue
                try:
                    batch_images.append(Image.open(p).convert("RGB"))
                    batch_ids.append(cid)
                except Exception as e:
                    print(f"\nWarning: could not load {p.name}: {e}")

            if batch_images:
                embs = embed_batch(batch_images, model, device)
                pending_ids.extend(batch_ids)
                pending_embs.append(embs)
                processed_now += len(batch_ids)

            if len(pending_ids) >= cache_every:
                wrote = write_part(cache_dir, part_idx, pending_ids, pending_embs)
                cached_ids.update(pending_ids)
                part_idx += 1
                pending_ids = []
                pending_embs = []
                pbar.set_postfix_str(f"cached+{wrote} now={processed_now:,} skipped={skipped_cached:,}")

            pbar.update(len(batch_paths))

    if pending_ids:
        wrote = write_part(cache_dir, part_idx, pending_ids, pending_embs)
        cached_ids.update(pending_ids)
        print(f"Wrote final cache part: {wrote} items")

    total = merge_parts_to_output(cache_dir, output_file)
    print(f"\nSaved {total:,} embeddings → {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DINOv2 embeddings (resumable, cache-aware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_VARIANT, metavar="VARIANT")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cache-every", type=int, default=DEFAULT_CACHE_EVERY,
                        help="Write cache part every N newly embedded cards.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Ignore previous cache/output and rebuild from scratch.")
    parser.add_argument("--images", type=Path, default=cfg.scryfall_images_dir / "front")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None, metavar="DEVICE")
    args = parser.parse_args()

    hub_name, variant = resolve_model(args.model)
    output = args.output or cfg.dinov2_vectors_file(variant)

    print(f"Model:   {hub_name} ({variant})")
    print(f"Images:  {args.images}")
    print(f"Output:  {output}")
    print(f"Batch:   {args.batch_size}")
    print(f"Cache:   every {args.cache_every} cards")
    print()

    build_vectors(
        image_dir=args.images,
        output_file=output,
        hub_name=hub_name,
        batch_size=args.batch_size,
        cache_every=args.cache_every,
        rebuild_cache=args.rebuild_cache,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
