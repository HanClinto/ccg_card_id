#!/usr/bin/env python3
"""Download Pokemon TCG card images from URLs in all_cards.json.

Images are written to:
    {cfg.pokemontcg_images_dir}/large/{a}/{b}/{card_id}.png

where {a} and {b} are the first two characters of the card ID, giving a
2-level directory hierarchy that mirrors the Scryfall image layout.

Existing files are skipped unless --rebuild is passed.

Usage (run from project root):
    python 01_data_sources/pokemontcgio/03_sync_images.py
    python 01_data_sources/pokemontcgio/03_sync_images.py --workers 8
    python 01_data_sources/pokemontcgio/03_sync_images.py --set-ids base1 base2
    python 01_data_sources/pokemontcgio/03_sync_images.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg


def _dst_path(images_dir: Path, card_id: str, ext: str) -> Path:
    a = card_id[0] if card_id else "x"
    b = card_id[1] if len(card_id) > 1 else "x"
    return images_dir / "large" / a / b / f"{card_id}.{ext}"


def _download_one(url: str, dst: Path, rebuild: bool) -> tuple[str, bool]:
    """Download url → dst. Returns (url, success). Skips if dst exists and not rebuild."""
    if dst.exists() and not rebuild:
        return url, True
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(resp.content)
        return url, True
    except Exception as e:
        return url, False


def sync_images(
    src_json: Path,
    images_dir: Path,
    set_filter: set[str] | None = None,
    workers: int = 8,
    rebuild: bool = False,
) -> None:
    print(f"Reading {src_json} ...")
    with src_json.open(encoding="utf-8") as f:
        cards = json.load(f)
    print(f"  {len(cards):,} cards loaded")

    tasks: list[tuple[str, Path]] = []
    skipped_set = 0
    no_image = 0

    for card in cards:
        set_id = card.get("set", {}).get("id", "")
        if set_filter and set_id not in set_filter:
            skipped_set += 1
            continue
        url = (card.get("images") or {}).get("large") or (card.get("images") or {}).get("small")
        if not url:
            no_image += 1
            continue
        ext = url.rsplit(".", 1)[-1].split("?")[0] or "png"
        dst = _dst_path(images_dir, card["id"], ext)
        tasks.append((url, dst))

    already = sum(1 for _, dst in tasks if dst.exists() and not rebuild)
    to_download = [(url, dst) for url, dst in tasks if not dst.exists() or rebuild]

    print(f"Cards selected : {len(tasks):,}  (skipped by set filter: {skipped_set:,}, no image: {no_image:,})")
    print(f"Already cached : {already:,}")
    print(f"To download    : {len(to_download):,}")

    if not to_download:
        print("Nothing to download.")
        return

    ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, url, dst, rebuild): (url, dst) for url, dst in to_download}
        with tqdm(as_completed(futs), total=len(futs), unit="img", desc="Downloading") as pbar:
            for fut in pbar:
                _, success = fut.result()
                if success:
                    ok += 1
                else:
                    err += 1
                    url, _ = futs[fut]
                    pbar.write(f"FAIL: {url}")

    print(f"\nDownloaded: {ok:,}  Errors: {err:,}")


def main() -> None:
    p = argparse.ArgumentParser(description="Sync Pokemon TCG card images")
    p.add_argument("--src", type=Path, default=cfg.pokemontcg_all_cards)
    p.add_argument("--images-dir", type=Path, default=cfg.pokemontcg_images_dir)
    p.add_argument("--set-ids", nargs="+", metavar="SET_ID",
                   help="Only download images for these set IDs (e.g. base1 base2)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rebuild", action="store_true",
                   help="Re-download even if file already exists")
    args = p.parse_args()

    if not args.src.exists():
        print(f"ERROR: {args.src} not found", file=sys.stderr)
        print("Run: python 01_data_sources/pokemontcgio/01_sync_data.py", file=sys.stderr)
        sys.exit(1)

    set_filter = set(args.set_ids) if args.set_ids else None
    if set_filter:
        print(f"Set filter: {sorted(set_filter)}")

    sync_images(args.src, args.images_dir, set_filter=set_filter,
                workers=args.workers, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
