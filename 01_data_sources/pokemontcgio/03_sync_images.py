#!/usr/bin/env python3
"""Download Pokemon TCG card images from URLs in all_cards.json.

Images are written to:
    {cfg.pokemontcg_images_dir}/large/{a}/{b}/{card_id}.png

where {a} and {b} are the first two characters of the card ID, giving a
2-level directory hierarchy that mirrors the Scryfall image layout.

Three sync modes
----------------
default          Skip if local file already exists (fast, no network for cached images).
--rebuild        HEAD request per existing image; redownload only if remote
                 Last-Modified is newer than the local file mtime. Downloads
                 missing images unconditionally. Use for weekly updates.
--force-rebuild  Redownload every image unconditionally regardless of local state.

Usage (run from project root):
    python 01_data_sources/pokemontcgio/03_sync_images.py
    python 01_data_sources/pokemontcgio/03_sync_images.py --rebuild
    python 01_data_sources/pokemontcgio/03_sync_images.py --force-rebuild
    python 01_data_sources/pokemontcgio/03_sync_images.py --workers 8
    python 01_data_sources/pokemontcgio/03_sync_images.py --set-ids base1 base2
"""
from __future__ import annotations

import argparse
import email.utils
import json
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


def _remote_newer(url: str, dst: Path) -> bool:
    """HEAD request: return True if remote Last-Modified is newer than local mtime."""
    try:
        resp = requests.head(url, timeout=10)
        resp.raise_for_status()
        last_modified = resp.headers.get("Last-Modified")
        if not last_modified:
            return True  # can't tell — assume update needed
        remote_mtime = email.utils.parsedate_to_datetime(last_modified).timestamp()
        return remote_mtime > dst.stat().st_mtime
    except Exception:
        return True  # on error, assume update needed


def _download_one(url: str, dst: Path, force: bool, check_remote: bool) -> tuple[bool, bool]:
    """Fetch url and write to dst.

    Returns (attempted, success).
    - force=True: always download
    - check_remote=True: HEAD first, download only if remote is newer
    - otherwise: skip if dst exists
    """
    if dst.exists() and not force:
        if check_remote:
            if not _remote_newer(url, dst):
                return False, True  # up to date, skipped
        else:
            return False, True  # exists, skipped

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(resp.content)
        # Preserve Last-Modified as local mtime if available
        last_modified = resp.headers.get("Last-Modified")
        if last_modified:
            try:
                mtime = email.utils.parsedate_to_datetime(last_modified).timestamp()
                import os
                os.utime(dst, (mtime, mtime))
            except Exception:
                pass
        return True, True
    except Exception:
        return True, False


def sync_images(
    src_json: Path,
    images_dir: Path,
    set_filter: set[str] | None = None,
    workers: int = 8,
    rebuild: bool = False,
    force_rebuild: bool = False,
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

    cached = sum(1 for _, dst in tasks if dst.exists())
    missing = len(tasks) - cached

    print(f"Cards selected : {len(tasks):,}  (skipped by set filter: {skipped_set:,}, no image: {no_image:,})")
    print(f"Cached locally : {cached:,}  |  Missing: {missing:,}")
    if force_rebuild:
        print("Mode: force-rebuild (redownload all)")
    elif rebuild:
        print("Mode: rebuild (HEAD check existing, redownload if remote newer)")
    else:
        print("Mode: default (skip existing, download missing only)")

    if not tasks:
        print("Nothing to do.")
        return

    downloaded = skipped = err = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_download_one, url, dst, force_rebuild, rebuild): (url, dst)
            for url, dst in tasks
        }
        with tqdm(as_completed(futs), total=len(futs), unit="img", desc="Syncing") as pbar:
            for fut in pbar:
                attempted, success = fut.result()
                if not attempted:
                    skipped += 1
                elif success:
                    downloaded += 1
                else:
                    err += 1
                    url, _ = futs[fut]
                    pbar.write(f"FAIL: {url}")

    print(f"\nDownloaded: {downloaded:,}  Skipped: {skipped:,}  Errors: {err:,}")


def main() -> None:
    p = argparse.ArgumentParser(description="Sync Pokemon TCG card images")
    p.add_argument("--src", type=Path, default=cfg.pokemontcg_all_cards)
    p.add_argument("--images-dir", type=Path, default=cfg.pokemontcg_images_dir)
    p.add_argument("--set-ids", nargs="+", metavar="SET_ID",
                   help="Only sync images for these set IDs (e.g. base1 base2)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rebuild", action="store_true",
                   help="HEAD-check existing images and redownload if remote is newer")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Redownload every image unconditionally")
    args = p.parse_args()

    if not args.src.exists():
        print(f"ERROR: {args.src} not found", file=sys.stderr)
        print("Run: python 01_data_sources/pokemontcgio/01_sync_data.py", file=sys.stderr)
        sys.exit(1)

    set_filter = set(args.set_ids) if args.set_ids else None
    if set_filter:
        print(f"Set filter: {sorted(set_filter)}")

    sync_images(
        args.src, args.images_dir,
        set_filter=set_filter,
        workers=args.workers,
        rebuild=args.rebuild,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
