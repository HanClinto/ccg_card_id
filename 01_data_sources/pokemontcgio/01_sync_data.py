#!/usr/bin/env python3
"""Sync Pokemon TCG card metadata from the public API.

Downloads all card objects (paginated) and writes them to:
    {cfg.data_dir}/catalog/pokemontcg/all_cards.json

Skips download if the local file is newer than --max-age-days (default 7).
Pass --rebuild to force a full re-download regardless of age.

API docs: https://docs.pokemontcg.io/

Usage (run from project root):
    python 01_data_sources/pokemontcgio/01_sync_data.py
    python 01_data_sources/pokemontcgio/01_sync_data.py --rebuild
    python 01_data_sources/pokemontcgio/01_sync_data.py --api-key YOUR_KEY
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg

BASE_URL = "https://api.pokemontcg.io/v2"
RATE_LIMIT_DELAY = 0.1  # seconds between requests


def _needs_download(out_path: Path, max_age_days: int, rebuild: bool) -> bool:
    if rebuild or not out_path.exists():
        return True
    age = (dt.datetime.now(dt.UTC) - dt.datetime.fromtimestamp(out_path.stat().st_mtime, dt.UTC)).days
    print(f"Local cache age: {age} days")
    return age > max_age_days


def sync_pokemon_data(
    out_path: Path,
    api_key: str | None = None,
    page_size: int = 250,
    max_age_days: int = 7,
    rebuild: bool = False,
) -> bool:
    """Fetch all cards from the Pokemon TCG API and write to out_path.

    Returns True if data was (re-)downloaded, False if cache was reused.
    """
    if not _needs_download(out_path, max_age_days, rebuild):
        print(f"Cache is up to date: {out_path}")
        return False

    headers = {"X-Api-Key": api_key} if api_key else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Pokemon TCG data → {out_path}")
    all_cards: list[dict] = []
    page = 1

    with tqdm(desc="Downloading", unit="page") as pbar:
        while True:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.get(
                f"{BASE_URL}/cards",
                headers=headers,
                params={"page": page, "pageSize": page_size},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("data", [])
            if not batch:
                break
            all_cards.extend(batch)
            pbar.update(1)
            pbar.set_postfix({"cards": len(all_cards)})
            if len(all_cards) >= data.get("totalCount", 0):
                break
            page += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_cards, f, indent=2, ensure_ascii=False)

    print(f"Downloaded {len(all_cards):,} cards → {out_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Sync Pokemon TCG card metadata")
    p.add_argument("--rebuild", action="store_true",
                   help="Force re-download even if cache is fresh")
    p.add_argument("--max-age-days", type=int, default=7,
                   help="Re-download if cache is older than this many days (default: 7)")
    p.add_argument("--page-size", type=int, default=250)
    p.add_argument("--api-key", default=os.environ.get("POKEMON_TCG_API_KEY"),
                   help="Optional API key (or set POKEMON_TCG_API_KEY env var)")
    p.add_argument("--out", type=Path, default=cfg.pokemontcg_all_cards,
                   help="Output path (default: cfg.pokemontcg_all_cards)")
    args = p.parse_args()

    changed = sync_pokemon_data(
        out_path=args.out,
        api_key=args.api_key,
        page_size=args.page_size,
        max_age_days=args.max_age_days,
        rebuild=args.rebuild,
    )
    print(f"Data updated: {changed}")

    with args.out.open(encoding="utf-8") as f:
        cards = json.load(f)
    sets = {c["set"]["id"] for c in cards if c.get("set")}
    print(f"Total cards: {len(cards):,}  |  Sets: {len(sets):,}")


if __name__ == "__main__":
    main()
