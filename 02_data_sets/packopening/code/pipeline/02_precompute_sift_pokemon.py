#!/usr/bin/env python3
"""Pre-compute SIFT keypoints and descriptors for Pokemon TCG card images.

Cached as .npz files in:
    datasets/packopening/sift_cache/pokemon/{set_id}/{card_id}.npz

Already-cached files are skipped automatically; use --rebuild to force recompute.

Usage (run from project root):
    # Specific set(s)
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift_pokemon.py --set-code base1
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift_pokemon.py --set-code sv10,zsv10pt5

    # All sets in the Pokemon TCG catalog
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift_pokemon.py --all

    # Only sets referenced by packopening DB Pokemon videos
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift_pokemon.py --all-db

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg
from ccg_card_id.pokemon_catalog import pokemon_catalog


# ---------------------------------------------------------------------------
# Helpers (shared with MTG precompute — kept inline to avoid cross-import)
# ---------------------------------------------------------------------------

def compute_sift_features(img_path: Path, sift=None):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    if sift is None:
        sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(img, None)
    if kps is None or descs is None or len(kps) == 0:
        return None, None
    kp_array = np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id] for k in kps],
        dtype=np.float32,
    )
    return kp_array, descs.astype(np.float32)


def _worker(job: tuple) -> str:
    """Top-level worker for ProcessPoolExecutor. Returns 'computed', 'cached', or 'failed'."""
    img_path_str, out_path_str, card_id, illustration_id = job
    out_path = Path(out_path_str)
    if out_path.exists():
        return "cached"
    kp_array, descs = compute_sift_features(Path(img_path_str))
    if kp_array is None:
        return "failed"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), keypoints=kp_array, descriptors=descs,
             card_id=np.array([card_id]),
             illustration_id=np.array([illustration_id]))
    return "computed"


def _pokemon_image_path(images_dir: Path, card_id: str) -> Path:
    """Return the expected image path for a Pokemon TCG card ID (e.g. 'base1-4')."""
    a = card_id[0] if card_id else "x"
    b = card_id[1] if len(card_id) > 1 else "x"
    return images_dir / "large" / a / b / f"{card_id}.png"


def load_cards_for_sets(set_ids: list[str], images_dir: Path) -> list[dict]:
    """Return card dicts for the given set IDs that have a local image file."""
    cards = pokemon_catalog.cards_for_sets(set_ids)
    results = []
    for card in cards:
        card_id = card.get("id", "")
        if not card_id:
            continue
        img_path = _pokemon_image_path(images_dir, card_id)
        if not img_path.exists():
            continue
        results.append({
            "card_id":        card_id,
            "set_id":         card.get("set_id", ""),
            "illustration_id": card.get("illustration_id", ""),
            "image_path":     img_path,
        })
    return results


def _process_set(set_id: str, images_dir: Path, cache_root: Path,
                 workers: int, rebuild: bool) -> tuple[int, int, int]:
    """Precompute SIFT for a single Pokemon set. Returns (computed, cached, failed)."""
    cards = load_cards_for_sets([set_id], images_dir)
    if not cards:
        return 0, 0, 0

    cache_dir = cache_root / "pokemon" / set_id
    pending = []
    cached = 0
    for card in cards:
        out_path = cache_dir / f"{card['card_id']}.npz"
        if out_path.exists() and not rebuild:
            cached += 1
        else:
            pending.append((
                str(card["image_path"]),
                str(out_path),
                card["card_id"],
                card["illustration_id"],
            ))

    computed = failed = 0
    if pending:
        n_workers = min(workers, len(pending))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, j): j for j in pending}
            with tqdm(as_completed(futures), total=len(pending),
                      unit="card", desc=f"SIFT {set_id}",
                      leave=False) as pbar:
                for fut in pbar:
                    result = fut.result()
                    if result == "computed":
                        computed += 1
                    elif result == "failed":
                        failed += 1
                    pbar.set_postfix(computed=computed, cached=cached, failed=failed)

    return computed, cached, failed


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-compute SIFT features for Pokemon TCG cards")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--set-code",
                        help="Space/comma-separated Pokemon set ID(s), e.g. 'base1' or 'sv10,zsv10pt5'")
    source.add_argument("--all", action="store_true",
                        help="Precompute for every set in the Pokemon TCG catalog")
    source.add_argument("--all-db", action="store_true",
                        help="Precompute for all Pokemon set codes present in the packopening DB")
    p.add_argument("--rebuild", action="store_true",
                   help="Recompute even if cache file already exists")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Parallel worker processes (default: all CPU cores)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    images_dir = cfg.pokemontcg_images_dir
    cache_root = args.data_dir / "datasets" / "packopening" / "sift_cache"

    # ── Resolve set IDs ──────────────────────────────────────────────────────
    if args.set_code:
        set_ids = [s for s in re.split(r"[\s,]+", args.set_code.strip().lower()) if s]

    elif args.all_db:
        db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
        _spec = _ilu.spec_from_file_location("db", Path(__file__).parents[1] / "db.py")
        _db = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_db)
        con = _db.open_db(db_path)
        rows = con.execute(
            "SELECT set_codes FROM videos WHERE game='pokemon' AND set_codes IS NOT NULL AND set_codes != ''"
        ).fetchall()
        seen: set[str] = set()
        for row in rows:
            for code in re.split(r"[\s,]+", row["set_codes"].strip().lower()):
                if code:
                    seen.add(code)
        set_ids = sorted(seen)
        print(f"Found {len(set_ids)} Pokemon set codes in packopening DB: {set_ids}")

    else:  # --all
        set_ids = sorted(pokemon_catalog.valid_set_codes())
        print(f"Found {len(set_ids)} sets in Pokemon TCG catalog.")

    if not set_ids:
        print("No set IDs to process.")
        return

    # ── Process ───────────────────────────────────────────────────────────────
    total_computed = total_cached = total_failed = total_skipped = 0

    for set_id in tqdm(set_ids, desc="Sets", unit="set"):
        computed, cached, failed = _process_set(
            set_id, images_dir, cache_root, args.workers, args.rebuild,
        )
        if computed == 0 and cached == 0 and failed == 0:
            total_skipped += 1
        else:
            total_computed += computed
            total_cached   += cached
            total_failed   += failed
            if computed or failed:
                tqdm.write(f"  {set_id}: +{computed} computed, {cached} cached, {failed} failed")

    print(f"\nDone. Computed: {total_computed}  Cached: {total_cached}  "
          f"Failed: {total_failed}  Sets with no images: {total_skipped}")
    print(f"Cache: {cache_root / 'pokemon'}")


if __name__ == "__main__":
    main()
