#!/usr/bin/env python3
"""Pre-compute SIFT keypoints and descriptors for Scryfall card images.

Cached as .npz files in: datasets/packopening/sift_cache/{set_code}/{card_id}.npz
Already-cached files are skipped automatically; use --rebuild to force recompute.

Typical timing: ~0.3 s/card → ~90 s for a 300-card set.

Usage (run from project root):
    # Specific set(s)
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --set-code lea
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --set-code otj,otp,big

    # All English sets in the Scryfall catalog (safe to re-run; skips cached)
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --all-scryfall

    # All English sets in the Scryfall catalog (non-English example)
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --all-scryfall --lang it

    # Only sets referenced by packopening DB videos
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --all-db

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg
from ccg_card_id.catalog import catalog


def sift_cache_key(set_code: str, lang: str) -> str:
    """Directory name under sift_cache/ for a given set + language.

    English uses the bare set code (backward-compatible with existing caches).
    Other languages append _{lang}, e.g. 'leg_it' for Italian Legends.
    """
    lang = lang.strip().lower() or "en"
    return set_code if lang == "en" else f"{set_code}_{lang}"


def load_cards_for_sets(data_dir: Path, set_codes: list[str], lang: str = "en") -> list[dict]:
    lang = lang.strip().lower() or "en"
    images_dir = cfg.scryfall_images_dir

    rows = catalog.cards_for_sets(set_codes, lang=lang)
    results = []
    for card in rows:
        if card.get("layout") in ("art_series", "token"):
            continue
        card_id = card["id"]
        if not card_id:
            continue

        # Front face
        front_path = images_dir / "front" / card_id[0] / card_id[1] / f"{card_id}.png"
        if front_path.exists():
            results.append({
                "card_id": card_id,
                "set_code": card["set_code"],
                "lang": lang,
                "face": "front",
                "image_path": front_path,
                "illustration_id": card.get("illustration_id", ""),
                "cache_stem": card_id,
            })

        # Back face (DFCs only)
        if card.get("back_image_uri_png"):
            back_path = images_dir / "back" / card_id[0] / card_id[1] / f"{card_id}.png"
            if back_path.exists():
                results.append({
                    "card_id": card_id,
                    "set_code": card["set_code"],
                    "lang": lang,
                    "face": "back",
                    "image_path": back_path,
                    "illustration_id": card.get("back_illustration_id", ""),
                    "cache_stem": f"{card_id}_back",
                })

    return results


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


def load_sift_features(npz_path: Path):
    data = np.load(str(npz_path))
    return data["keypoints"], data["descriptors"]


def _process_set(set_code: str, lang: str, data_dir: Path, cache_root: Path,
                 workers: int, rebuild: bool) -> tuple[int, int, int]:
    """Precompute SIFT for a single set+lang. Returns (computed, cached, failed)."""
    cards = load_cards_for_sets(data_dir, [set_code], lang=lang)
    if not cards:
        return 0, 0, 0

    pending = []
    cached = 0
    for card in cards:
        cache_dir = sift_cache_key(card["set_code"], card["lang"])
        out_path = cache_root / cache_dir / f"{card['cache_stem']}.npz"
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
                      unit="card", desc=f"SIFT {set_code}",
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
    p = argparse.ArgumentParser(description="Pre-compute SIFT features for Scryfall cards")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--set-code",
                        help="Space/comma-separated set code(s), e.g. 'lea' or '2x2 t2x2'")
    source.add_argument("--video-id",
                        help="Look up set codes from the DB for this YouTube video ID")
    source.add_argument("--all-db", action="store_true",
                        help="Precompute for all set codes present in the packopening DB")
    source.add_argument("--all-scryfall", action="store_true",
                        help="Precompute for every set in the Scryfall catalog for --lang (default: en)")
    p.add_argument("--lang", default="en",
                   help="Scryfall language code (default: en). Examples: it, fr, de, ja. "
                        "Auto-detected from DB when using --video-id.")
    p.add_argument("--rebuild", action="store_true",
                   help="Recompute even if cache file already exists")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Parallel worker processes (default: all CPU cores)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    lang = args.lang.strip().lower() or "en"
    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    cache_root = args.data_dir / "datasets" / "packopening" / "sift_cache"

    # ── Resolve the list of (set_codes_batch, lang) jobs ─────────────────────
    # For --all-scryfall we process one set at a time so progress is visible.
    # For other modes we keep the existing batched behaviour.

    if args.set_code:
        set_codes = [s for s in re.split(r"[\s,]+", args.set_code.strip().lower()) if s]
        jobs: list[tuple[list[str], str]] = [(set_codes, lang)]
        one_set_at_a_time = False

    elif args.video_id:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("db", Path(__file__).parents[1] / "db.py")
        _db = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_db)
        con = _db.open_db(db_path)
        row = con.execute("SELECT set_codes, lang, title FROM videos WHERE video_id=?",
                          (args.video_id,)).fetchone()
        if not row or not row["set_codes"]:
            print(f"ERROR: video_id '{args.video_id}' not found or has no set_codes.", file=sys.stderr)
            sys.exit(1)
        print(f"  [{args.video_id}] {row['title'][:70]}")
        set_codes = [s for s in re.split(r"[\s,]+", row["set_codes"].strip().lower()) if s]
        lang = args.lang.strip().lower() or row["lang"] or "en"
        jobs = [(set_codes, lang)]
        one_set_at_a_time = False

    elif args.all_db:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("db", Path(__file__).parents[1] / "db.py")
        _db = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_db)
        con = _db.open_db(db_path)
        rows = con.execute(
            "SELECT set_codes, lang FROM videos WHERE set_codes IS NOT NULL AND set_codes != ''"
        ).fetchall()
        seen: dict[str, set] = {}  # lang -> set of set_codes
        for row in rows:
            row_lang = (row["lang"] or "en").strip().lower()
            for code in re.split(r"[\s,]+", row["set_codes"].strip().lower()):
                if code:
                    seen.setdefault(row_lang, set()).add(code)
        jobs = [(sorted(codes), row_lang) for row_lang, codes in sorted(seen.items())]
        one_set_at_a_time = False

    else:  # --all-scryfall
        all_set_codes = sorted(catalog.set_codes_for_lang(lang))
        print(f"Found {len(all_set_codes)} set codes for lang={lang} in catalog.")
        jobs = [([sc], lang) for sc in all_set_codes]
        one_set_at_a_time = True

    # ── Process ───────────────────────────────────────────────────────────────
    total_cached = total_computed = total_failed = total_skipped = 0

    if one_set_at_a_time:
        # --all-scryfall: one set at a time with a per-set summary line
        for (set_codes_batch, batch_lang) in tqdm(jobs, desc="Sets", unit="set"):
            set_code = set_codes_batch[0]
            computed, cached, failed = _process_set(
                set_code, batch_lang, args.data_dir, cache_root,
                args.workers, args.rebuild,
            )
            if computed == 0 and cached == 0 and failed == 0:
                total_skipped += 1  # no images found for this set
            else:
                total_computed += computed
                total_cached   += cached
                total_failed   += failed
                if computed or failed:
                    tqdm.write(f"  {set_code}: +{computed} computed, {cached} cached, {failed} failed")
    else:
        for set_codes_batch, batch_lang in jobs:
            lang_label = f" lang={batch_lang}" if batch_lang != "en" else ""
            print(f"\nLoading cards for sets {set_codes_batch}{lang_label}")
            cards = load_cards_for_sets(args.data_dir, set_codes_batch, lang=batch_lang)
            print(f"  {len(cards)} cards with reference images")
            if not cards:
                print("  (no images found — check that Scryfall images are synced)")
                continue

            pending = []
            cached = 0
            for card in cards:
                cache_dir = sift_cache_key(card["set_code"], card["lang"])
                out_path = cache_root / cache_dir / f"{card['cache_stem']}.npz"
                if out_path.exists() and not args.rebuild:
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
                n_workers = min(args.workers, len(pending))
                print(f"  Computing {len(pending)} cards with {n_workers} workers "
                      f"({cached} already cached)...")
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_worker, j): j for j in pending}
                    with tqdm(as_completed(futures), total=len(pending),
                              unit="card", desc="SIFT precompute") as pbar:
                        for fut in pbar:
                            result = fut.result()
                            if result == "computed":
                                computed += 1
                            elif result == "failed":
                                failed += 1
                            pbar.set_postfix(computed=computed, cached=cached, failed=failed)
            else:
                print(f"  All {cached} cards already cached.")

            print(f"  Computed: {computed}  Cached: {cached}  Failed: {failed}")
            total_cached += cached; total_computed += computed; total_failed += failed

    print(f"\nDone. Computed: {total_computed}  Cached: {total_cached}  "
          f"Failed: {total_failed}  Sets with no images: {total_skipped}")
    print(f"Cache: {cache_root}")


if __name__ == "__main__":
    main()
