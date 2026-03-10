#!/usr/bin/env python3
"""Pre-compute SIFT keypoints and descriptors for all cards in a Scryfall set.

Cached as .npz files in: datasets/packopening/sift_cache/{set_code}/{card_id}.npz
Needs to run once per set; re-running skips cached files (use --rebuild to force).

Typical timing: ~0.3 s/card → ~90 s for a 300-card set.

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --set-code lea
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --set-code otj,otp,big
    python 02_data_sets/packopening/code/pipeline/02_precompute_sift.py --set-code lea --rebuild

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
        img_path = images_dir / "front" / card_id[0] / card_id[1] / f"{card_id}.png"
        if not img_path.exists():
            continue
        results.append({
            "card_id": card_id,
            "set_code": card["set_code"],
            "lang": lang,
            "image_path": img_path,
            "illustration_id": card.get("illustration_id", ""),
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


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-compute SIFT features for a Scryfall set")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--set-code",
                        help="Space/comma-separated set code(s), e.g. 'lea' or '2x2 t2x2'")
    source.add_argument("--video-id",
                        help="Look up set codes from the DB for this YouTube video ID")
    source.add_argument("--all", action="store_true",
                        help="Precompute for all set codes present in the DB (pending/downloaded/etc.)")
    p.add_argument("--lang", default="",
                   help="Scryfall language code (e.g. 'it', 'fr', 'de', 'ja'). "
                        "Defaults to 'en'. Auto-detected from DB when using --video-id.")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Parallel worker processes (default: all CPU cores)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"

    # List of (set_codes, lang) pairs to precompute
    jobs: list[tuple[list[str], str]] = []

    if args.set_code:
        set_codes = [s for s in re.split(r"[\s,]+", args.set_code.strip().lower()) if s]
        lang = args.lang.strip().lower() or "en"
        jobs = [(set_codes, lang)]
    else:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("db", Path(__file__).parents[1] / "db.py")
        _db = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_db)
        con = _db.open_db(db_path)

        if args.video_id:
            row = con.execute("SELECT set_codes, lang, title FROM videos WHERE video_id=?",
                              (args.video_id,)).fetchone()
            if not row or not row["set_codes"]:
                print(f"ERROR: video_id '{args.video_id}' not found or has no set_codes.", file=sys.stderr)
                sys.exit(1)
            print(f"  [{args.video_id}] {row['title'][:70]}")
            set_codes = [s for s in re.split(r"[\s,]+", row["set_codes"].strip().lower()) if s]
            lang = args.lang.strip().lower() or row["lang"] or "en"
            jobs = [(set_codes, lang)]
        else:  # --all: group by (set_code, lang)
            rows = con.execute(
                "SELECT set_codes, lang FROM videos WHERE set_codes IS NOT NULL AND set_codes != ''"
            ).fetchall()
            seen: dict[tuple, set] = {}  # lang -> set of set_codes
            for row in rows:
                lang = (row["lang"] or "en").strip().lower()
                for code in re.split(r"[\s,]+", row["set_codes"].strip().lower()):
                    if code:
                        seen.setdefault(lang, set()).add(code)
            jobs = [(sorted(codes), lang) for lang, codes in sorted(seen.items())]

    cache_root = args.data_dir / "datasets" / "packopening" / "sift_cache"
    total_cached = total_computed = total_failed = 0

    for set_codes, lang in jobs:
        print(f"\nLoading cards for sets {set_codes} lang={lang}")
        cards = load_cards_for_sets(args.data_dir, set_codes, lang=lang)
        lang_label = f"{lang} " if lang != "en" else ""
        print(f"  {len(cards)} {lang_label}cards with reference images")
        if not cards:
            print("  (no images found — check that Scryfall images are synced for this language)")
            continue

        pending = []
        cached = 0
        for card in cards:
            cache_dir = sift_cache_key(card["set_code"], card["lang"])
            out_path = cache_root / cache_dir / f"{card['card_id']}.npz"
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
            workers = min(args.workers, len(pending))
            print(f"  Computing {len(pending)} cards with {workers} workers "
                  f"({cached} already cached)...")
            with ProcessPoolExecutor(max_workers=workers) as pool:
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

    print(f"\nDone. Computed: {total_computed}  Cached: {total_cached}  Failed: {total_failed}")
    print(f"Cache: {cache_root}")


if __name__ == "__main__":
    main()
