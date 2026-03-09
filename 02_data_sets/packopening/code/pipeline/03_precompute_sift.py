#!/usr/bin/env python3
"""Pre-compute SIFT keypoints and descriptors for all cards in a Scryfall set.

Cached as .npz files in: datasets/packopening/sift_cache/{set_code}/{card_id}.npz
Needs to run once per set; re-running skips cached files (use --rebuild to force).

Typical timing: ~0.3 s/card → ~90 s for a 300-card set.

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/03_precompute_sift.py --set-code lea
    python 02_data_sets/packopening/code/pipeline/03_precompute_sift.py --set-code otj,otp,big
    python 02_data_sets/packopening/code/pipeline/03_precompute_sift.py --set-code lea --rebuild

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from tqdm import tqdm
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg


def load_cards_for_sets(data_dir: Path, set_codes: list[str]) -> list[dict]:
    default_cards_path = data_dir / "default_cards.json"
    if not default_cards_path.exists():
        raise FileNotFoundError(f"default_cards.json not found: {default_cards_path}")

    cards_json = json.loads(default_cards_path.read_text(encoding="utf-8"))
    set_codes_lower = {s.lower() for s in set_codes}
    images_dir = cfg.scryfall_images_dir

    results = []
    for card in cards_json:
        if card.get("set", "").lower() not in set_codes_lower:
            continue
        if card.get("lang", "en") != "en":
            continue
        if card.get("layout") in ("art_series", "token"):
            continue
        card_id = card.get("id", "")
        if not card_id:
            continue
        img_path = images_dir / "front" / card_id[0] / card_id[1] / f"{card_id}.png"
        if not img_path.exists():
            continue
        results.append({
            "card_id": card_id,
            "set_code": card["set"].lower(),
            "image_path": img_path,
            "illustration_id": card.get("illustration_id", ""),
        })
    return results


def compute_sift_features(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(img, None)
    if kps is None or descs is None or len(kps) == 0:
        return None, None
    kp_array = np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id] for k in kps],
        dtype=np.float32,
    )
    return kp_array, descs.astype(np.float32)


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
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"

    if args.set_code:
        set_codes = [s for s in re.split(r"[\s,]+", args.set_code.strip().lower()) if s]
    else:
        # Import db helper (sibling of this script's parent)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("db", Path(__file__).parents[1] / "db.py")
        _db = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_db)
        con = _db.open_db(db_path)

        if args.video_id:
            row = con.execute("SELECT set_codes, title FROM videos WHERE video_id=?",
                              (args.video_id,)).fetchone()
            if not row or not row["set_codes"]:
                print(f"ERROR: video_id '{args.video_id}' not found or has no set_codes.", file=sys.stderr)
                sys.exit(1)
            print(f"  [{args.video_id}] {row['title'][:70]}")
            set_codes = [s for s in re.split(r"[\s,]+", row["set_codes"].strip().lower()) if s]
        else:  # --all
            rows = con.execute(
                "SELECT set_codes FROM videos WHERE set_codes IS NOT NULL AND set_codes != ''"
            ).fetchall()
            set_codes = sorted({
                code
                for row in rows
                for code in re.split(r"[\s,]+", row["set_codes"].strip().lower())
                if code
            })
    cache_root = args.data_dir / "datasets" / "packopening" / "sift_cache"

    print(f"Loading cards for sets: {set_codes}")
    cards = load_cards_for_sets(args.data_dir, set_codes)
    print(f"  {len(cards)} English cards with reference images")

    cached = computed = failed = 0
    with tqdm(cards, unit="card", desc="SIFT precompute") as pbar:
        for card in pbar:
            out_path = cache_root / card["set_code"] / f"{card['card_id']}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.rebuild:
                cached += 1
                pbar.set_postfix(computed=computed, cached=cached, failed=failed)
                continue
            pbar.set_postfix(set=card["set_code"], computed=computed, cached=cached, failed=failed)
            kp_array, descs = compute_sift_features(card["image_path"])
            if kp_array is None:
                failed += 1
                continue
            np.savez_compressed(str(out_path), keypoints=kp_array, descriptors=descs,
                                card_id=np.array([card["card_id"]]),
                                illustration_id=np.array([card["illustration_id"]]))
            computed += 1

    print(f"Done. Computed: {computed}  Cached: {cached}  Failed: {failed}")
    print(f"Cache: {cache_root}")


if __name__ == "__main__":
    main()
