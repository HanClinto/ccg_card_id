#!/usr/bin/env python3
"""Pre-compute SIFT keypoints and descriptors for all cards in a Scryfall set.

Results are cached as .npz files (one per card) in:
    datasets/packopening/sift_cache/{set_code}/{card_id}.npz

This only needs to run once per set. Re-running is safe (cached files are
skipped unless --rebuild is passed).

Typical timing: ~0.3 s per card  → ~90 s for a 300-card set.

Usage (run from project root):
    python 02_data_sets/packopening/code/03_precompute_sift.py --set-code lea
    python 02_data_sets/packopening/code/03_precompute_sift.py --set-code otj,otp,big
    python 02_data_sets/packopening/code/03_precompute_sift.py --set-code lea --rebuild

Requires:
    pip install opencv-contrib-python-headless   (SIFT is in contrib)
    Scryfall reference images at cfg.scryfall_images_dir/.../
    Scryfall default_cards.json at cfg.data_dir/default_cards.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg


def load_cards_for_sets(data_dir: Path, set_codes: list[str]) -> list[dict]:
    """Return list of {card_id, set_code, image_path} for all English cards in the given sets."""
    default_cards_path = data_dir / "default_cards.json"
    if not default_cards_path.exists():
        raise FileNotFoundError(f"default_cards.json not found at {default_cards_path}")

    cards_json = json.loads(default_cards_path.read_text(encoding="utf-8"))
    set_codes_lower = {s.lower() for s in set_codes}

    results = []
    images_dir = cfg.scryfall_images_dir
    for card in cards_json:
        sc = card.get("set", "").lower()
        if sc not in set_codes_lower:
            continue
        if card.get("lang", "en") != "en":
            continue
        if card.get("layout") in ("art_series", "token"):
            continue

        card_id = card.get("id", "")
        if not card_id:
            continue

        # Scryfall image path: images/png/front/{first_char}/{second_char}/{card_id}.png
        img_path = images_dir / "front" / card_id[0] / card_id[1] / f"{card_id}.png"
        if not img_path.exists():
            continue

        results.append({
            "card_id": card_id,
            "set_code": sc,
            "image_path": img_path,
            "illustration_id": card.get("illustration_id", ""),
        })
    return results


def compute_sift_features(img_path: Path):
    """Return (keypoints_serialisable, descriptors) for an image, or (None, None) on failure."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(img, None)
    if kps is None or descs is None or len(kps) == 0:
        return None, None

    # Serialise keypoints as float32 array: [x, y, size, angle, response, octave, class_id]
    kp_array = np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id] for k in kps],
        dtype=np.float32,
    )
    return kp_array, descs.astype(np.float32)


def load_sift_features(npz_path: Path):
    """Load pre-computed SIFT features. Returns (kps_array, descriptors)."""
    data = np.load(str(npz_path))
    return data["keypoints"], data["descriptors"]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-compute SIFT features for all cards in a Scryfall set"
    )
    p.add_argument(
        "--set-code",
        required=True,
        help="Comma-separated Scryfall set code(s), e.g. 'lea' or 'otj,otp,big'",
    )
    p.add_argument("--rebuild", action="store_true", help="Recompute even if cache exists")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    set_codes = [s.strip().lower() for s in args.set_code.split(",") if s.strip()]
    cache_root = args.data_dir / "datasets" / "packopening" / "sift_cache"

    print(f"Loading cards for sets: {set_codes}")
    cards = load_cards_for_sets(args.data_dir, set_codes)
    print(f"  Found {len(cards)} English cards with reference images")

    cached = computed = failed = 0
    for card in cards:
        card_id = card["card_id"]
        sc = card["set_code"]
        out_path = cache_root / sc / f"{card_id}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.rebuild:
            cached += 1
            continue

        kp_array, descs = compute_sift_features(card["image_path"])
        if kp_array is None:
            print(f"  SKIP (no features): {card_id}")
            failed += 1
            continue

        np.savez_compressed(
            str(out_path),
            keypoints=kp_array,
            descriptors=descs,
            card_id=np.array([card_id]),
            illustration_id=np.array([card.get("illustration_id", "")]),
        )
        computed += 1

    print(f"\nDone. Computed: {computed}  Cached (skipped): {cached}  Failed: {failed}")
    total = computed + cached
    print(f"SIFT cache: {cache_root}")
    print(f"Total cards available for matching: {total}")


if __name__ == "__main__":
    main()
