#!/usr/bin/env python3
"""Sync Scryfall card images (step 3 of 3 in the Scryfall update pipeline).

Downloads PNG images for all cards in the catalog DB into:
  catalog/scryfall/images/png/front/{a}/{b}/{uuid}.png
  catalog/scryfall/images/png/back/{a}/{b}/{uuid}.png  (DFCs only)

Image URLs from Scryfall include a timestamp query parameter; the local file
mtime is compared against this timestamp to skip already-current images.

Run after syncing bulk data and rebuilding the card DB:
  python 01_data_sources/scryfall/01_sync_data.py
  python 01_data_sources/scryfall/02_build_card_db.py
  python 01_data_sources/scryfall/03_sync_images.py

Usage (run from project root):
    python 01_data_sources/scryfall/03_sync_images.py
    python 01_data_sources/scryfall/03_sync_images.py --sets lea leb 2ed
"""

import argparse
import os
import re
import sys
import requests
# import orjson
import json
from datetime import datetime
import datetime as dt
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg

image_quality = "png"

#BULK_DATA_PATH = str(cfg.data_dir / "default_cards.json")
BULK_DATA_PATH = str(cfg.data_dir / "all_cards.json")
IMAGES_DIR = str(cfg.scryfall_images_dir)

def load_bulk_data():
    """
    Loads the bulk data JSON from local cache.
    """
    then = datetime.now()
    print(f"Loading bulk data from {BULK_DATA_PATH}...")
    # Load bulk data JSON
    with open(BULK_DATA_PATH, "r", encoding="utf-8") as f:
        # bulk_data = orjson.loads(f.read())
        bulk_data = json.load(f)
    print(f"Loaded {len(bulk_data)} cards in {(datetime.now() - then).total_seconds():.2f} seconds.")
    return bulk_data

def prioritize_cards(bulk_data):
    """
    Prioritizes cards that are most ambiguous according to the data that we have.
    """
    then = datetime.now()
    scryfall_cards_by_oracle_id = {}
    scryfall_cards_by_illustration_id = {}
    for card in bulk_data:
        if card.get("image_status") in ("missing", "placeholder"):
            continue  # Skip cards without valid images

        if english_only and card.get("lang") != "en":
            continue  # Skip non-English cards

        oracle_id = card.get("oracle_id")
        if oracle_id:
            scryfall_cards_by_oracle_id.setdefault(oracle_id, []).append(card)

        illustration_id = card.get("illustration_id")
        if illustration_id:
            scryfall_cards_by_illustration_id.setdefault(illustration_id, []).append(card)

    print(f"Prioritized {len(scryfall_cards_by_oracle_id)} unique oracle_ids and {len(scryfall_cards_by_illustration_id)} unique illustration_ids in {(datetime.now() - then).total_seconds():.2f} seconds.")

    return scryfall_cards_by_oracle_id, scryfall_cards_by_illustration_id

def sync_scryfall_images(cards, lang_filter: str | None = None, set_filter: set | None = None,
                         first_n: int = 0):
    """
    Syncs Scryfall card images based on the local all_cards.json bulk data file.
    Downloads images that are missing or outdated.
    first_n: if > 0, stop after downloading this many images (skipped/cached don't count).
    """
    then = datetime.now()
    print(f"Syncing images for {len(cards)} cards...")

    # Display progress using tqdm
    with tqdm(cards, desc="Syncing images", unit="card") as pbar:
        for card in pbar:
            card_id = card["id"]
            pbar.set_postfix({"card_id": card_id})
            faces = [card]
            if "image_uris" not in card:
                faces = card.get("card_faces", [card])  # Use card_faces if present, else single face
            # If the card's image_status is "missing" or "placeholder", skip downloading images
            if card.get("image_status") in ["missing", "placeholder"]:
                continue
            if lang_filter and card.get("lang") != lang_filter:
                continue
            if set_filter and card.get("set", "").lower() not in set_filter:
                continue
            for face in faces:
                image_uris = face.get("image_uris", {})
                image_url = image_uris.get(image_quality)
                if not image_url:
                    continue  # No image available for this quality
                # Determine local image path
                subdir1 = card_id[0]
                subdir2 = card_id[1]
                # Given an image URL that looks like: 
                #  https://cards.scryfall.io/small/front/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.jpg?1651492800
                #  https://cards.scryfall.io/small/back/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.jpg?1651492800
                # Extract the string "front" or "back" from the image URL
                face_name = image_url.split("/")[4]
                local_dir = os.path.join(IMAGES_DIR, face_name, subdir1, subdir2)
                os.makedirs(local_dir, exist_ok=True)
                image_extension = image_url.split(".")[-1].split("?")[0]  # Extract extension before query params
                local_image_path = os.path.join(local_dir, f"{card_id}.{image_extension}")
                # Extract timestamp from URL
                timestamp_str = image_url.split("?")[-1]
                try:
                    remote_timestamp = int(timestamp_str)
                    remote_dt = datetime.fromtimestamp(remote_timestamp, dt.UTC)
                except ValueError:
                    remote_dt = None
        
                # Check if local file exists and its mtime
                need_download = False
                if not os.path.exists(local_image_path):
                    need_download = True
                elif remote_dt:
                    local_mtime = datetime.fromtimestamp(os.path.getmtime(local_image_path), dt.UTC)
                    if remote_dt > local_mtime:
                        need_download = True
            
                # Download image if needed
                # Delete the local image if it's only a placeholder
                if card.get("image_status") == "placeholder" and os.path.exists(local_image_path):
                    os.remove(local_image_path)
                elif need_download:
                    # Update tqdm bar instead of printing
                    pbar.set_postfix({"card_id": card_id})
                    try:
                        img_resp = requests.get(image_url)
                        img_resp.raise_for_status()
                        with open(local_image_path, "wb") as img_file:
                            img_file.write(img_resp.content)
                        # Update mtime to match remote timestamp
                        if remote_dt:
                            mtime = remote_dt.timestamp()
                            os.utime(local_image_path, (mtime, mtime))
                        if first_n > 0:
                            first_n -= 1
                            if first_n == 0:
                                pbar.write(f"Reached --first-n limit, stopping.")
                                return
                    except Exception as e:
                        pbar.write(f"Error downloading image for card {card_id} from {image_url}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sync Scryfall card images")
    p.add_argument("--lang", default="en",
                   help="Language code to sync (default: en). Use 'all' for all languages.")
    p.add_argument("--set-code", dest="set_codes",
                   help="Comma/space-separated set codes to limit download (default: all sets)")
    p.add_argument("--first-n", type=int, default=0, metavar="N",
                   help="Stop after downloading N new images (0 = no limit)")
    args = p.parse_args()

    lang_filter = None if args.lang == "all" else args.lang
    set_filter = None
    if args.set_codes:
        set_filter = {s.lower() for s in re.split(r"[\s,]+", args.set_codes.strip()) if s}

    if lang_filter:
        print(f"Language filter: {lang_filter}")
    if set_filter:
        print(f"Set filter: {sorted(set_filter)}")

    bulk_data = load_bulk_data()
    sync_scryfall_images(bulk_data, lang_filter=lang_filter, set_filter=set_filter,
                         first_n=args.first_n)
