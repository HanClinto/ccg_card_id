# Sync Scryfall card images
# Given a bulk data JSON from Scryfall exists in ./cache/all_cards.json
# Download images for all cards listed in that file into ./cache/images/png/front/{card_id}[0]/{card_id}[1]/{card_id}.jpg
# Cards in the bulk data JSON have an "image_uris" field with URLs for various image types and sizes
# Examples:
#  https://cards.scryfall.io/png/front/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.png?1651492800
#  https://cards.scryfall.io/png/back/0/2/02d6d693-f1f3-4317-bcc0-c21fa8490d38.png?1651492800
# Note that the image URL includes a timestamp query parameter, which should be compared against the local file mtime to determine if the image needs to be re-downloaded
# If the local image file does not exist, or if the timestamp in the URL is newer than the local file mtime, download the image and save it to the appropriate path
# If a card has multiple faces, then download images for every face and save in the appropriate locations
# For a single-faced card, then image_uris is directly on the card object
# For a multi-faced card, then image_uris is on each face object within the object's card_faces array

import os
import requests
# import orjson
import json
from datetime import datetime
import datetime as dt
from tqdm import tqdm

image_types = ["png", "large", "normal", "small", "art_crop", "border_crop"]
image_quality = "png"

english_only = True

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
BULK_DATA_PATH = os.path.join(CACHE_DIR, "all_cards.json")
IMAGES_DIR = os.path.join(CACHE_DIR, "images", image_quality)

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

def sync_scryfall_images(cards):
    """
    Syncs Scryfall card images based on the local all_cards.json bulk data file.
    Downloads images that are missing or outdated.
    """
    then = datetime.now()
    print(f"Syncing images for {len(cards)} cards...")
    
    # Display progress using tqdm
    with tqdm(cards, desc="Syncing images", unit="card") as pbar:
        for card in pbar:
            card_id = card["id"]
            pbar.set_postfix({"card_id": card_id})
            faces = card.get("card_faces", [card])  # Use card_faces if present, else single face
            # If the card's image_status is "missing" or "placeholder", skip downloading images
            if card.get("image_status") in ["missing", "placeholder"]:
                continue
            if english_only and card.get("lang") != "en":
                continue  # Skip non-English cards
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
                    img_resp = requests.get(image_url)
                    img_resp.raise_for_status()
                    with open(local_image_path, "wb") as img_file:
                        img_file.write(img_resp.content)
                    # Update mtime to match remote timestamp
                    if remote_dt:
                        mtime = remote_dt.timestamp()
                        os.utime(local_image_path, (mtime, mtime))

if __name__ == "__main__":
    bulk_data = load_bulk_data()

    scryfall_cards_by_oracle_id, scryfall_cards_by_illustration_id = prioritize_cards(bulk_data)

    # Sort oracle_ids by number of associated cards (descending)
    sorted_oracle_ids = sorted(scryfall_cards_by_oracle_id.items(), key=lambda x: len(x[1]), reverse=True)
    # Sort illustration_ids by number of associated cards (descending)
    sorted_illustration_ids = sorted(scryfall_cards_by_illustration_id.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"{len(sorted_oracle_ids)} unique oracle_ids and {len(sorted_illustration_ids)} unique illustration_ids found.")
    
    top_n = 1000
    print(f"Syncing images for top {top_n} illustration_ids...")
    top_illustration_cards = []
    for illustration_id, cards in sorted_illustration_ids[:top_n]:
        top_illustration_cards.extend(cards)

    # Dump top cards to a JSON file for reference
    TOP_CARDS_PATH = os.path.join(CACHE_DIR, f"by_illustration_top_{top_n}.json")
    with open(TOP_CARDS_PATH, "w", encoding="utf-8") as f:
        json.dump(top_illustration_cards, f, ensure_ascii=False, indent=2)

    sync_scryfall_images(top_illustration_cards)
