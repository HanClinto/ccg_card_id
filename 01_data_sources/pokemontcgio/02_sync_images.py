# Sync Pokemon TCG card images
# Given a bulk data JSON from Pokemon TCG API exists in ./cache/all_cards.json
# Download images for all cards listed in that file

import os
import requests
import json
from datetime import datetime
import datetime as dt
from tqdm import tqdm

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
BULK_DATA_PATH = os.path.join(CACHE_DIR, "all_cards.json")
IMAGES_DIR = os.path.join(CACHE_DIR, "images")

# Image quality options: small, large
image_quality = "large"

def load_pokemon_data():
    """
    Loads the Pokemon TCG data from local cache.
    """
    then = datetime.now()
    print(f"Loading Pokemon TCG data from {BULK_DATA_PATH}...")
    with open(BULK_DATA_PATH, "r", encoding="utf-8") as f:
        pokemon_data = json.load(f)
    print(f"Loaded {len(pokemon_data)} cards in {(datetime.now() - then).total_seconds():.2f} seconds.")
    return pokemon_data

def sync_pokemon_images(cards):
    """
    Syncs Pokemon TCG card images based on the local all_cards.json file.
    Downloads images that are missing.
    """
    then = datetime.now()
    print(f"Syncing images for {len(cards)} cards...")
    
    # Ensure images directory exists
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Display progress using tqdm
    with tqdm(cards, desc="Syncing images", unit="card") as pbar:
        for card in pbar:
            card_id = card["id"]
            pbar.set_postfix({"card_id": card_id})
            
            # Get image URLs
            images = card.get("images", {})
            image_url = images.get(image_quality) or images.get("large") or images.get("small")
            
            if not image_url:
                continue  # No image available
            
            # Determine local image path
            subdir1 = card_id[0]
            subdir2 = card_id[1] if len(card_id) > 1 else "0"
            local_dir = os.path.join(IMAGES_DIR, image_quality, subdir1, subdir2)
            os.makedirs(local_dir, exist_ok=True)
            
            # Determine file extension from URL
            image_extension = image_url.split(".")[-1].split("?")[0]
            local_image_path = os.path.join(local_dir, f"{card_id}.{image_extension}")
            
            # Download image if it doesn't exist
            if not os.path.exists(local_image_path):
                try:
                    img_resp = requests.get(image_url, timeout=30)
                    img_resp.raise_for_status()
                    with open(local_image_path, "wb") as img_file:
                        img_file.write(img_resp.content)
                except Exception as e:
                    pbar.write(f"Failed to download {card_id}: {e}")

if __name__ == "__main__":
    pokemon_data = load_pokemon_data()
    sync_pokemon_images(pokemon_data)
    print("Image sync complete!")
