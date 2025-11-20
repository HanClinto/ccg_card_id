# Prioritize downloading images for cards that are most ambiguous.
# For each card with images (image_status is not "missing" nor "placeholder") in all_cards.json:
#  Append each card to a dictionary scryfall_cards_by_oracle_id
#  Append each card to a dictionary scryfall_cards_by_illustration_id (if illustration_id exists)


import os
import requests
# import orjson
import json
from datetime import datetime
import datetime as dt
from tqdm import tqdm

image_quality = "png"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
BULK_DATA_PATH = os.path.join(CACHE_DIR, "all_cards.json")
IMAGES_DIR = os.path.join(CACHE_DIR, "images", image_quality)

english_only = True

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

if __name__ == "__main__":
    bulk_data = load_bulk_data()

    scryfall_cards_by_oracle_id, scryfall_cards_by_illustration_id = prioritize_cards(bulk_data)

    # Sort oracle_ids by number of associated cards (descending)
    sorted_oracle_ids = sorted(scryfall_cards_by_oracle_id.items(), key=lambda x: len(x[1]), reverse=True)
    # Sort illustration_ids by number of associated cards (descending)
    sorted_illustration_ids = sorted(scryfall_cards_by_illustration_id.items(), key=lambda x: len(x[1]), reverse=True)

    num_cards = 10
    print(f"Top {num_cards} most ambiguous oracle_ids:")
    for oracle_id, cards in sorted_oracle_ids[:num_cards]:
        example_card = cards[0]
        print(f"  {example_card.get('name')}: {len(cards)} cards")
        print(f"    [ https://scryfall.com/search?q=oracleid%3A{oracle_id}&unique=prints&as=grid ]")

    print()
    print(f"Top {num_cards} most ambiguous illustration_ids:")
    for illustration_id, cards in sorted_illustration_ids[:num_cards]:
        example_card = cards[0]
        print(f"  {example_card.get('name')}: {len(cards)} cards")
        print(f"    [ https://scryfall.com/search?q=illustrationid%3A{illustration_id}&unique=prints&as=grid ]")
