# Pull down the latest version of Pokemon TCG data
# API documentation: https://docs.pokemontcg.io/

import os
import requests
import json
from datetime import datetime
import datetime as dt
from tqdm import tqdm

BASE_URL = "https://api.pokemontcg.io/v2"
RATE_LIMIT_DELAY = 0.1  # Be respectful to the API

def load_pokemon_data():
    """
    Loads the Pokemon TCG data from local cache.
    Returns the list of cards if cache exists, otherwise empty list.
    """
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
    CACHE_PATH = os.path.join(CACHE_DIR, "all_cards.json")
    
    if not os.path.exists(CACHE_PATH):
        return []
    
    print(f"Loading Pokemon TCG data from {CACHE_PATH}...")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        pokemon_data = json.load(f)
    print(f"Loaded {len(pokemon_data)} cards.")
    return pokemon_data

def sync_pokemon_data(api_key=None, page_size=250):
    """
    Fetches all Pokemon TCG cards from the API and saves to local cache.
    Returns True if new data was downloaded, False if using cached data.
    """
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
    CACHE_PATH = os.path.join(CACHE_DIR, "all_cards.json")
    
    # Setup headers
    headers = {}
    if api_key:
        headers["X-Api-Key"] = api_key
    
    # Check if we should download new data
    # For Pokemon TCG API, we'll re-download if cache is older than 7 days
    need_download = False
    if not os.path.exists(CACHE_PATH):
        print(f"Local cache not found: {CACHE_PATH}")
        need_download = True
    else:
        local_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH), tz=dt.UTC)
        age_days = (datetime.now(dt.UTC) - local_mtime).days
        print(f"Local cache age: {age_days} days")
        if age_days > 7:
            need_download = True
    
    if need_download:
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        print("Downloading Pokemon TCG data from API...")
        all_cards = []
        page = 1
        
        with tqdm(desc="Downloading cards", unit="page") as pbar:
            while True:
                import time
                time.sleep(RATE_LIMIT_DELAY)
                
                url = f"{BASE_URL}/cards"
                params = {"page": page, "pageSize": page_size}
                
                try:
                    response = requests.get(url, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    cards = data.get("data", [])
                    if not cards:
                        break
                    
                    all_cards.extend(cards)
                    pbar.update(1)
                    pbar.set_postfix({"total_cards": len(all_cards)})
                    
                    # Check if there are more pages
                    total_count = data.get("totalCount", 0)
                    if len(all_cards) >= total_count:
                        break
                    
                    page += 1
                    
                except Exception as e:
                    print(f"Error fetching page {page}: {e}")
                    break
        
        print(f"Downloaded {len(all_cards)} cards")
        
        # Save to cache
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(all_cards, f, indent=2, ensure_ascii=False)
        
        return True
    else:
        print(f"Local cache is up to date: {CACHE_PATH}")
        return False

if __name__ == "__main__":
    api_key = os.environ.get("POKEMON_TCG_API_KEY")  # Optional
    changed = sync_pokemon_data(api_key=api_key)
    print(f"Data updated: {changed}")
    
    # Load and print stats
    cards = load_pokemon_data()
    if cards:
        print(f"Total cards in cache: {len(cards)}")
