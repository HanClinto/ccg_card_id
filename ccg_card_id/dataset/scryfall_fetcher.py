"""
Scryfall API fetcher for Magic: The Gathering cards
API documentation: https://scryfall.com/docs/api
"""

import json
import os
import time
from typing import Dict, List, Optional
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


class ScryfallFetcher:
    """Fetches Magic: The Gathering card data and images from Scryfall API"""

    BASE_URL = "https://api.scryfall.com"
    RATE_LIMIT_DELAY = 0.1  # Scryfall asks for 50-100ms between requests

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Scryfall fetcher

        Args:
            data_dir: Base directory for storing data
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / "mtg"
        self.metadata_dir = self.data_dir / "raw" / "mtg"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all_cards(self, max_cards: Optional[int] = None) -> List[Dict]:
        """
        Fetch all Magic cards from Scryfall

        Args:
            max_cards: Maximum number of cards to fetch (None for all)

        Returns:
            List of card metadata dictionaries
        """
        print("Fetching cards from Scryfall...")
        all_cards = []
        url = f"{self.BASE_URL}/cards"
        
        params = {"page": 1}
        
        while url:
            time.sleep(self.RATE_LIMIT_DELAY)
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            cards = data.get("data", [])
            all_cards.extend(cards)
            
            print(f"Fetched {len(all_cards)} cards...")
            
            if max_cards and len(all_cards) >= max_cards:
                all_cards = all_cards[:max_cards]
                break
            
            # Check if there's a next page
            if data.get("has_more"):
                url = data.get("next_page")
                params = {}  # URL already includes params
            else:
                url = None
        
        print(f"Total cards fetched: {len(all_cards)}")
        return all_cards

    def download_card_images(
        self, cards: List[Dict], image_quality: str = "normal"
    ) -> None:
        """
        Download card images

        Args:
            cards: List of card metadata dictionaries
            image_quality: Image quality (small, normal, large, png, art_crop, border_crop)
        """
        print(f"Downloading {len(cards)} card images...")
        
        for card in tqdm(cards, desc="Downloading images"):
            card_id = card.get("id")
            if not card_id:
                continue
            
            # Get image URL
            image_uris = card.get("image_uris")
            if not image_uris:
                # Handle double-faced cards
                card_faces = card.get("card_faces", [])
                if card_faces and "image_uris" in card_faces[0]:
                    image_uris = card_faces[0]["image_uris"]
                else:
                    continue
            
            image_url = image_uris.get(image_quality)
            if not image_url:
                continue
            
            # Download and save image
            image_path = self.images_dir / f"{card_id}.jpg"
            if image_path.exists():
                continue
            
            try:
                time.sleep(self.RATE_LIMIT_DELAY)
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Save image
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
            except Exception as e:
                print(f"Failed to download {card_id}: {e}")
                continue

    def save_metadata(self, cards: List[Dict], filename: str = "cards.json") -> None:
        """
        Save card metadata to JSON file

        Args:
            cards: List of card metadata dictionaries
            filename: Name of the output file
        """
        output_path = self.metadata_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {output_path}")

    def fetch_specific_set(self, set_code: str) -> List[Dict]:
        """
        Fetch all cards from a specific set

        Args:
            set_code: Three-letter set code (e.g., "mid" for Midnight Hunt)

        Returns:
            List of card metadata dictionaries
        """
        print(f"Fetching cards from set: {set_code}")
        url = f"{self.BASE_URL}/cards/search"
        params = {"q": f"set:{set_code}", "order": "set"}
        
        all_cards = []
        
        while url:
            time.sleep(self.RATE_LIMIT_DELAY)
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            cards = data.get("data", [])
            all_cards.extend(cards)
            
            if data.get("has_more"):
                url = data.get("next_page")
                params = {}
            else:
                url = None
        
        print(f"Fetched {len(all_cards)} cards from set {set_code}")
        return all_cards

    def fetch_sample_dataset(self, num_cards: int = 1000) -> List[Dict]:
        """
        Fetch a sample dataset of popular/recent cards

        Args:
            num_cards: Number of cards to fetch

        Returns:
            List of card metadata dictionaries
        """
        print(f"Fetching sample dataset of {num_cards} cards...")
        url = f"{self.BASE_URL}/cards/search"
        # Query for cards with images, sorted by popularity
        params = {"q": "has:image", "order": "edhrec", "unique": "prints"}
        
        all_cards = []
        
        while url and len(all_cards) < num_cards:
            time.sleep(self.RATE_LIMIT_DELAY)
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            cards = data.get("data", [])
            all_cards.extend(cards)
            
            if data.get("has_more") and len(all_cards) < num_cards:
                url = data.get("next_page")
                params = {}
            else:
                break
        
        all_cards = all_cards[:num_cards]
        print(f"Fetched {len(all_cards)} cards")
        return all_cards
