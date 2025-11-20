"""
Pokemon TCG API fetcher
API documentation: https://docs.pokemontcg.io/
"""

import json
import os
import time
from typing import Dict, List, Optional
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


class PokemonFetcher:
    """Fetches Pokemon TCG card data and images from pokemontcg.io API"""

    BASE_URL = "https://api.pokemontcg.io/v2"
    RATE_LIMIT_DELAY = 0.1  # Be respectful to the API

    def __init__(self, data_dir: str = "data", api_key: Optional[str] = None):
        """
        Initialize the Pokemon TCG fetcher

        Args:
            data_dir: Base directory for storing data
            api_key: Optional API key for higher rate limits
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images" / "pokemon"
        self.metadata_dir = self.data_dir / "raw" / "pokemon"
        self.api_key = api_key
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup headers
        self.headers = {}
        if api_key:
            self.headers["X-Api-Key"] = api_key

    def fetch_all_cards(
        self, max_cards: Optional[int] = None, page_size: int = 250
    ) -> List[Dict]:
        """
        Fetch all Pokemon cards

        Args:
            max_cards: Maximum number of cards to fetch (None for all)
            page_size: Number of cards per page (max 250)

        Returns:
            List of card metadata dictionaries
        """
        print("Fetching cards from Pokemon TCG API...")
        all_cards = []
        page = 1
        
        while True:
            time.sleep(self.RATE_LIMIT_DELAY)
            url = f"{self.BASE_URL}/cards"
            params = {"page": page, "pageSize": page_size}
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                cards = data.get("data", [])
                if not cards:
                    break
                
                all_cards.extend(cards)
                print(f"Fetched {len(all_cards)} cards...")
                
                if max_cards and len(all_cards) >= max_cards:
                    all_cards = all_cards[:max_cards]
                    break
                
                # Check if there are more pages
                total_count = data.get("totalCount", 0)
                if len(all_cards) >= total_count:
                    break
                
                page += 1
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break
        
        print(f"Total cards fetched: {len(all_cards)}")
        return all_cards

    def download_card_images(
        self, cards: List[Dict], image_quality: str = "large"
    ) -> None:
        """
        Download card images

        Args:
            cards: List of card metadata dictionaries
            image_quality: Image quality (small, large)
        """
        print(f"Downloading {len(cards)} card images...")
        
        for card in tqdm(cards, desc="Downloading images"):
            card_id = card.get("id")
            if not card_id:
                continue
            
            # Get image URL
            images = card.get("images", {})
            image_url = images.get(image_quality)
            if not image_url:
                # Fallback to small if large not available
                image_url = images.get("small")
            
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

    def fetch_specific_set(self, set_id: str) -> List[Dict]:
        """
        Fetch all cards from a specific set

        Args:
            set_id: Set ID (e.g., "base1" for Base Set)

        Returns:
            List of card metadata dictionaries
        """
        print(f"Fetching cards from set: {set_id}")
        url = f"{self.BASE_URL}/cards"
        params = {"q": f"set.id:{set_id}"}
        
        all_cards = []
        page = 1
        
        while True:
            params["page"] = page
            time.sleep(self.RATE_LIMIT_DELAY)
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                cards = data.get("data", [])
                if not cards:
                    break
                
                all_cards.extend(cards)
                
                # Check if there are more pages
                total_count = data.get("totalCount", 0)
                if len(all_cards) >= total_count:
                    break
                
                page += 1
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break
        
        print(f"Fetched {len(all_cards)} cards from set {set_id}")
        return all_cards

    def fetch_sample_dataset(self, num_cards: int = 1000) -> List[Dict]:
        """
        Fetch a sample dataset of cards

        Args:
            num_cards: Number of cards to fetch

        Returns:
            List of card metadata dictionaries
        """
        print(f"Fetching sample dataset of {num_cards} cards...")
        url = f"{self.BASE_URL}/cards"
        params = {"pageSize": min(num_cards, 250)}
        
        all_cards = []
        page = 1
        
        while len(all_cards) < num_cards:
            params["page"] = page
            time.sleep(self.RATE_LIMIT_DELAY)
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                cards = data.get("data", [])
                if not cards:
                    break
                
                all_cards.extend(cards)
                
                if len(all_cards) >= num_cards:
                    all_cards = all_cards[:num_cards]
                    break
                
                page += 1
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break
        
        print(f"Fetched {len(all_cards)} cards")
        return all_cards
