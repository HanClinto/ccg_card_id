#!/usr/bin/env python3
"""
Script to download card datasets from Scryfall and Pokemon TCG APIs
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccg_card_id.dataset import ScryfallFetcher, PokemonFetcher, DatasetManager


def download_mtg_dataset(data_dir: str, num_cards: int = 1000):
    """Download Magic: The Gathering dataset"""
    print("=" * 60)
    print("Downloading Magic: The Gathering dataset")
    print("=" * 60)
    
    fetcher = ScryfallFetcher(data_dir=data_dir)
    
    # Fetch sample dataset
    cards = fetcher.fetch_sample_dataset(num_cards=num_cards)
    
    # Save metadata
    fetcher.save_metadata(cards)
    
    # Download images
    fetcher.download_card_images(cards)
    
    print(f"\nDownloaded {len(cards)} Magic: The Gathering cards")


def download_pokemon_dataset(data_dir: str, num_cards: int = 1000, api_key: str = None):
    """Download Pokemon TCG dataset"""
    print("=" * 60)
    print("Downloading Pokemon TCG dataset")
    print("=" * 60)
    
    fetcher = PokemonFetcher(data_dir=data_dir, api_key=api_key)
    
    # Fetch sample dataset
    cards = fetcher.fetch_sample_dataset(num_cards=num_cards)
    
    # Save metadata
    fetcher.save_metadata(cards)
    
    # Download images
    fetcher.download_card_images(cards)
    
    print(f"\nDownloaded {len(cards)} Pokemon TCG cards")


def create_splits(data_dir: str, game: str):
    """Create train/val/test splits"""
    print(f"\nCreating data splits for {game}...")
    
    dataset_manager = DatasetManager(data_dir=data_dir)
    dataset_manager.create_splits(game=game)


def main():
    parser = argparse.ArgumentParser(
        description="Download card datasets from various APIs"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["mtg", "pokemon", "both"],
        default="both",
        help="Which game to download (default: both)",
    )
    parser.add_argument(
        "--num-cards",
        type=int,
        default=1000,
        help="Number of cards to download per game (default: 1000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store data (default: data)",
    )
    parser.add_argument(
        "--pokemon-api-key",
        type=str,
        default=None,
        help="Pokemon TCG API key (optional, for higher rate limits)",
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/val/test splits after downloading",
    )
    
    args = parser.parse_args()
    
    # Download datasets
    if args.game in ["mtg", "both"]:
        download_mtg_dataset(args.data_dir, args.num_cards)
        if args.create_splits:
            create_splits(args.data_dir, "mtg")
    
    if args.game in ["pokemon", "both"]:
        download_pokemon_dataset(args.data_dir, args.num_cards, args.pokemon_api_key)
        if args.create_splits:
            create_splits(args.data_dir, "pokemon")
    
    print("\nDataset download complete!")


if __name__ == "__main__":
    main()
