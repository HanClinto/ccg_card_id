"""
Dataset manager for organizing and accessing collectible card datasets
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class DatasetManager:
    """Manages card datasets including train/val/test splits"""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset manager

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.images_dir = self.data_dir / "images"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self, game: str) -> List[Dict]:
        """
        Load card metadata for a specific game

        Args:
            game: Game name (e.g., "mtg", "pokemon")

        Returns:
            List of card metadata dictionaries
        """
        metadata_path = self.raw_dir / game / "cards.json"
        if not metadata_path.exists():
            return []
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_splits(
        self,
        game: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create train/validation/test splits

        Args:
            game: Game name
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Load metadata
        cards = self.load_metadata(game)
        if not cards:
            return [], [], []
        
        # Shuffle with fixed seed
        np.random.seed(seed)
        indices = np.random.permutation(len(cards))
        
        # Calculate split sizes
        n_train = int(len(cards) * train_ratio)
        n_val = int(len(cards) * val_ratio)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Create splits
        train_data = [cards[i] for i in train_indices]
        val_data = [cards[i] for i in val_indices]
        test_data = [cards[i] for i in test_indices]
        
        # Save splits
        splits_dir = self.processed_dir / game / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            split_path = splits_dir / f"{split_name}.json"
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created splits for {game}:")
        print(f"  Train: {len(train_data)} cards")
        print(f"  Val: {len(val_data)} cards")
        print(f"  Test: {len(test_data)} cards")
        
        return train_data, val_data, test_data

    def load_split(self, game: str, split: str) -> List[Dict]:
        """
        Load a specific data split

        Args:
            game: Game name
            split: Split name ("train", "val", or "test")

        Returns:
            List of card metadata dictionaries
        """
        split_path = self.processed_dir / game / "splits" / f"{split}.json"
        if not split_path.exists():
            return []
        
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_image_path(self, game: str, card_id: str) -> Optional[Path]:
        """
        Get the path to a card image

        Args:
            game: Game name
            card_id: Card ID

        Returns:
            Path to the image file, or None if not found
        """
        image_path = self.images_dir / game / f"{card_id}.jpg"
        if image_path.exists():
            return image_path
        
        # Try PNG as well
        image_path = self.images_dir / game / f"{card_id}.png"
        if image_path.exists():
            return image_path
        
        return None

    def load_image(self, game: str, card_id: str) -> Optional[Image.Image]:
        """
        Load a card image

        Args:
            game: Game name
            card_id: Card ID

        Returns:
            PIL Image object, or None if not found
        """
        image_path = self.get_image_path(game, card_id)
        if image_path is None:
            return None
        
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_statistics(self, game: str) -> Dict:
        """
        Get dataset statistics

        Args:
            game: Game name

        Returns:
            Dictionary of statistics
        """
        cards = self.load_metadata(game)
        
        stats = {
            "total_cards": len(cards),
            "game": game,
        }
        
        # Count images
        images_dir = self.images_dir / game
        if images_dir.exists():
            image_files = list(images_dir.glob("*"))
            stats["images_downloaded"] = len(image_files)
        else:
            stats["images_downloaded"] = 0
        
        return stats
