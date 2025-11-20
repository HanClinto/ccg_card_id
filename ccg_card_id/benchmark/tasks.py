"""
MIEB (Multimodal Image Embedding Benchmark) tasks for collectible card recognition
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..dataset.dataset_manager import DatasetManager
from ..utils.metrics import (
    compute_accuracy,
    compute_mean_average_precision,
    compute_recall_at_k,
)


class CardMatchingTask:
    """
    Task: Given two card images, determine if they are the same card
    This tests the model's ability to match identical cards across different conditions
    (e.g., different lighting, angles, or image quality)
    """

    def __init__(self, dataset_manager: DatasetManager, game: str):
        """
        Initialize the matching task

        Args:
            dataset_manager: DatasetManager instance
            game: Game name (e.g., "mtg", "pokemon")
        """
        self.dataset_manager = dataset_manager
        self.game = game

    def create_pairs(
        self,
        split: str = "test",
        num_positive: int = 100,
        num_negative: int = 100,
        seed: int = 42,
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Create positive and negative pairs for matching

        Args:
            split: Data split to use
            num_positive: Number of positive pairs (same card)
            num_negative: Number of negative pairs (different cards)
            seed: Random seed

        Returns:
            Tuple of (pairs, labels) where pairs are (card_id1, card_id2) and labels are 0/1
        """
        np.random.seed(seed)
        cards = self.dataset_manager.load_split(self.game, split)
        
        pairs = []
        labels = []
        
        # Create positive pairs (same card)
        # In practice, we'd need multiple images of the same card
        # For now, we'll use the same image (simplified)
        card_ids = [card.get("id") for card in cards if card.get("id")]
        selected_ids = np.random.choice(card_ids, size=min(num_positive, len(card_ids)), replace=False)
        
        for card_id in selected_ids:
            pairs.append((card_id, card_id))
            labels.append(1)
        
        # Create negative pairs (different cards)
        for _ in range(num_negative):
            id1, id2 = np.random.choice(card_ids, size=2, replace=False)
            pairs.append((id1, id2))
            labels.append(0)
        
        return pairs, labels

    def evaluate(self, similarities: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate matching performance

        Args:
            similarities: Similarity scores for each pair
            labels: Ground truth labels (0/1)

        Returns:
            Dictionary of metrics
        """
        # Find optimal threshold
        thresholds = np.linspace(similarities.min(), similarities.max(), 100)
        best_acc = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (similarities > threshold).astype(int)
            acc = compute_accuracy(predictions, labels)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return {
            "accuracy": best_acc,
            "threshold": best_threshold,
        }


class CardRetrievalTask:
    """
    Task: Given a query card image, retrieve the most similar cards from a gallery
    This tests the model's ability to rank cards by similarity
    """

    def __init__(self, dataset_manager: DatasetManager, game: str):
        """
        Initialize the retrieval task

        Args:
            dataset_manager: DatasetManager instance
            game: Game name
        """
        self.dataset_manager = dataset_manager
        self.game = game

    def create_query_gallery(
        self,
        query_split: str = "test",
        gallery_split: str = "train",
        num_queries: int = 100,
        seed: int = 42,
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Create query and gallery sets

        Args:
            query_split: Split to use for queries
            gallery_split: Split to use for gallery
            num_queries: Number of query images
            seed: Random seed

        Returns:
            Tuple of (query_ids, gallery_ids, target_indices)
            target_indices[i] is the index in gallery_ids of the correct match for query i
        """
        np.random.seed(seed)
        
        query_cards = self.dataset_manager.load_split(self.game, query_split)
        gallery_cards = self.dataset_manager.load_split(self.game, gallery_split)
        
        query_ids = [card.get("id") for card in query_cards if card.get("id")]
        gallery_ids = [card.get("id") for card in gallery_cards if card.get("id")]
        
        # Select random queries
        query_ids = list(np.random.choice(query_ids, size=min(num_queries, len(query_ids)), replace=False))
        
        # For each query, find its match in the gallery
        # In a real scenario, we'd have duplicates or versions
        # For now, we'll add each query to the gallery at a random position
        target_indices = []
        for query_id in query_ids:
            if query_id not in gallery_ids:
                gallery_ids.append(query_id)
                target_indices.append(len(gallery_ids) - 1)
            else:
                target_indices.append(gallery_ids.index(query_id))
        
        return query_ids, gallery_ids, target_indices

    def evaluate(
        self,
        similarity_matrix: np.ndarray,
        target_indices: np.ndarray,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict:
        """
        Evaluate retrieval performance

        Args:
            similarity_matrix: Similarity scores (n_queries x n_gallery)
            target_indices: Ground truth indices for each query
            k_values: List of k values for Recall@K

        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute Recall@K for each k
        for k in k_values:
            recall = compute_recall_at_k(similarity_matrix, target_indices, k=k)
            metrics[f"recall@{k}"] = recall
        
        # Compute mAP
        relevant_indices = [[idx] for idx in target_indices]
        map_score = compute_mean_average_precision(similarity_matrix, relevant_indices)
        metrics["mAP"] = map_score
        
        return metrics


class CardClassificationTask:
    """
    Task: Given a card image, classify it into predefined categories
    (e.g., card type, rarity, set, etc.)
    """

    def __init__(self, dataset_manager: DatasetManager, game: str, category: str = "rarity"):
        """
        Initialize the classification task

        Args:
            dataset_manager: DatasetManager instance
            game: Game name
            category: Category to classify (e.g., "rarity", "type")
        """
        self.dataset_manager = dataset_manager
        self.game = game
        self.category = category

    def get_labels(self, split: str = "train") -> Tuple[List[str], List[str]]:
        """
        Get card IDs and their labels

        Args:
            split: Data split to use

        Returns:
            Tuple of (card_ids, labels)
        """
        cards = self.dataset_manager.load_split(self.game, split)
        
        card_ids = []
        labels = []
        
        for card in cards:
            card_id = card.get("id")
            if not card_id:
                continue
            
            # Get label based on category
            if self.game == "mtg":
                if self.category == "rarity":
                    label = card.get("rarity", "unknown")
                elif self.category == "type":
                    label = card.get("type_line", "unknown")
                else:
                    label = "unknown"
            elif self.game == "pokemon":
                if self.category == "rarity":
                    label = card.get("rarity", "unknown")
                elif self.category == "type":
                    types = card.get("types", [])
                    label = types[0] if types else "unknown"
                else:
                    label = "unknown"
            else:
                label = "unknown"
            
            card_ids.append(card_id)
            labels.append(label)
        
        return card_ids, labels

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Evaluate classification performance

        Args:
            predictions: Predicted labels
            targets: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        accuracy = compute_accuracy(predictions, targets)
        
        return {
            "accuracy": accuracy,
        }
