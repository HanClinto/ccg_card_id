"""
Evaluation metrics for card recognition tasks
"""

from typing import List, Tuple

import numpy as np
import torch


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy

    Args:
        predictions: Predicted labels
        targets: Ground truth labels

    Returns:
        Accuracy score (0-1)
    """
    return np.mean(predictions == targets)


def compute_recall_at_k(
    similarities: np.ndarray,
    targets: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute Recall@K for retrieval tasks

    Args:
        similarities: Matrix of similarity scores (n_queries x n_gallery)
        targets: Ground truth indices for each query
        k: Number of top results to consider

    Returns:
        Recall@K score (0-1)
    """
    # Get top-k indices for each query
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    
    # Check if target is in top-k for each query
    hits = 0
    for i, target_idx in enumerate(targets):
        if target_idx in top_k_indices[i]:
            hits += 1
    
    return hits / len(targets)


def compute_mean_average_precision(
    similarities: np.ndarray,
    relevant_indices: List[List[int]],
) -> float:
    """
    Compute Mean Average Precision (mAP)

    Args:
        similarities: Matrix of similarity scores (n_queries x n_gallery)
        relevant_indices: List of lists containing relevant indices for each query

    Returns:
        mAP score (0-1)
    """
    aps = []
    
    for i, relevant in enumerate(relevant_indices):
        if len(relevant) == 0:
            continue
        
        # Sort gallery by similarity
        sorted_indices = np.argsort(-similarities[i])
        
        # Calculate average precision
        hits = 0
        precisions = []
        
        for rank, idx in enumerate(sorted_indices):
            if idx in relevant:
                hits += 1
                precision = hits / (rank + 1)
                precisions.append(precision)
        
        if precisions:
            ap = np.mean(precisions)
            aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def compute_cosine_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between two sets of embeddings

    Args:
        embeddings1: First set of embeddings (n1 x d)
        embeddings2: Second set of embeddings (n2 x d)

    Returns:
        Similarity matrix (n1 x n2)
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    embeddings1_norm = embeddings1 / (norm1 + 1e-8)
    embeddings2_norm = embeddings2 / (norm2 + 1e-8)
    
    # Compute cosine similarity
    return np.dot(embeddings1_norm, embeddings2_norm.T)


def compute_euclidean_distance(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance between two sets of embeddings

    Args:
        embeddings1: First set of embeddings (n1 x d)
        embeddings2: Second set of embeddings (n2 x d)

    Returns:
        Distance matrix (n1 x n2)
    """
    # Compute squared L2 distance
    dist = np.sum(embeddings1**2, axis=1, keepdims=True) + \
           np.sum(embeddings2**2, axis=1) - \
           2 * np.dot(embeddings1, embeddings2.T)
    
    # Take square root and ensure non-negative
    dist = np.sqrt(np.maximum(dist, 0))
    
    return dist
