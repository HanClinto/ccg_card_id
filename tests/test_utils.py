"""Tests for utility functions"""

import numpy as np
import pytest
from PIL import Image

from ccg_card_id.utils.image_utils import (
    preprocess_image,
    resize_image_keeping_aspect,
)
from ccg_card_id.utils.metrics import (
    compute_accuracy,
    compute_cosine_similarity,
    compute_recall_at_k,
)


def test_preprocess_image():
    """Test image preprocessing"""
    # Create a test image
    img = Image.new("RGB", (100, 100), color="red")
    
    # Preprocess without normalization
    tensor = preprocess_image(img, size=224, normalize=False)
    assert tensor.shape == (3, 224, 224)
    
    # Preprocess with normalization
    tensor_norm = preprocess_image(img, size=224, normalize=True)
    assert tensor_norm.shape == (3, 224, 224)


def test_resize_image_keeping_aspect():
    """Test aspect ratio preserving resize"""
    # Create a non-square test image
    img = Image.new("RGB", (100, 200), color="blue")
    
    # Resize to square
    resized = resize_image_keeping_aspect(img, target_size=224)
    assert resized.size == (224, 224)


def test_compute_accuracy():
    """Test accuracy computation"""
    predictions = np.array([1, 2, 3, 1, 2])
    targets = np.array([1, 2, 3, 1, 3])
    
    accuracy = compute_accuracy(predictions, targets)
    assert accuracy == 0.8  # 4 out of 5 correct


def test_compute_cosine_similarity():
    """Test cosine similarity computation"""
    embeddings1 = np.array([[1, 0, 0], [0, 1, 0]])
    embeddings2 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    assert similarities.shape == (2, 3)
    # First embedding should be most similar to first gallery item
    assert similarities[0, 0] > similarities[0, 1]
    # Second embedding should be most similar to third gallery item
    assert similarities[1, 2] > similarities[1, 0]


def test_compute_recall_at_k():
    """Test Recall@K computation"""
    # Create a simple similarity matrix
    similarities = np.array([
        [0.9, 0.7, 0.5, 0.3],  # Query 0: target at index 0
        [0.3, 0.9, 0.7, 0.5],  # Query 1: target at index 1
        [0.5, 0.3, 0.9, 0.7],  # Query 2: target at index 2
    ])
    targets = np.array([0, 1, 2])
    
    # All targets should be in top-1
    recall_at_1 = compute_recall_at_k(similarities, targets, k=1)
    assert recall_at_1 == 1.0
    
    # All targets should be in top-3
    recall_at_3 = compute_recall_at_k(similarities, targets, k=3)
    assert recall_at_3 == 1.0
