"""Tests for model implementations"""

import numpy as np
import pytest
from PIL import Image

from ccg_card_id.models import PHashModel


def create_test_image(color="red", size=(224, 224)):
    """Helper function to create test images"""
    return Image.new("RGB", size, color=color)


def test_phash_model_init():
    """Test pHash model initialization"""
    model = PHashModel(hash_size=8)
    assert model.hash_size == 8
    assert len(model.gallery_hashes) == 0
    assert len(model.gallery_ids) == 0


def test_phash_compute_hash():
    """Test hash computation"""
    model = PHashModel(hash_size=8)
    img = create_test_image()
    
    hash1 = model.compute_hash(img)
    hash2 = model.compute_hash(img)
    
    # Same image should produce same hash
    assert hash1 == hash2


def test_phash_build_gallery():
    """Test gallery building"""
    model = PHashModel()
    
    colors = ["red", "green", "blue", "yellow", "white"]
    images = [create_test_image(color) for color in colors]
    ids = [f"card_{i}" for i in range(5)]
    
    model.build_gallery(images, ids)
    
    assert len(model.gallery_hashes) == 5
    assert len(model.gallery_ids) == 5
    assert model.gallery_ids == ids


def test_phash_find_matches():
    """Test finding matches"""
    model = PHashModel()
    
    # Create gallery
    gallery_images = [
        create_test_image("red"),
        create_test_image("green"),
        create_test_image("blue"),
    ]
    gallery_ids = ["red_card", "green_card", "blue_card"]
    model.build_gallery(gallery_images, gallery_ids)
    
    # Query with red image
    query = create_test_image("red")
    matches = model.find_matches(query, top_k=3)
    
    assert len(matches) == 3
    # First match should be the red card (distance 0)
    assert matches[0][0] == "red_card"
    assert matches[0][1] == 0


def test_phash_similarity_matrix():
    """Test similarity matrix computation"""
    model = PHashModel()
    
    # Create gallery
    gallery_colors = ["red", "green", "blue", "yellow", "white"]
    gallery_images = [create_test_image(color) for color in gallery_colors]
    gallery_ids = [f"card_{i}" for i in range(5)]
    model.build_gallery(gallery_images, gallery_ids)
    
    # Create queries
    query_colors = ["red", "green", "blue"]
    query_images = [create_test_image(color) for color in query_colors]
    
    # Compute similarity matrix
    sim_matrix = model.compute_similarity_matrix(query_images)
    
    assert sim_matrix.shape == (3, 5)


def test_phash_clear_gallery():
    """Test clearing gallery"""
    model = PHashModel()
    
    images = [create_test_image() for _ in range(3)]
    ids = ["card_1", "card_2", "card_3"]
    model.build_gallery(images, ids)
    
    model.clear_gallery()
    
    assert len(model.gallery_hashes) == 0
    assert len(model.gallery_ids) == 0
