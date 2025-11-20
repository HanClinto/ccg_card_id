# Build pHash vectors for card images
# Reads card images from data sources and generates perceptual hash vectors

import os
import imagehash
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

HASH_SIZE = 8  # 8x8 = 64 bit hash

def compute_phash(image_path, hash_size=HASH_SIZE):
    """
    Compute perceptual hash for an image.
    Returns the hash as a hex string.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        phash = imagehash.phash(img, hash_size=hash_size)
        return str(phash)
    except Exception as e:
        print(f"Error computing hash for {image_path}: {e}")
        return None

def build_phash_vectors(image_dir, output_file):
    """
    Build pHash vectors for all images in a directory.
    Saves results as JSON with card_id -> hash mapping.
    """
    vectors = {}
    
    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images")
    
    # Compute hashes
    with tqdm(image_files, desc="Computing pHashes") as pbar:
        for image_path in pbar:
            # Extract card ID from filename
            card_id = os.path.splitext(os.path.basename(image_path))[0]
            pbar.set_postfix({"card_id": card_id})
            
            phash = compute_phash(image_path)
            if phash:
                vectors[card_id] = phash
    
    # Save vectors
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(vectors, f, indent=2)
    
    print(f"Saved {len(vectors)} pHash vectors to {output_file}")
    return vectors

if __name__ == "__main__":
    # Example: Build vectors for Scryfall images
    scryfall_images = "../../01_data_sources/scryfall/cache/images/png/front"
    output_file = "./cache/scryfall_phash_vectors.json"
    
    if os.path.exists(scryfall_images):
        print("Building pHash vectors for Scryfall images...")
        vectors = build_phash_vectors(scryfall_images, output_file)
    else:
        print(f"Image directory not found: {scryfall_images}")
        print("Run Scryfall sync scripts first!")
