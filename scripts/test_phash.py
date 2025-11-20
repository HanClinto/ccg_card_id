#!/usr/bin/env python3
"""
Script to test pHash baseline on card recognition tasks
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccg_card_id.benchmark import CardRetrievalTask
from ccg_card_id.dataset import DatasetManager
from ccg_card_id.models import PHashModel


def test_retrieval(args):
    """Test pHash on card retrieval task"""
    print("=" * 60)
    print("Testing pHash on Card Retrieval Task")
    print("=" * 60)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(data_dir=args.data_dir)
    
    # Initialize task
    task = CardRetrievalTask(dataset_manager, game=args.game)
    query_ids, gallery_ids, target_indices = task.create_query_gallery(
        num_queries=args.num_queries
    )
    
    print(f"\nQuery set size: {len(query_ids)}")
    print(f"Gallery set size: {len(gallery_ids)}")
    
    # Initialize model
    model = PHashModel(hash_size=args.hash_size)
    
    # Build gallery
    print("\nBuilding gallery...")
    gallery_images = []
    for card_id in tqdm(gallery_ids, desc="Loading gallery images"):
        img = dataset_manager.load_image(args.game, card_id)
        if img is not None:
            gallery_images.append(img)
        else:
            # Use a blank image as placeholder
            from PIL import Image
            gallery_images.append(Image.new("RGB", (224, 224)))
    
    model.build_gallery(gallery_images, gallery_ids)
    
    # Process queries
    print("\nProcessing queries...")
    query_images = []
    for card_id in tqdm(query_ids, desc="Loading query images"):
        img = dataset_manager.load_image(args.game, card_id)
        if img is not None:
            query_images.append(img)
        else:
            from PIL import Image
            query_images.append(Image.new("RGB", (224, 224)))
    
    # Compute similarity matrix
    similarity_matrix = model.compute_similarity_matrix(query_images)
    
    # Evaluate
    print("\nEvaluating...")
    metrics = task.evaluate(similarity_matrix, np.array(target_indices))
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "model": "pHash",
            "hash_size": args.hash_size,
            "game": args.game,
            "num_queries": len(query_ids),
            "gallery_size": len(gallery_ids),
            "metrics": metrics,
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test pHash baseline on card recognition tasks"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["mtg", "pokemon"],
        required=True,
        help="Which game to test on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data (default: data)",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Hash size for pHash (default: 8)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of query images (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phash_results.json",
        help="Output file for results (default: results/phash_results.json)",
    )
    
    args = parser.parse_args()
    
    test_retrieval(args)


if __name__ == "__main__":
    main()
