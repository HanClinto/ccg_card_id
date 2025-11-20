# Evaluate retrieval performance of different vectorizers
# Tests the ability to find similar cards given a query card

import os
import json
import numpy as np
import imagehash
from tqdm import tqdm

def load_phash_vectors(vectors_file):
    """Load pHash vectors from JSON file."""
    with open(vectors_file, "r") as f:
        vectors = json.load(f)
    return vectors

def load_dinov2_vectors(vectors_file):
    """Load DINOv2 vectors from npz file."""
    data = np.load(vectors_file, allow_pickle=True)
    card_ids = data["card_ids"].tolist()
    embeddings = data["embeddings"]
    # Create dict mapping card_id to embedding
    vectors = {cid: emb for cid, emb in zip(card_ids, embeddings)}
    return vectors

def compute_recall_at_k(similarity_matrix, targets, k=5):
    """
    Compute Recall@K metric.
    
    Args:
        similarity_matrix: (n_queries x n_gallery) similarity scores
        targets: Ground truth indices for each query
        k: Number of top results to consider
    
    Returns:
        Recall@K score
    """
    # Get top-k indices for each query
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
    
    # Check if target is in top-k for each query
    hits = 0
    for i, target_idx in enumerate(targets):
        if target_idx in top_k_indices[i]:
            hits += 1
    
    return hits / len(targets)

def eval_phash_retrieval(vectors, query_ids, gallery_ids):
    """
    Evaluate pHash retrieval performance.
    Returns similarity matrix where lower distance = more similar.
    """
    n_queries = len(query_ids)
    n_gallery = len(gallery_ids)
    similarity_matrix = np.zeros((n_queries, n_gallery))
    
    for i, q_id in enumerate(query_ids):
        if q_id not in vectors:
            continue
        q_hash = imagehash.hex_to_hash(vectors[q_id])
        
        for j, g_id in enumerate(gallery_ids):
            if g_id not in vectors:
                similarity_matrix[i, j] = 999  # Max distance for missing
                continue
            g_hash = imagehash.hex_to_hash(vectors[g_id])
            # Hamming distance (lower is better)
            distance = q_hash - g_hash
            # Convert to similarity (higher is better)
            similarity_matrix[i, j] = 64 - distance  # Max hamming distance is 64
    
    return similarity_matrix

def eval_dinov2_retrieval(vectors, query_ids, gallery_ids):
    """
    Evaluate DINOv2 retrieval performance using cosine similarity.
    Returns similarity matrix.
    """
    # Build query and gallery matrices
    query_embeddings = []
    gallery_embeddings = []
    
    for q_id in query_ids:
        if q_id in vectors:
            query_embeddings.append(vectors[q_id])
        else:
            query_embeddings.append(np.zeros(vectors[list(vectors.keys())[0]].shape))
    
    for g_id in gallery_ids:
        if g_id in vectors:
            gallery_embeddings.append(vectors[g_id])
        else:
            gallery_embeddings.append(np.zeros(vectors[list(vectors.keys())[0]].shape))
    
    query_embeddings = np.array(query_embeddings)
    gallery_embeddings = np.array(gallery_embeddings)
    
    # Normalize embeddings
    query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    gallery_norm = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    
    query_embeddings = query_embeddings / (query_norm + 1e-8)
    gallery_embeddings = gallery_embeddings / (gallery_norm + 1e-8)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(query_embeddings, gallery_embeddings.T)
    
    return similarity_matrix

def create_test_set(all_card_ids, num_queries=100):
    """
    Create a simple test set where query and gallery overlap.
    In a real scenario, you'd have separate sets or duplicates.
    """
    np.random.seed(42)
    query_ids = list(np.random.choice(all_card_ids, size=min(num_queries, len(all_card_ids)), replace=False))
    
    # For this simple test, gallery includes all cards
    gallery_ids = all_card_ids
    
    # Target indices: where each query card appears in gallery
    targets = [gallery_ids.index(qid) if qid in gallery_ids else -1 for qid in query_ids]
    
    return query_ids, gallery_ids, targets

def main():
    print("=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)
    
    # Paths to vector files
    phash_vectors_file = "../04_vectorize/phash/cache/scryfall_phash_vectors.json"
    dinov2_vectors_file = "../04_vectorize/dinov2/cache/scryfall_dinov2_vectors.npz"
    
    # Load vectors
    results = {}
    
    if os.path.exists(phash_vectors_file):
        print("\nEvaluating pHash...")
        phash_vectors = load_phash_vectors(phash_vectors_file)
        all_card_ids = list(phash_vectors.keys())
        
        query_ids, gallery_ids, targets = create_test_set(all_card_ids, num_queries=100)
        similarity_matrix = eval_phash_retrieval(phash_vectors, query_ids, gallery_ids)
        
        recall_at_1 = compute_recall_at_k(similarity_matrix, targets, k=1)
        recall_at_5 = compute_recall_at_k(similarity_matrix, targets, k=5)
        recall_at_10 = compute_recall_at_k(similarity_matrix, targets, k=10)
        
        results["pHash"] = {
            "Recall@1": recall_at_1,
            "Recall@5": recall_at_5,
            "Recall@10": recall_at_10,
        }
        
        print(f"pHash Recall@1: {recall_at_1:.4f}")
        print(f"pHash Recall@5: {recall_at_5:.4f}")
        print(f"pHash Recall@10: {recall_at_10:.4f}")
    
    if os.path.exists(dinov2_vectors_file):
        print("\nEvaluating DINOv2...")
        dinov2_vectors = load_dinov2_vectors(dinov2_vectors_file)
        all_card_ids = list(dinov2_vectors.keys())
        
        query_ids, gallery_ids, targets = create_test_set(all_card_ids, num_queries=100)
        similarity_matrix = eval_dinov2_retrieval(dinov2_vectors, query_ids, gallery_ids)
        
        recall_at_1 = compute_recall_at_k(similarity_matrix, targets, k=1)
        recall_at_5 = compute_recall_at_k(similarity_matrix, targets, k=5)
        recall_at_10 = compute_recall_at_k(similarity_matrix, targets, k=10)
        
        results["DINOv2"] = {
            "Recall@1": recall_at_1,
            "Recall@5": recall_at_5,
            "Recall@10": recall_at_10,
        }
        
        print(f"DINOv2 Recall@1: {recall_at_1:.4f}")
        print(f"DINOv2 Recall@5: {recall_at_5:.4f}")
        print(f"DINOv2 Recall@10: {recall_at_10:.4f}")
    
    # Save results
    if results:
        output_file = "./cache/retrieval_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
