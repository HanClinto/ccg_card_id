# Build DINOv2 vectors for card images
# Reads card images from data sources and generates DINOv2 embeddings

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 32

def load_dinov2_model(device=None):
    """
    Load the DINOv2 model and processor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading DINOv2 model on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    return model, processor, device

@torch.no_grad()
def compute_dinov2_embedding(images, model, processor, device):
    """
    Compute DINOv2 embeddings for a batch of images.
    Returns numpy array of embeddings.
    """
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embeddings

def build_dinov2_vectors(image_dir, output_file, batch_size=BATCH_SIZE):
    """
    Build DINOv2 vectors for all images in a directory.
    Saves results as numpy .npz file with card_ids and embeddings.
    """
    model, processor, device = load_dinov2_model()
    
    # Find all image files
    image_files = []
    card_ids = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                card_id = os.path.splitext(file)[0]
                image_files.append(image_path)
                card_ids.append(card_id)
    
    print(f"Found {len(image_files)} images")
    
    all_embeddings = []
    
    # Process in batches
    with tqdm(total=len(image_files), desc="Computing DINOv2 embeddings") as pbar:
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i:i + batch_size]
            batch_ids = card_ids[i:i + batch_size]
            
            # Load images
            batch_images = []
            valid_indices = []
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            if batch_images:
                # Compute embeddings
                embeddings = compute_dinov2_embedding(batch_images, model, processor, device)
                all_embeddings.append(embeddings)
            
            pbar.update(len(batch_paths))
    
    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        
        # Save vectors
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez(
            output_file,
            embeddings=all_embeddings,
            card_ids=np.array(card_ids)
        )
        
        print(f"Saved {len(all_embeddings)} DINOv2 vectors to {output_file}")
        print(f"Embedding dimension: {all_embeddings.shape[1]}")
    else:
        print("No embeddings computed!")

if __name__ == "__main__":
    # Example: Build vectors for Scryfall images
    scryfall_images = "../../01_data_sources/scryfall/cache/images/png/front"
    output_file = "./cache/scryfall_dinov2_vectors.npz"
    
    if os.path.exists(scryfall_images):
        print("Building DINOv2 vectors for Scryfall images...")
        build_dinov2_vectors(scryfall_images, output_file)
    else:
        print(f"Image directory not found: {scryfall_images}")
        print("Run Scryfall sync scripts first!")
