#!/usr/bin/env python3
"""
Script to fine-tune DINOv2 on card recognition dataset
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccg_card_id.dataset import DatasetManager


class CardDataset(Dataset):
    """Dataset for card images with metric learning"""

    def __init__(self, dataset_manager, game, split, processor):
        self.dataset_manager = dataset_manager
        self.game = game
        self.split = split
        self.processor = processor
        
        # Load card data
        self.cards = dataset_manager.load_split(game, split)
        
        # Filter cards with valid images
        self.valid_cards = []
        for card in self.cards:
            card_id = card.get("id")
            if card_id and dataset_manager.get_image_path(game, card_id):
                self.valid_cards.append(card)
        
        print(f"Loaded {len(self.valid_cards)} valid cards from {split} split")

    def __len__(self):
        return len(self.valid_cards)

    def __getitem__(self, idx):
        card = self.valid_cards[idx]
        card_id = card.get("id")
        
        # Load image
        image = self.dataset_manager.load_image(self.game, card_id)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "card_id": card_id,
            "idx": idx,
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Batch of embeddings (batch_size x embedding_dim)
            labels: Batch indices (used to determine positive/negative pairs)
        """
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise cosine similarity
        similarity = torch.mm(embeddings, embeddings.t())
        
        # Create positive and negative masks
        batch_size = embeddings.size(0)
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.t()).float()
        mask_negative = (labels != labels.t()).float()
        
        # Remove diagonal (self-similarity)
        mask_positive = mask_positive - torch.eye(batch_size, device=embeddings.device)
        
        # Contrastive loss
        # Positive pairs: maximize similarity
        # Negative pairs: minimize similarity (push apart up to margin)
        positive_loss = (1 - similarity) * mask_positive
        negative_loss = torch.clamp(similarity - (1 - self.margin), min=0) * mask_negative
        
        loss = (positive_loss.sum() + negative_loss.sum()) / (mask_positive.sum() + mask_negative.sum() + 1e-8)
        
        return loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch["pixel_values"].to(device)
        indices = batch["idx"].to(device)
        
        # Forward pass
        outputs = model(pixel_values)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Compute loss
        loss = criterion(embeddings, indices)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            indices = batch["idx"].to(device)
            
            # Forward pass
            outputs = model(pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Compute loss
            loss = criterion(embeddings, indices)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv2 on card recognition dataset"
    )
    parser.add_argument(
        "--game",
        type=str,
        choices=["mtg", "pokemon"],
        required=True,
        help="Which game to train on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data (default: data)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Base DINOv2 model name (default: facebook/dinov2-base)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned",
        help="Directory to save fine-tuned model (default: models/finetuned)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(data_dir=args.data_dir)
    
    # Load model and processor
    print(f"Loading model: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model = model.to(device)
    
    # Create datasets
    train_dataset = CardDataset(dataset_manager, args.game, "train", processor)
    val_dataset = CardDataset(dataset_manager, args.game, "val", processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Setup training
    criterion = ContrastiveLoss(margin=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path(args.output_dir) / args.game
            output_path.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            print(f"Saved best model to {output_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
