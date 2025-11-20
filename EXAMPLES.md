# CCG Card ID - Usage Examples

This document provides detailed examples of how to use the CCG Card ID library.

## Example 1: Download and Process MTG Dataset

```python
from ccg_card_id.dataset import ScryfallFetcher, DatasetManager

# Initialize fetcher
fetcher = ScryfallFetcher(data_dir="data")

# Fetch a sample of 500 popular cards
cards = fetcher.fetch_sample_dataset(num_cards=500)

# Save metadata
fetcher.save_metadata(cards)

# Download card images
fetcher.download_card_images(cards)

# Create train/val/test splits
dataset_manager = DatasetManager(data_dir="data")
train_data, val_data, test_data = dataset_manager.create_splits(
    game="mtg",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```

## Example 2: Card Retrieval with pHash

```python
from ccg_card_id.dataset import DatasetManager
from ccg_card_id.models import PHashModel
from PIL import Image

# Initialize dataset manager
dataset_manager = DatasetManager(data_dir="data")

# Load test split
test_cards = dataset_manager.load_split("mtg", "test")

# Initialize pHash model
model = PHashModel(hash_size=8)

# Build gallery with training images
train_cards = dataset_manager.load_split("mtg", "train")
gallery_images = []
gallery_ids = []

for card in train_cards[:100]:  # Use first 100 cards
    card_id = card.get("id")
    img = dataset_manager.load_image("mtg", card_id)
    if img:
        gallery_images.append(img)
        gallery_ids.append(card_id)

model.build_gallery(gallery_images, gallery_ids)

# Find matches for a query card
query_card = test_cards[0]
query_id = query_card.get("id")
query_img = dataset_manager.load_image("mtg", query_id)

matches = model.find_matches(query_img, top_k=5)
print(f"Query: {query_id}")
print("Top 5 matches:")
for card_id, distance in matches:
    print(f"  {card_id}: distance={distance}")
```

## Example 3: Card Recognition with DINOv2

```python
from ccg_card_id.dataset import DatasetManager
from ccg_card_id.models import DINOv2Model

# Initialize dataset manager and model
dataset_manager = DatasetManager(data_dir="data")
model = DINOv2Model(model_name="facebook/dinov2-base")

# Build gallery
train_cards = dataset_manager.load_split("pokemon", "train")
gallery_images = []
gallery_ids = []

for card in train_cards[:200]:
    card_id = card.get("id")
    img = dataset_manager.load_image("pokemon", card_id)
    if img:
        gallery_images.append(img)
        gallery_ids.append(card_id)

model.build_gallery(gallery_images, gallery_ids, batch_size=32)

# Query with test image
test_cards = dataset_manager.load_split("pokemon", "test")
query_card = test_cards[0]
query_img = dataset_manager.load_image("pokemon", query_card["id"])

matches = model.find_matches(query_img, top_k=5)
print(f"Query: {query_card['name']}")
print("Top 5 matches:")
for card_id, similarity in matches:
    print(f"  {card_id}: similarity={similarity:.4f}")
```

## Example 4: Benchmark Evaluation

```python
from ccg_card_id.dataset import DatasetManager
from ccg_card_id.benchmark import CardRetrievalTask
from ccg_card_id.models import DINOv2Model
import numpy as np

# Setup
dataset_manager = DatasetManager(data_dir="data")
task = CardRetrievalTask(dataset_manager, game="mtg")

# Create query and gallery sets
query_ids, gallery_ids, target_indices = task.create_query_gallery(
    num_queries=50
)

# Initialize model
model = DINOv2Model()

# Load images
gallery_images = [dataset_manager.load_image("mtg", id) for id in gallery_ids]
query_images = [dataset_manager.load_image("mtg", id) for id in query_ids]

# Filter None values
gallery_images = [img for img in gallery_images if img is not None]
query_images = [img for img in query_images if img is not None]

# Build gallery and compute similarities
model.build_gallery(gallery_images, gallery_ids)
similarity_matrix = model.compute_similarity_matrix(query_images)

# Evaluate
metrics = task.evaluate(similarity_matrix, np.array(target_indices))
print("Retrieval Metrics:")
for metric_name, value in metrics.items():
    print(f"  {metric_name}: {value:.4f}")
```

## Example 5: Fine-tuning DINOv2

```python
from ccg_card_id.dataset import DatasetManager
from ccg_card_id.models import DINOv2Model
import torch
from torch.utils.data import Dataset, DataLoader

# This is a simplified example - see scripts/finetune_dinov2.py for full implementation

# Create custom dataset
class CardDataset(Dataset):
    def __init__(self, cards, dataset_manager, game, processor):
        self.cards = cards
        self.dataset_manager = dataset_manager
        self.game = game
        self.processor = processor
    
    def __len__(self):
        return len(self.cards)
    
    def __getitem__(self, idx):
        card = self.cards[idx]
        image = self.dataset_manager.load_image(self.game, card["id"])
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

# Load data and create dataloaders
dataset_manager = DatasetManager(data_dir="data")
train_cards = dataset_manager.load_split("mtg", "train")

# Initialize model and processor
from transformers import AutoImageProcessor, AutoModel
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

# Setup training loop
# ... (see scripts/finetune_dinov2.py for complete implementation)
```

## Example 6: Working with Multiple Card Games

```python
from ccg_card_id.dataset import (
    ScryfallFetcher,
    PokemonFetcher,
    DatasetManager
)

# Download datasets from both games
mtg_fetcher = ScryfallFetcher(data_dir="data")
pokemon_fetcher = PokemonFetcher(data_dir="data")

# Fetch Magic cards
mtg_cards = mtg_fetcher.fetch_sample_dataset(num_cards=500)
mtg_fetcher.save_metadata(mtg_cards)
mtg_fetcher.download_card_images(mtg_cards)

# Fetch Pokemon cards
pokemon_cards = pokemon_fetcher.fetch_sample_dataset(num_cards=500)
pokemon_fetcher.save_metadata(pokemon_cards)
pokemon_fetcher.download_card_images(pokemon_cards)

# Create splits for both
dataset_manager = DatasetManager(data_dir="data")
dataset_manager.create_splits("mtg")
dataset_manager.create_splits("pokemon")

# Get statistics
mtg_stats = dataset_manager.get_statistics("mtg")
pokemon_stats = dataset_manager.get_statistics("pokemon")

print("MTG Dataset:", mtg_stats)
print("Pokemon Dataset:", pokemon_stats)
```

## Example 7: Save and Load Model Galleries

```python
from ccg_card_id.models import DINOv2Model, PHashModel
from ccg_card_id.dataset import DatasetManager

dataset_manager = DatasetManager(data_dir="data")

# Build and save DINOv2 gallery
dinov2_model = DINOv2Model()
train_cards = dataset_manager.load_split("mtg", "train")
gallery_images = [dataset_manager.load_image("mtg", c["id"]) for c in train_cards[:100]]
gallery_images = [img for img in gallery_images if img is not None]
gallery_ids = [c["id"] for c in train_cards[:100]]

dinov2_model.build_gallery(gallery_images, gallery_ids)
dinov2_model.save_gallery("models/dinov2_gallery.npz")

# Later, load the gallery
new_model = DINOv2Model()
new_model.load_gallery("models/dinov2_gallery.npz")

# Use the loaded model
query_img = dataset_manager.load_image("mtg", train_cards[0]["id"])
matches = new_model.find_matches(query_img, top_k=5)
print(matches)
```

## Tips and Best Practices

1. **Rate Limiting**: Both Scryfall and Pokemon TCG APIs have rate limits. The fetchers include built-in delays, but for large downloads, consider splitting into multiple sessions.

2. **Image Quality**: For Scryfall, you can choose different image qualities:
   - `small`: Small thumbnail
   - `normal`: Standard card image (default)
   - `large`: High-resolution image
   - `png`: PNG format for transparency

3. **Memory Management**: When working with large datasets, process images in batches to avoid memory issues:
   ```python
   # Process in batches
   batch_size = 100
   for i in range(0, len(cards), batch_size):
       batch = cards[i:i+batch_size]
       # Process batch
   ```

4. **GPU Acceleration**: DINOv2 can leverage GPU for faster processing:
   ```python
   model = DINOv2Model(device="cuda")  # Use GPU if available
   ```

5. **Experiment with Hash Sizes**: For pHash, different hash sizes work better for different use cases:
   - Smaller (4-8): Faster, more tolerant to variations
   - Larger (16-32): More precise, better for exact matching
