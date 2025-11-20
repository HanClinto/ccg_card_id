# CCG Card ID

Computer vision dataset and library for visual recognition of collectible cards (such as Magic: The Gathering, Pokemon, Yu-Gi-Oh!, and more).

## Overview

This repository provides:

1. **Dataset Management**: Scripts to download and manage collectible card datasets from APIs like Scryfall (Magic: The Gathering) and pokemontcg.io (Pokemon TCG)
2. **Benchmark Tasks**: MIEB (Multimodal Image Embedding Benchmark) compatible tasks for evaluating card recognition models
3. **Baseline Models**: Implementations of pHash and DINOv2 for card recognition
4. **Fine-tuning**: Scripts to fine-tune DINOv2 on card datasets for improved performance

## Installation

```bash
# Clone the repository
git clone https://github.com/HanClinto/ccg_card_id.git
cd ccg_card_id

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

### 1. Download Datasets

Download sample datasets for Magic: The Gathering and/or Pokemon TCG:

```bash
# Download both MTG and Pokemon datasets (1000 cards each)
python scripts/download_dataset.py --game both --num-cards 1000 --create-splits

# Download only MTG dataset
python scripts/download_dataset.py --game mtg --num-cards 500

# Download Pokemon dataset with API key (for higher rate limits)
python scripts/download_dataset.py --game pokemon --num-cards 2000 --pokemon-api-key YOUR_API_KEY
```

### 2. Test Baseline Models

#### Test pHash Baseline

```bash
# Test pHash on MTG retrieval task
python scripts/test_phash.py --game mtg --num-queries 100

# Test with custom hash size
python scripts/test_phash.py --game pokemon --hash-size 16 --num-queries 200
```

#### Test DINOv2 Baseline

```bash
# Test DINOv2 on MTG retrieval task
python scripts/test_dinov2.py --game mtg --num-queries 100

# Test with custom model and batch size
python scripts/test_dinov2.py --game pokemon --model-name facebook/dinov2-large --batch-size 64
```

### 3. Fine-tune DINOv2

Fine-tune DINOv2 on your card dataset for improved performance:

```bash
# Fine-tune on MTG dataset
python scripts/finetune_dinov2.py --game mtg --epochs 10 --batch-size 16

# Fine-tune with custom learning rate
python scripts/finetune_dinov2.py --game pokemon --epochs 20 --lr 5e-6
```

## Project Structure

```
ccg_card_id/
├── ccg_card_id/           # Main package
│   ├── dataset/           # Dataset fetching and management
│   │   ├── scryfall_fetcher.py    # MTG/Scryfall API
│   │   ├── pokemon_fetcher.py     # Pokemon TCG API
│   │   └── dataset_manager.py     # Dataset management
│   ├── benchmark/         # MIEB benchmark tasks
│   │   └── tasks.py      # Card matching, retrieval, classification
│   ├── models/           # Model implementations
│   │   ├── phash_model.py        # pHash baseline
│   │   └── dinov2_model.py       # DINOv2 model
│   └── utils/            # Utility functions
│       ├── image_utils.py        # Image processing
│       └── metrics.py            # Evaluation metrics
├── scripts/              # Executable scripts
│   ├── download_dataset.py       # Download datasets
│   ├── test_phash.py            # Test pHash baseline
│   ├── test_dinov2.py           # Test DINOv2 baseline
│   └── finetune_dinov2.py       # Fine-tune DINOv2
├── data/                 # Data storage (created on first run)
│   ├── raw/             # Raw metadata
│   ├── processed/       # Processed data and splits
│   └── images/          # Card images
└── tests/               # Unit tests

```

## Benchmark Tasks

The repository includes three MIEB-compatible benchmark tasks:

1. **Card Matching**: Given two card images, determine if they represent the same card
2. **Card Retrieval**: Given a query card image, retrieve similar cards from a gallery
3. **Card Classification**: Classify cards by attributes (rarity, type, etc.)

## API Documentation

### Dataset Fetchers

#### Scryfall Fetcher (Magic: The Gathering)

```python
from ccg_card_id.dataset import ScryfallFetcher

fetcher = ScryfallFetcher(data_dir="data")

# Fetch sample dataset
cards = fetcher.fetch_sample_dataset(num_cards=1000)

# Fetch specific set
cards = fetcher.fetch_specific_set(set_code="mid")

# Download images
fetcher.download_card_images(cards)

# Save metadata
fetcher.save_metadata(cards)
```

#### Pokemon Fetcher

```python
from ccg_card_id.dataset import PokemonFetcher

fetcher = PokemonFetcher(data_dir="data", api_key="optional_api_key")

# Fetch sample dataset
cards = fetcher.fetch_sample_dataset(num_cards=1000)

# Download images
fetcher.download_card_images(cards)
```

### Models

#### pHash Model

```python
from ccg_card_id.models import PHashModel
from PIL import Image

model = PHashModel(hash_size=8)

# Build gallery
gallery_images = [Image.open(f"card_{i}.jpg") for i in range(100)]
gallery_ids = [f"card_{i}" for i in range(100)]
model.build_gallery(gallery_images, gallery_ids)

# Find matches
query_image = Image.open("query.jpg")
matches = model.find_matches(query_image, top_k=5)
```

#### DINOv2 Model

```python
from ccg_card_id.models import DINOv2Model
from PIL import Image

model = DINOv2Model(model_name="facebook/dinov2-base")

# Build gallery
gallery_images = [Image.open(f"card_{i}.jpg") for i in range(100)]
gallery_ids = [f"card_{i}" for i in range(100)]
model.build_gallery(gallery_images, gallery_ids)

# Find matches
query_image = Image.open("query.jpg")
matches = model.find_matches(query_image, top_k=5)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Scryfall](https://scryfall.com/) for providing the Magic: The Gathering API
- [Pokemon TCG API](https://pokemontcg.io/) for providing the Pokemon card data
- [DINOv2](https://github.com/facebookresearch/dinov2) from Meta AI Research
