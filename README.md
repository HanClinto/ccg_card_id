# CCG Card ID

Computer vision dataset and library for visual recognition of collectible cards (such as Magic: The Gathering, Pokemon, Yu-Gi-Oh!, and more).

Represents the entire pipeline, from data gathering and cleaning, to card detection (finding the location of a card in an image), to card vectorization, and finally, to card lookup.

Data is cached in each step, and scripts are intended to be run once, in sequence, and then re-run later to update with the latest data available (such as when new sets are printed, or when new real-life camera images are available and added).

Each script should check the dates of each piece of data downloaded or calculated, and only redownload / recalculate if our cached data is missing or outdated.

In this way, we can run the full pipeline of scripts on a nightly basis, and only have to spend time on a minimum of recalculation and download.

## Project Structure

```
01_data_sources/    # Data gathering from APIs
  ├── scryfall/     # Magic: The Gathering (Scryfall API)
  │   ├── 01_sync_data.py     # Download bulk card data
  │   ├── 02_sync_images.py   # Download card images
  │   └── 03_prioritize_data.py  # Prioritize cards by ambiguity
  └── pokemontcgio/ # Pokemon TCG (pokemontcg.io API)
      ├── 01_sync_data.py     # Download Pokemon card data
      └── 02_sync_images.py   # Download Pokemon images

02_data_sets/       # Curated datasets for training/evaluation

03_detector/        # Card detection in images

04_vectorize/       # Convert card images to vectors
  ├── phash/        # Perceptual hash baseline
  ├── dinov2/       # DINOv2 transformer model
  └── brief/        # BRIEF descriptor

05_build/           # Build vector databases

06_eval/            # Evaluation and benchmarking
  └── 01_eval_retrieval.py  # Retrieval performance evaluation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/HanClinto/ccg_card_id.git
cd ccg_card_id

# Install dependencies
pip install -r requirements.txt
```

## Usage Pipeline

### Step 1: Sync Data from APIs

Download card metadata and images from Scryfall (Magic: The Gathering):

```bash
cd 01_data_sources/scryfall
python 01_sync_data.py    # Downloads bulk card data
python 02_sync_images.py  # Downloads card images
```

Download Pokemon TCG cards:

```bash
cd 01_data_sources/pokemontcgio
python 01_sync_data.py    # Downloads Pokemon card data
python 02_sync_images.py  # Downloads Pokemon images

# Optional: Set API key for higher rate limits
export POKEMON_TCG_API_KEY=your_key_here
python 01_sync_data.py
```

### Step 2: Build Vectors

Generate perceptual hash vectors:

```bash
cd 04_vectorize/phash
python 01_build_vectors.py
```

Generate DINOv2 embeddings (requires GPU for best performance):

```bash
cd 04_vectorize/dinov2
python 01_build_vectors.py
```

### Step 3: Evaluate Performance

Compare retrieval performance of different vectorizers:

```bash
cd 06_eval
python 01_eval_retrieval.py
```

## Dependencies

- **requests**: API calls
- **tqdm**: Progress bars
- **orjson** (optional): Faster JSON parsing
- **Pillow**: Image processing
- **imagehash**: Perceptual hashing
- **torch**: Deep learning framework
- **transformers**: DINOv2 model
- **numpy**: Numerical computations

Install all dependencies:

```bash
pip install requests tqdm pillow imagehash torch transformers numpy
```

## Data Flow

1. **Data Sources** → Download raw card data and images from APIs
2. **Data Sets** → Process and organize data into training/test splits
3. **Vectorize** → Convert card images into fixed-length vectors
4. **Build** → Create searchable vector databases
5. **Eval** → Benchmark and compare different methods

## Features

- **Incremental Updates**: Scripts check timestamps and only download/process new data
- **Multiple Data Sources**: Scryfall (MTG), Pokemon TCG API, extensible to more
- **Multiple Vectorizers**: pHash (fast), DINOv2 (accurate), BRIEF (efficient)
- **Evaluation Framework**: Standardized benchmarks for comparing methods
- **Cache Management**: All downloaded and computed data is cached locally

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Scryfall](https://scryfall.com/) for providing the Magic: The Gathering API
- [Pokemon TCG API](https://pokemontcg.io/) for providing Pokemon card data
- [DINOv2](https://github.com/facebookresearch/dinov2) from Meta AI Research
