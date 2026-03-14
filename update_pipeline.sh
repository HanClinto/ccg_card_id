#!/usr/bin/env bash
# Full CCG Card ID update pipeline.
#
# Run this after new Scryfall data is available or after a new training run
# to bring the catalog, gallery embeddings, and price data up to date.
#
# Usage (run from project root):
#   bash update_pipeline.sh
#   bash update_pipeline.sh --skip-images   # skip image sync (fastest)
#   bash update_pipeline.sh --skip-gallery  # skip embedding recompute
#
# Steps:
#   1. Sync Scryfall bulk JSON
#   2. Rebuild card catalog SQLite DB
#   3. Sync Scryfall card images          [skippable with --skip-images]
#   4. Refresh TCGplayer prices
#   5. Rebuild artwork_id manifest
#   6. Precache gallery images at 224x224 [skippable with --skip-images]
#   7. Recompute gallery embeddings       [skippable with --skip-gallery]
#        - pHash (8x8, 16x16, 32x32)
#        - ArcFace (latest checkpoint)

set -euo pipefail

SKIP_IMAGES=false
SKIP_GALLERY=false

for arg in "$@"; do
  case $arg in
    --skip-images)  SKIP_IMAGES=true  ;;
    --skip-gallery) SKIP_GALLERY=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

echo "=================================================="
echo " CCG Card ID — full update pipeline"
echo "=================================================="
echo ""

echo "=== Step 1: Sync Scryfall bulk data ==="
uv run python 01_data_sources/scryfall/01_sync_data.py

echo ""
echo "=== Step 2: Rebuild card catalog DB ==="
uv run python 01_data_sources/scryfall/02_build_card_db.py --rebuild

if [ "$SKIP_IMAGES" = false ]; then
  echo ""
  echo "=== Step 3: Sync Scryfall card images ==="
  uv run python 01_data_sources/scryfall/03_sync_images.py
else
  echo ""
  echo "=== Step 3: Sync Scryfall card images (SKIPPED) ==="
fi

echo ""
echo "=== Step 4: Refresh TCGplayer prices ==="
uv run python 07_web_scanner/server/populate_prices.py

echo ""
echo "=== Step 5: Rebuild artwork ID manifest ==="
uv run python 04_build/mobilevit_xxs/01_build_manifest.py

if [ "$SKIP_IMAGES" = false ]; then
  echo ""
  echo "=== Step 6: Precache gallery images at 224x224 ==="
  uv run python 06_eval/precache_gallery.py --workers 8
else
  echo ""
  echo "=== Step 6: Precache gallery images (SKIPPED) ==="
fi

if [ "$SKIP_GALLERY" = false ]; then
  echo ""
  echo "=== Step 7: Recompute gallery embeddings ==="
  uv run python 05_build/02_update_gallery_vectors.py
else
  echo ""
  echo "=== Step 7: Recompute gallery embeddings (SKIPPED) ==="
fi

echo ""
echo "=================================================="
echo " Done."
echo "=================================================="
