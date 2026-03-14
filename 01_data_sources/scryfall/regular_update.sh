#!/usr/bin/env bash
# Regular Scryfall data update pipeline.
# Run from the project root:
#   bash 01_data_sources/scryfall/regular_update.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== Step 1: Sync bulk data ==="
uv run python 01_data_sources/scryfall/01_sync_data.py

echo ""
echo "=== Step 2: Rebuild card catalog DB ==="
uv run python 01_data_sources/scryfall/02_build_card_db.py --rebuild

echo ""
echo "=== Step 3: Sync card images ==="
uv run python 01_data_sources/scryfall/03_sync_images.py

echo ""
echo "=== Step 4: Refresh TCGplayer prices ==="
uv run python 07_web_scanner/server/populate_prices.py

echo ""
echo "=== Done ==="
