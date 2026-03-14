#!/usr/bin/env bash
# Overnight training queue — runs sequentially after the corner detector finishes.
#
# Steps:
#   1. Wait for the corner detector (train.py) to finish
#   2. Resume single-task ArcFace (illustration_id) from e90 for 15 more epochs
#   3. Restart multitask ArcFace (illustration_id + set_code) from scratch,
#      seeded from the single-task last.pt, with arcface-scale=8.0 to prevent collapse
#
# Usage (run from project root, leave terminal open or use nohup/screen):
#   bash overnight_training.sh
#   nohup bash overnight_training.sh > overnight_training.log 2>&1 &

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SINGLE_TASK_CKPT_DIR="$(python3 -c "
import sys; sys.path.insert(0,'.')
from ccg_card_id.config import cfg
print(cfg.data_dir / 'results/mobilevit_xxs/mobilevit_xxs_illustration_id_128d')
")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Step 1 — wait for corner detector to finish
# ---------------------------------------------------------------------------

log "Checking for running corner detector..."
CORNER_PID=$(pgrep -f "train.py.*packopening" || true)

if [ -n "$CORNER_PID" ]; then
  log "Corner detector is running (PID $CORNER_PID) — waiting for it to finish..."
  while kill -0 "$CORNER_PID" 2>/dev/null; do
    sleep 30
  done
  log "Corner detector finished."
else
  log "Corner detector is not running — proceeding immediately."
fi

# ---------------------------------------------------------------------------
# Step 2 — resume single-task ArcFace for 15 more epochs
# ---------------------------------------------------------------------------

log "Starting single-task ArcFace resume (illustration_id, 15 epochs from e90)..."
uv run python 04_build/mobilevit_xxs/02_train.py \
  --label-field illustration_id \
  --lr 2e-3 \
  --epochs 15 \
  --arcface-scale 32.0 \
  --num-workers 4 \
  --checkpoint-every 5

log "Single-task ArcFace resume complete."
log "New last.pt: $SINGLE_TASK_CKPT_DIR/last.pt"

# ---------------------------------------------------------------------------
# Step 3 — restart multitask ArcFace from scratch, seeded from new single-task
# ---------------------------------------------------------------------------

SEED_CKPT="$SINGLE_TASK_CKPT_DIR/last.pt"

if [ ! -f "$SEED_CKPT" ]; then
  log "ERROR: seed checkpoint not found at $SEED_CKPT"
  exit 1
fi

log "Starting multitask ArcFace from scratch (illustration_id + set_code, seeded from $SEED_CKPT)..."
uv run python 04_build/mobilevit_xxs/03_train_multitask.py \
  --label-fields illustration_id set_code \
  --lr 2e-3 \
  --epochs 15 \
  --arcface-scale 8.0 \
  --num-workers 4 \
  --checkpoint-every 5 \
  --rebuild \
  --seed-checkpoint "$SEED_CKPT"

log "Multitask ArcFace complete."
log "All training done."
