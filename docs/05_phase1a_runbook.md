# Phase 1a Runbook — Head Mini-Bakeoff (Backbone Fixed)

Goal: find a stable head/loss setup for a **single embedding** that serves both:
- Name/Artwork identification (Top-1)
- Edition identification (Top-1)

Backbone is fixed in this phase: `mobilevit_xxs`.

---

## Locked assumptions

- Seed: **42** (global default; not included in run names)
- Input size: **448x448**
- Train data: packopening aligned cache on SSD (`aligned_448`)
- pHash quality filter: **max phash_dist = 15** (start), optional stricter run at 10 later
- Light augmentation default: ColorJitter only
- No HFlip / VFlip
- Optional exact 180° rotation via `--rotate-180-p` (default off)

Eval/reporting datasets:
- `solring` (primary torture test for edition)
- `clint_backgrounds`
- `daniel`
- `munchie`

---

## Naming convention (phase 1a)

`<bb>_<head>_<trainv>_img<sz>_ph<dist>_e<ep>`

Examples:
- `mvitxxs_shared2h_arcface_v1_img448_ph15_e030`
- `mvitxxs_shared1h_art_v1_img448_ph15_e030`
- `mvitxxs_shared1h_ed_v1_img448_ph15_e030`

During training, use `--run-tag` without epoch; append epoch in reports.

---

## Step 0 — Ensure cached 448 manifest exists

Expected output from precache step:
- `FAST/datasets/packopening/manifest_aligned_448_ph15.csv`

Where `FAST` is `cfg.fast_data_dir`.

If you need to regenerate with filtering:

```bash
./.venv312/bin/python 04_vectorize/precache_packopening_aligned.py \
  --size 448 \
  --max-phash-dist 15 \
  --workers 8
```

Default behavior skips existing cached files. Use `--rebuild` only when needed.

---

## Step 1 — Head variant runs (baseline set)

All runs below use the same manifest + backbone + image size.

### 1A) shared 2-head ArcFace (primary baseline)

```bash
./.venv312/bin/python 04_vectorize/mobilevit_xxs/03_train_multitask.py \
  --manifest /Users/hanclaw/claw/fast_data/ccg_card_id/datasets/packopening/manifest_aligned_448_ph15.csv \
  --backbone mobilevit_xxs \
  --label-fields illustration_id set_code \
  --task-weights 1.0 1.0 \
  --embedding-dims 128 \
  --image-size 448 \
  --epochs 30 \
  --batch-size 64 \
  --lr 3e-4 \
  --checkpoint-every 5 \
  --run-tag mvitxxs_shared2h_arcface_v1_img448_ph15
```

### 1B) shared 1-head artwork ArcFace (control)

```bash
./.venv312/bin/python 04_vectorize/mobilevit_xxs/03_train_multitask.py \
  --manifest /Users/hanclaw/claw/fast_data/ccg_card_id/datasets/packopening/manifest_aligned_448_ph15.csv \
  --backbone mobilevit_xxs \
  --label-fields illustration_id \
  --task-weights 1.0 \
  --embedding-dims 128 \
  --image-size 448 \
  --epochs 30 \
  --batch-size 64 \
  --lr 3e-4 \
  --checkpoint-every 5 \
  --run-tag mvitxxs_shared1h_art_arcface_v1_img448_ph15
```

### 1C) shared 1-head edition-proxy ArcFace (control)

```bash
./.venv312/bin/python 04_vectorize/mobilevit_xxs/03_train_multitask.py \
  --manifest /Users/hanclaw/claw/fast_data/ccg_card_id/datasets/packopening/manifest_aligned_448_ph15.csv \
  --backbone mobilevit_xxs \
  --label-fields set_code \
  --task-weights 1.0 \
  --embedding-dims 128 \
  --image-size 448 \
  --epochs 30 \
  --batch-size 64 \
  --lr 3e-4 \
  --checkpoint-every 5 \
  --run-tag mvitxxs_shared1h_ed_arcface_v1_img448_ph15
```

---

## Step 2 — Evaluate checkpoints

Use the resulting checkpoints with:

```bash
./.venv312/bin/python 06_eval/04_eval_mobilevit_xxs.py \
  --checkpoint /path/to/last.pt \
  --query-manifest daniel=/Volumes/carbonite/claw/data/ccg_card_id/datasets/daniel_scans/query_manifest.csv \
  --query-manifest clint_backgrounds=/Volumes/carbonite/claw/data/ccg_card_id/datasets/clint_cards_with_backgrounds/query_manifest.csv \
  --query-manifest munchie=/Volumes/carbonite/claw/data/ccg_card_id/datasets/munchie/manifest.csv \
  --skip-base --skip-phash \
  --image-size 448
```

Primary decision metrics:
- Top-1 Name/Artwork
- Top-1 Edition

Primary gate:
- beat or approach pHash on name while improving edition, especially on `solring`.

---

## Note on old dual-head checkpoint evaluation

A quick reference eval of the last historical dual-head checkpoint was attempted, but a full fresh run is expensive with full gallery embedding passes and no cached dual-head eval artifact was found in prior summary outputs.

Treat historical checkpoints as informal reference only; use Phase 1a runs above as clean comparable baselines under the new 448 + filtered-data policy.
