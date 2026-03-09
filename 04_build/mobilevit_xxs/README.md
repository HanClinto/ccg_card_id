# MobileViT-XXS Training Pipeline

Metric-learning pipeline for CCG card identification using MobileViT-XXS + ArcFace loss.

---

## Scripts

| Script | Purpose |
|---|---|
| `01_build_manifest.py` | Build reproducible train/val/test manifest CSV from Scryfall images + real-world datasets |
| `02_train.py` | Single-task ArcFace training (one label field at a time) |
| `03_train_multitask.py` | Multi-task ArcFace training (shared backbone, one head per label field) |
| `models.py` | `EmbeddingNet`, `MultiTaskEmbeddingNet`, `ArcFaceLoss`, `build_backbone` |
| `data.py` | `ManifestRow`, `load_manifest`, `build_manifest_from_scryfall` |
| `retrieval.py` | Embedding + pHash eval utilities used by `06_eval/` |

Evaluation lives in **`06_eval/04_eval_mobilevit_xxs.py`** — runs all model variants and pHash baselines across multiple query datasets and writes CSV/Markdown reports.

---

## Training phases

The intended training progression is hierarchical, each phase building on the previous:

| Phase | Label field | Classes | Notes |
|---|---|---|---|
| 1 — Artwork ID | `illustration_id` | ~20 k | Groups all printings sharing the same artwork. Train first. |
| 2 — Joint (art + set) | `illustration_id` + `set_code` | ~20 k + ~1 k | Multi-task from scratch; teaches both artwork and set-frame features simultaneously. |
| 3 — Edition ID | `card_id` (or `set_code`) | ~1 k sets / 108 k cards | Distinguishes specific printings. Use `set_code` as ArcFace classes (100–300 samples/class) rather than `card_id` (1 sample/class). |
| 4 — Language ID | `lang` | ~20 | Future. |

**Why `set_code` instead of `card_id` for edition training:**
ArcFace needs multiple samples per class. `card_id` has ~1 Scryfall image per card (108 k one-sample classes → diverges). `set_code` has ~100–300 cards per set (~900 classes → well-suited for ArcFace). Combined with an artwork embedding, this gives printing-level retrieval.

---

## Single-task training (`02_train.py`)

```bash
# Artwork ID — train from scratch
python 02_train.py \
  --manifest $DATA/mobilevit_xxs/artwork_id_manifest.csv \
  --label-field illustration_id \
  --lr 2e-3 --epochs 15 --batch-size 64 \
  --scheduler cosine --checkpoint-every 5

# LR range test first (recommended before a new phase)
python 02_train.py --manifest ... --label-field illustration_id --lr-find

# Continue training (auto-resumes from last.pt; fresh cosine cycle from --lr)
python 02_train.py --manifest ... --label-field illustration_id \
  --lr 2e-3 --epochs 15

# Train from scratch, ignoring any existing checkpoint
python 02_train.py --manifest ... --rebuild
```

Checkpoint directory: `<output-dir>/<backbone>_<label_field>_<dim>d/`
e.g. `results/mobilevit_xxs/mobilevit_xxs_illustration_id_128d/`

**LR schedule note:** On resume, `--lr` is always used as the starting LR for the new cosine cycle regardless of where the previous run ended. This gives a clean warm-restart each time.

---

## Multi-task training (`03_train_multitask.py`)

Trains a shared MobileViT-XXS backbone with one ArcFace head per task. Two embedding modes:

```
shared (default)    backbone → [one projection] → z (128-d) → ArcFace_task0
                                                             → ArcFace_task1
                    One vector at query time. Tasks share the same embedding space.

separate-heads      backbone → [proj_0] → z₀ (128-d) → ArcFace_task0
                             → [proj_1] → z₁  (64-d) → ArcFace_task1
                    Independent embeddings per task (can have different dims).
                    Concatenate at query time or use the relevant head.
```

```bash
# Joint artwork + set from scratch — shared embedding (recommended first experiment)
python 03_train_multitask.py \
  --lr 2e-3 --epochs 15 --batch-size 64 --scheduler cosine

# Separate projections: 128-d for artwork, 64-d for set
python 03_train_multitask.py \
  --separate-heads --embedding-dims 128 64 \
  --lr 2e-3 --epochs 15

# Custom task weights (down-weight set_code early on)
python 03_train_multitask.py --task-weights 1.0 0.5

# LR range test
python 03_train_multitask.py --lr-find

# Resume (auto-detects last.pt; resets LR to --lr for fresh cosine cycle)
python 03_train_multitask.py --lr 2e-3 --epochs 15

# Change label fields (any ManifestRow column with enough samples)
python 03_train_multitask.py --label-fields illustration_id set_code lang
```

Checkpoint directory: `<backbone>_multitask_<fields>_<mode>_<dims>/`
e.g. `mobilevit_xxs_multitask_illustration_id+set_code_shared_128d/`
e.g. `mobilevit_xxs_multitask_illustration_id+set_code_sep_128d+64d/`

---

## Seed checkpoints

Well-trained intermediate checkpoints are copied to `results/mobilevit_xxs/seeds/`
for use as transfer-learning starting points. Pass with `--resume-checkpoint`.

| Checkpoint | Epochs | Suggested use |
|---|---|---|
| `artwork_id_e10_128d.pt` | 10 | General card features; best seed for language ID |
| `artwork_id_e15_128d.pt` | 15 | End of first cosine cycle |
| `artwork_id_e20_128d.pt` | 20 | Start of second cycle; good balance for edition/set ID |
| `artwork_id_e25_128d.pt` | 25 | Strong artwork; recommended seed for edition/set training |
| `artwork_id_e30_128d.pt` | 30 | End of second cycle |

```bash
# Start edition training from the e25 artwork seed
python 02_train.py \
  --resume-checkpoint $DATA/results/mobilevit_xxs/seeds/artwork_id_e25_128d.pt \
  --label-field set_code \
  --lr 2e-3 --epochs 15 --rebuild
```

---

## Evaluation

Run from the project root. Auto-discovers all checkpoints under `--results-root`.

```bash
DATA=/Volumes/carbonite/claw/data/ccg_card_id

python 06_eval/04_eval_mobilevit_xxs.py \
  --manifest $DATA/mobilevit_xxs/manifest.csv \
  --query-manifest daniel=$DATA/datasets/daniel_scans/query_manifest.csv \
  --query-manifest clint_backgrounds=$DATA/datasets/clint_cards_with_backgrounds/query_manifest.csv \
  --query-manifest munchie=$DATA/datasets/munchie/manifest.csv

# Skip base model and pHash (faster, just fine-tuned checkpoints)
python 06_eval/04_eval_mobilevit_xxs.py ... --skip-base --skip-phash

# pHash sizes to evaluate (default: 8 16 32 → 64/256/1024-bit)
python 06_eval/04_eval_mobilevit_xxs.py ... --phash-sizes 8 16 32
```

Two accuracy criteria per query dataset:
- **artwork** — query matches any gallery card sharing the same `illustration_id`
- **edition** — query matches the exact `card_id`

Results written to `results/eval/<timestamp>/`: `summary.csv`, `overview.md`, `failures.jsonl`, per-variant markdown files, and cumulative `history_results.csv` / `latest_results.csv`.

---

## Manifest

The `artwork_id_manifest.csv` used for training combines Scryfall reference images and
real-world Munchie scanner scans, filtered to `illustration_id`s with ≥ 2 samples:

- ~81 k rows, ~20.5 k unique `illustration_id` classes
- Split: 69.7 k train / 8 k val / 4 k test (by `illustration_id`)
- Built by `01_build_manifest.py` (see that script for rebuild instructions)
