# CCG Card ID — Project Guide

## Working conventions

- **Commit and push frequently** — after completing any meaningful unit of work (a feature, a fix, a refactor), stage the relevant files, commit with a descriptive message, and `git push`. Do not wait to be asked.
- Never commit `.venv312/`, data files, or model checkpoints — these are excluded by `.gitignore`.

## Goal

Build a system that identifies Magic: The Gathering cards from real-world images
(phone camera, flatbed scanner, video frames). The full pipeline is:

```
Raw image → [Corner detection / dewarp] → [Card identification]
```

Card identification uses metric learning (ArcFace) on a MobileViT-XXS backbone.
The gallery is ~108k Scryfall reference images; queries are real-world captures.

---

## Environment

```bash
source .venv312/bin/activate   # Python 3.12 virtualenv
```

Data root is set via env var — all code reads it through `cfg`:

```python
from ccg_card_id.config import cfg   # cfg.data_dir → Path
```

Resolution order: `CCG_DATA_DIR` env var → `~/claw/data/ccg_card_id/`.
On the primary dev machine the data lives on an external drive:
`/Volumes/carbonite/claw/data/ccg_card_id/`.

---

## Code layout

```
ccg_card_id/                    package — config, shared utilities
  config.py                     cfg object and data-dir resolution
  project_settings.py           get_data_dir()

02_data_sets/                   per-dataset processing pipelines
  sol_ring/
  clint_cards_with_backgrounds/
  munchie/
  daniel_scans/

04_build/
  mobilevit_xxs/
    01_build_manifest.py        build artwork_id_manifest.csv
    02_train.py                 single-task ArcFace training
    03_train_multitask.py       multi-task ArcFace (illustration_id + set_code)
    models.py                   EmbeddingNet, MultiTaskEmbeddingNet, ArcFaceLoss
    data.py                     ManifestRow, load_manifest
    retrieval.py                embed_paths, evaluate_retrieval, pHash eval

06_eval/
  04_eval_mobilevit_xxs.py      full evaluation harness
  reporting.py                  CSV/Markdown report writers
```

---

## Data layout (relative to `cfg.data_dir`)

```
catalog/scryfall/images/png/front/{a}/{b}/{uuid}.png   Scryfall reference PNGs
all_cards.json                  Scryfall bulk JSON, all languages (~517k cards)
default_cards.json              Scryfall bulk JSON, English defaults

mobilevit_xxs/
  artwork_id_manifest.csv       81 834 rows — training manifest (see below)

datasets/
  solring/04_data/aligned/      307 homography-aligned Sol Ring video frames
  clint_cards_with_backgrounds/ 1271 phone-video frames, 39 cards, backgrounds
    data/04_data/corners.csv    per-frame corner coordinates (relative paths)
  daniel_scans/                 150 phone scans, clean backgrounds
  munchie/                      564 flatbed scanner images, 134 illustration_ids

results/
  mobilevit_xxs/                training checkpoints
    mobilevit_xxs_illustration_id_128d/   single-task run (epochs 1–36)
      epoch_0005.pt … epoch_0035.pt
      last.pt                   epoch 36
      train_history.json
    mobilevit_xxs_multitask_illustration_id+set_code_shared_128d/  (in progress)
    seeds/                      hand-picked checkpoints for transfer learning
      artwork_id_e10_128d.pt … artwork_id_e30_128d.pt
  eval/{timestamp}/             per-run eval reports (summary.csv, overview.md)
    history_results.csv         cumulative across all runs
    latest_results.csv          most recent result per algorithm×dataset

vectors/mobilevit_xxs/img224/   embedding cache (.npz), keyed by model variant
vectors/phash/                  pHash cache (.npz)
```

---

## Manifests

**ManifestRow** schema (`data.py`):

| Column | Description |
|---|---|
| `image_path` | Path relative to `cfg.data_dir` |
| `card_id` | Scryfall UUID (lowercase) |
| `card_name` | Display name |
| `set_code` | 3-4 letter set abbreviation |
| `split` | `train` / `val` / `test` |
| `illustration_id` | Artwork UUID — groups reprints sharing the same art |
| `oracle_id` | Card oracle UUID |
| `lang` | Language code (`en`, `de`, …) |
| `source` | `scryfall` or `munchie` |

`artwork_id_manifest.csv`: 81 834 rows, ~20 524 unique `illustration_id` classes.
Split: 69 731 train / 8 028 val / 4 075 test (stratified by illustration_id).
Combines Scryfall reference images + Munchie flatbed scans.
Built by `04_build/mobilevit_xxs/01_build_manifest.py`.

> **Note:** Existing manifests still contain absolute paths — migrate to
> relative paths when regenerating (see Path conventions below).

---

## Path conventions

**All `image_path` fields in manifests, CSVs, and JSON must be relative to
`cfg.data_dir`**, not absolute. This makes files portable across machines.

```python
# Writing
row["image_path"] = str(path.relative_to(cfg.data_dir))

# Reading
absolute = cfg.data_dir / row["image_path"]
```

---

## Card identity

Reference cards by **Scryfall UUID** (`card_id`). Captured-image filenames
embed the UUID as the first underscore-delimited field:

```
{card_uuid}_{HumanName}_{date}.mp4-{frame}.jpg
```

Extract the UUID with a regex (do not use `split('_')[0]` — while that works
for the current naming scheme, the regex is more explicit and robust):

```python
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
card_id = UUID_RE.search(filename).group(0).lower()
```

---

## Model architecture

**EmbeddingNet** (`models.py`): MobileViT-XXS backbone + linear projection →
L2-normalised 128-d embedding. Trained with ArcFace loss (angular margin).

**MultiTaskEmbeddingNet**: shared backbone, one projection per task (or a
single shared projection), one ArcFace head per label field.

Training uses `AdamW` + `CosineAnnealingLR`. **On every resume, reset the
optimizer LR to `args.lr`** before creating the scheduler — without this,
the scheduler inherits the near-zero end-of-cycle LR and the new cycle is
useless. This is already implemented in both training scripts.

---

## Training phases

| Phase | Label field | Classes | Notes |
|---|---|---|---|
| 1 — Artwork ID | `illustration_id` | ~20.5k | Trains first; groups all printings sharing the same art |
| 2 — Joint (art + set) | `illustration_id` + `set_code` | 20.5k + 802 | Multi-task from scratch (currently training) |
| 3 — Edition ID | `set_code` | ~900 sets | Use set_code not card_id: ~100–300 samples/class vs 1 |
| 4 — Language ID | `lang` | ~20 | Future |

**Why `set_code` not `card_id` for edition training:** ArcFace needs multiple
samples per class. `card_id` has ~1 Scryfall image per card (~108k one-sample
classes → training diverges). `set_code` has ~100–300 cards per set.

### Single-task training

```bash
# From scratch (artwork ID)
python 04_build/mobilevit_xxs/02_train.py \
  --label-field illustration_id --lr 2e-3 --epochs 15 --rebuild

# Resume (adds a fresh cosine cycle)
python 04_build/mobilevit_xxs/02_train.py \
  --label-field illustration_id --lr 2e-3 --epochs 15
```

Checkpoints: `results/mobilevit_xxs/mobilevit_xxs_{label}_{dim}d/`

### Multi-task training

```bash
# Shared embedding, from scratch
python 04_build/mobilevit_xxs/03_train_multitask.py \
  --label-fields illustration_id set_code --lr 2e-3 --epochs 15 --rebuild

# Separate per-task projections (128-d art, 64-d set)
python 04_build/mobilevit_xxs/03_train_multitask.py \
  --separate-heads --embedding-dims 128 64 --lr 2e-3 --epochs 15 --rebuild
```

Checkpoints: `results/mobilevit_xxs/mobilevit_xxs_multitask_{fields}_{mode}_{dims}/`

---

## Evaluation

```bash
python 06_eval/04_eval_mobilevit_xxs.py \
  --manifest $DATA/mobilevit_xxs/artwork_id_manifest.csv \
  --query-manifest daniel=$DATA/datasets/daniel_scans/query_manifest.csv \
  --query-manifest clint_backgrounds=$DATA/datasets/clint_cards_with_backgrounds/query_manifest.csv \
  --query-manifest munchie=$DATA/datasets/munchie/manifest.csv
```

Auto-discovers all `epoch_XXXX.pt` checkpoints under `--results-root`.
Evaluates two criteria per model × dataset:
- **artwork** — query matches any gallery card sharing the same `illustration_id`
- **edition** — query matches the exact `card_id`

Also evaluates pHash baselines at 8×8 (64-bit, 8 B), 16×16 (256-bit, 32 B),
32×32 (1024-bit, 128 B). Skip with `--skip-phash`.

Gallery embeddings are cached per model variant and reused across query datasets.

---

## Key results (as of epoch 36, single-task illustration_id)

Top-1 **artwork** accuracy:

| Model | Bytes | solring | daniel | clint_bg | munchie |
|---|---:|---:|---:|---:|---:|
| pHash 16×16 | 32 | 100% | 99.3% | 93.5% | 99.3% |
| pHash 32×32 | 128 | 100% | 92.0% | 96.9% | 99.3% |
| ArcFace e20 | 512 | 43% | 93.3% | 29.9% | 95.0% |
| ArcFace e25 | 512 | 55% | 98.7% | 45.4% | 98.6% |
| ArcFace e30 | 512 | 59% | 98.7% | 50.4% | 99.1% |

**pHash dominates artwork ID** at far lower storage cost. ArcFace pulls ahead
of pHash on **edition (exact printing) identification**, especially on
structured captures. The domain gap (Scryfall PNGs vs video frames) is the
main limiter for solring and clint_backgrounds.

---

## Corner detection / dewarping

The identification model expects a dewarped card image as input. The upstream
step locates card corners in a raw image using SIFT homography against the
known Scryfall reference for that card.

Key library: `../ClintUtils/align_img.py` (SIFT + FLANN, pure OpenCV).

Per-frame corner data for `clint_cards_with_backgrounds`:
`datasets/clint_cards_with_backgrounds/data/04_data/corners.csv`
Columns: `img_path` (relative), `card_id`, `corner0_x/y` … `corner3_x/y`
(normalised 0–1), `num_good_matches`, `matching_area_pct`.

Generation script: `02_data_sets/clint_cards_with_backgrounds/code/03_extract_corners.py`
Supports resume via `corners_progress.json`.

---

## Known issues / technical debt

- Existing manifests use **absolute paths** — must migrate to relative on
  regeneration (see Path conventions above).
- `train_arcface.py` in older runs was the precursor to `02_train.py` — ignore
  it; `02_train.py` is the current single-task training script.
- Early training runs (`ft_e5_128d`, `ft_e14_128d`) used `card_id` as the
  ArcFace class label (~108k classes, ~1 sample each) and diverged or
  collapsed. All current runs use `illustration_id` (~20.5k classes,
  multi-sample) which trains stably.
- **Val split is not open-set** — the current manifest stratifies by
  `illustration_id`, so every class has images in both train and val. This
  means validation loss measures recognition of *trained* artworks, not
  generalisation to unseen ones. **For the next training run, rebuild the
  manifest with a proper open-set split: reserve ~15–20% of `illustration_id`s
  entirely (no training images from those classes), and use only those held-out
  classes for validation retrieval.** This is critical for evaluating transfer
  to new Magic printings, Pokémon, Yu-Gi-Oh, or any other card game.
