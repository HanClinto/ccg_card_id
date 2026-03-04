# MobileViT-XXS Pipeline (Build + Train + Eval)

This is the canonical MobileViT-XXS workflow in the main project structure.

## Stage map

| Stage | Script | Purpose | Primary outputs |
|---|---|---|---|
| 1 | `01_build_manifest.py` | Build reproducible train/val/test manifest | `.../mobilevit_xxs/manifest.csv` |
| 2 | `02_build_triplets.py` | Build task-balanced triplets + hard negatives | `.../mobilevit_xxs/triplets.csv`, `.../mobilevit_xxs/hard_negatives.json` |
| 3 | `03_train_arcface.py` | Train MobileViT-XXS ArcFace embedding model | `.../results/mobilevit_xxs/mobilevit_xxs_arcface_<dim>/last.pt`, `train_history.json` |
| 4 | `04_eval_retrieval.py` | Evaluate one fine-tuned checkpoint on Sol Ring retrieval | `retrieval_summary.json/csv`, `retrieval_predictions.jsonl` |
| 5 | `05_compare_models.py` | Compare base backbone vs one/more fine-tuned checkpoints | `comparison.json`, `comparison.csv` |

For consolidated cross-algorithm reporting (pHash vs DINO vs MobileViT), run:
- `06_eval/04_eval_mobilevit_xxs.py`

For explicit vector precompute/build stage, run:
- `05_build/01_precompute_mobilevit_vectors.py`

---

## Data/output defaults

- Manifest/triplets default root: `~/claw/data/ccg_card_id/mobilevit_xxs/`
- Training results default root (if you pass it): `~/claw/data/ccg_card_id/results/mobilevit_xxs/`
- Sol Ring eval queries: `~/claw/data/ccg_card_id/datasets/solring/04_data/aligned`

Use absolute paths (e.g., `/Volumes/carbonite/...`) if your data lives on external storage.

---

## Caching/resume behavior (project standards)

### 1) `01_build_manifest.py`
- If output CSV already exists, script returns cached result by default.
- Use `--rebuild-cache` to force regenerate.

### 2) `02_build_triplets.py`
- Hard-negative mining resumes by default from `--out-hard-negs-json`.
- Writes periodic checkpoints during long runs.
- Use `--no-resume` to rebuild from scratch.

### 3) `03_train_arcface.py`
- Auto-resumes from `<run_dir>/last.pt` by default when present.
- Explicit resume supported with `--resume-checkpoint`.
- Use `--rebuild-cache` to ignore previous checkpoint and train from scratch.

### 4) `04_eval_retrieval.py`
- Reuses existing `retrieval_summary.json` by default.
- Use `--rebuild-cache` to recompute embeddings/metrics.

### 5) `05_compare_models.py`
- Reuses existing `comparison.json` by default.
- Use `--rebuild-cache` to recompute comparison.

---

## Example commands

### 1) Build manifest

```bash
cd 04_build/mobilevit_xxs
python 01_build_manifest.py \
  --out /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv
```

### 2) Build triplets

```bash
python 02_build_triplets.py \
  --out-csv /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/triplets.csv \
  --out-hard-negs-json /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/hard_negatives.json
```

### 3) Train (initial)

```bash
python 03_train_arcface.py \
  --manifest /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --triplets-csv /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/triplets.csv \
  --task-weights card_id=0.6,set_id=0.25,lang_id=0.15 \
  --output-dir /Volumes/carbonite/claw/data/ccg_card_id/results/mobilevit_xxs \
  --backbone mobilevit_xxs \
  --embedding-dim 128 \
  --image-size 192 \
  --epochs 5 \
  --batch-size 16
```

### 3b) Train (resume additional epochs)

```bash
python 03_train_arcface.py \
  --manifest /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --triplets-csv /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/triplets.csv \
  --output-dir /Volumes/carbonite/claw/data/ccg_card_id/results/mobilevit_xxs \
  --backbone mobilevit_xxs \
  --embedding-dim 128 \
  --epochs 3
```

(That command auto-resumes from `last.pt` unless `--rebuild-cache` is provided.)

### 4) Evaluate one checkpoint

```bash
python 04_eval_retrieval.py \
  --checkpoint /Volumes/carbonite/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/last.pt \
  --manifest /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --solring-dir /Volumes/carbonite/claw/data/ccg_card_id/datasets/solring/04_data/aligned \
  --out-dir /Volumes/carbonite/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/eval_solring
```

### 5) Compare base vs fine-tuned

```bash
python 05_compare_models.py \
  --manifest /Volumes/carbonite/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --out-dir /Volumes/carbonite/claw/data/ccg_card_id/results/mobilevit_xxs/comparisons \
  --solring-dir /Volumes/carbonite/claw/data/ccg_card_id/datasets/solring/04_data/aligned \
  --base-backbone mobilevit_xxs \
  --checkpoint /Volumes/carbonite/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/last.pt
```
