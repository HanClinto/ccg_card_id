# MobileViT-XXS Training + Retrieval Eval

This folder contains the MobileViT-XXS training pipeline (ArcFace + triplet regularization) and companion retrieval eval utilities.

## Scripts

- `01_build_manifest.py` — build training manifest CSV
- `04_build_triplets.py` — build triplets + hard negatives
- `02_train_arcface.py` — train (supports `--resume-checkpoint`)
- `03_eval_retrieval.py` — evaluate a fine-tuned checkpoint on Sol Ring retrieval
- `05_compare_models.py` — compare base vs fine-tuned embeddings
- `03_build_local_real_manifest.py` — optional local real-image manifest builder

## 1) Build manifest

```bash
cd 04_build/mobilevit_xxs
python 01_build_manifest.py \
  --out ~/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv
```

## 2) Build triplets

```bash
python 04_build_triplets.py \
  --out-csv ~/claw/data/ccg_card_id/mobilevit_xxs/triplets.csv \
  --out-hard-negs-json ~/claw/data/ccg_card_id/mobilevit_xxs/hard_negatives.json
```

## 3) Train

```bash
python 02_train_arcface.py \
  --manifest ~/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --triplets-csv ~/claw/data/ccg_card_id/mobilevit_xxs/triplets.csv \
  --task-weights card_id=0.6,set_id=0.25,lang_id=0.15 \
  --output-dir ~/claw/data/ccg_card_id/results/mobilevit_xxs \
  --backbone mobilevit_xxs \
  --embedding-dim 128 \
  --epochs 5 \
  --batch-size 16 \
  --image-size 192 \
  --eval-solring
```

Resume for additional epochs:

```bash
python 02_train_arcface.py \
  --manifest ~/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --output-dir ~/claw/data/ccg_card_id/results/mobilevit_xxs \
  --backbone mobilevit_xxs \
  --embedding-dim 128 \
  --resume-checkpoint ~/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/last.pt \
  --epochs 3
```

## 4) Evaluate checkpoint

```bash
python 03_eval_retrieval.py \
  --checkpoint ~/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/last.pt \
  --manifest ~/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --out-dir ~/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/eval_solring
```

## 5) Compare base vs fine-tuned

```bash
python 05_compare_models.py \
  --manifest ~/claw/data/ccg_card_id/mobilevit_xxs/manifest.csv \
  --out-dir ~/claw/data/ccg_card_id/results/mobilevit_xxs/comparisons \
  --base-backbone mobilevit_xxs \
  --checkpoint ~/claw/data/ccg_card_id/results/mobilevit_xxs/mobilevit_xxs_arcface_128/last.pt
```

Use `06_eval/04_eval_mobilevit_xxs.py` to include base/fine-tuned MobileViT metrics in the standard eval reports.
