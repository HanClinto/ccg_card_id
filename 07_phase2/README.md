# Phase 2: Metric Learning Scaffold (ArcFace)

This folder adds a conservative, reproducible Phase-2 training scaffold to improve retrieval beyond discrete hashes at 128/256 bits.

## What is included

- **Manifest pipeline**: reproducible CSV with
  `image_path, card_id, card_name, set_code, split`
- **Triplet/pair mining helpers**:
  - positive = same-card image if available
  - fallback positive = synthetic augmentation of anchor
  - negative = hard negatives from eval failures JSONL when available, else random different-card negative
- **Baselines**:
  - TinyViT + ArcFace (`--backbone tinyvit`)
  - ResNet-50 + ArcFace (`--backbone resnet50`)
- **Retrieval eval hook** for Sol Ring style set:
  - writes `top1/top3/top10`
  - writes summary CSV/JSON + per-query JSONL predictions

## Dependencies (optional for phase2)

```bash
pip install torch torchvision timm
```

## 1) Build manifest

```bash
cd 07_phase2
python 01_build_manifest.py \
  --out ~/claw/data/ccg_card_id/phase2/manifest.csv \
  --seed 42
```

Optional hard-negatives input can come from existing eval failures, e.g.:
`~/claw/data/ccg_card_id/results/eval/<run_id>/failures.jsonl`

## 2) Train TinyViT + ArcFace (128-dim)

```bash
cd 07_phase2
python 02_train_arcface.py \
  --manifest ~/claw/data/ccg_card_id/phase2/manifest.csv \
  --output-dir ~/claw/data/ccg_card_id/results/phase2 \
  --backbone tinyvit \
  --embedding-dim 128 \
  --epochs 5 \
  --batch-size 16 \
  --eval-solring \
  --hard-negatives-jsonl ~/claw/data/ccg_card_id/results/eval/<run_id>/failures.jsonl
```

## 3) Train TinyViT + ArcFace (256-dim)

```bash
cd 07_phase2
python 02_train_arcface.py \
  --manifest ~/claw/data/ccg_card_id/phase2/manifest.csv \
  --output-dir ~/claw/data/ccg_card_id/results/phase2 \
  --backbone tinyvit \
  --embedding-dim 256 \
  --epochs 5 \
  --batch-size 16 \
  --eval-solring
```

## 4) Optional ResNet-50 baseline

```bash
cd 07_phase2
python 02_train_arcface.py \
  --manifest ~/claw/data/ccg_card_id/phase2/manifest.csv \
  --output-dir ~/claw/data/ccg_card_id/results/phase2 \
  --backbone resnet50 \
  --embedding-dim 128 \
  --epochs 5 \
  --batch-size 8 \
  --eval-solring
```

## 5) Standalone retrieval eval from checkpoint

```bash
cd 07_phase2
python 03_eval_retrieval.py \
  --checkpoint ~/claw/data/ccg_card_id/results/phase2/tinyvit_arcface_128/last.pt \
  --manifest ~/claw/data/ccg_card_id/phase2/manifest.csv \
  --out-dir ~/claw/data/ccg_card_id/results/phase2/tinyvit_arcface_128/eval_solring
```

## Outputs

For run dir `~/claw/data/ccg_card_id/results/phase2/<backbone>_arcface_<dim>/`:

- `last.pt` — latest checkpoint
- `train_history.json` — epoch losses
- `eval_solring/retrieval_summary.json` — top-1/top-3/top-10 summary
- `eval_solring/retrieval_summary.csv` — one-row CSV summary
- `eval_solring/retrieval_predictions.jsonl` — per-query predictions

## Notes

- Defaults are CPU/MPS-safe (`num_workers=0`, moderate batch sizes).
- Increase `--batch-size` only if memory allows.
- This scaffold is modular and does not modify existing Phase-1 eval scripts.
