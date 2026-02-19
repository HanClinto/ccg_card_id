# 06_eval

Evaluation scripts for Sol Ring retrieval tests.

## Scripts

- `01_eval_retrieval.py`: perceptual-hash retrieval evaluation (`phash`, `whash_db4`, etc.)
- `02_eval_dinov2.py`: DINOv2 embedding retrieval evaluation

Both scripts still print terminal summaries and now also persist run artifacts by default.

## Output artifacts

By default, each run writes to:

`06_eval/results/<timestamp>/`

You can override the root or run id with CLI flags.

Each run directory contains at least:

- `summary.csv` (columns: `algorithm_variant,topk,correct,total,accuracy`)
- `summary.json`
- `failures.jsonl` (one miss per line, includes image path, true/pred IDs, score)
- `overview.md` (cross-variant accuracy comparison)
- Per-algorithm markdown files, e.g.:
  - `phash_64.md`
  - `dinov2_small.md`

Markdown reports include a worst-case failure table with:

- image path
- true id
- predicted id
- score
- true rank (when available)

## Usage examples

```bash
cd 06_eval

# Hash retrieval eval
python 01_eval_retrieval.py --top-k 1 3 10
python 01_eval_retrieval.py --methods phash --sizes 64 128 --top-k 1 3 10
python 01_eval_retrieval.py --output-root ./results --run-id phash_sanity

# DINOv2 eval
python 02_eval_dinov2.py --models small --top-k 1 3 10
python 02_eval_dinov2.py --output-root ./results --run-id dinov2_small_top10

# Disable artifact writing (terminal summary only)
python 02_eval_dinov2.py --no-write-results
```

## New reporting flags

Both scripts support:

- `--output-root PATH`: root folder for run directories
- `--run-id NAME`: explicit run directory name (default is timestamp)
- `--worst-n N`: number of failures shown in markdown worst-case tables
- `--no-write-results`: skip writing result artifacts

## Centralized historical/latest CSVs

In addition to per-run folders, each run updates two shared files under `--output-root`:

- `history_results.csv`: append-only log of every run record (`run_at,run_id,benchmark,dataset,algorithm_variant,topk,correct,total,accuracy`)
- `latest_results.csv`: one most-recent row per unique `(benchmark, dataset, algorithm_variant, topk)`

This gives you both full history and a clean "current leaderboard" view.
