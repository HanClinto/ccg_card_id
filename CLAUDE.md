# CCG Card ID — Project Conventions

## Data directory

All data lives under a single root resolved at runtime from the `CCG_DATA_DIR`
environment variable (fallback: `~/claw/data/ccg_card_id/`). Code always reads
this via `cfg.data_dir`:

```python
from ccg_card_id.config import cfg
path = cfg.data_dir / "datasets" / "solring" / ...
```

Never hardcode absolute paths in source files.

## Path storage in manifests, CSVs, and JSON

**All `image_path` and file-path fields in manifests, CSVs, and JSON cache
files must be stored as paths relative to `cfg.data_dir`**, not as absolute
paths. This makes files portable across machines and drives with different
directory structures.

**Writing** — strip `cfg.data_dir` before saving:

```python
relative = path.relative_to(cfg.data_dir)
```

**Reading** — resolve against `cfg.data_dir` before use:

```python
absolute = cfg.data_dir / row["image_path"]
```

> **Current state:** existing manifests (`artwork_id_manifest.csv`,
> `query_manifest.csv`, `munchie/manifest.csv`, etc.) still contain absolute
> paths. Migrate them as they are regenerated — do not bulk-rewrite existing
> files unless that is the explicit task.

## Card identity

Reference cards by **Scryfall UUID** (`card_id`), not by local filename or
display name. Filenames for captured images should embed the UUID so it can be
recovered with a regex without consulting any external index:

```
{card_uuid}_{HumanName}_{date}.mp4-{frame}.jpg
```

Extract with:

```python
import re
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
card_id = UUID_RE.search(filename).group(0).lower()
```

Do not use `filename.split("_")[0]` — this only captures the first segment of
a hyphenated UUID.

## Python environment

Use the project virtualenv at `.venv312/` (Python 3.12). Activate with:

```bash
source .venv312/bin/activate
```

## Code layout

```
ccg_card_id/          package — config, shared utilities
02_data_sets/         per-dataset processing pipelines
  sol_ring/
  clint_cards_with_backgrounds/
  munchie/
  daniel_scans/
04_build/
  mobilevit_xxs/      training scripts, models, data loading
    01_build_manifest.py
    02_train.py              single-task ArcFace (illustration_id)
    03_train_multitask.py    multi-task ArcFace (illustration_id + set_code)
    models.py / data.py / retrieval.py
06_eval/
  04_eval_mobilevit_xxs.py   evaluation harness
```

## Key conventions

- **Manifests** are CSV files with at minimum: `image_path`, `card_id`,
  `illustration_id`. `image_path` must be relative to `cfg.data_dir`.
- **Vector caches** live under `cfg.data_dir / "vectors" / ...` as `.npz` files.
- **Eval reports** are written to `cfg.data_dir / "results" / "eval" /
  {timestamp}/`.
- **Training checkpoints** live under `cfg.data_dir / "results" /
  "mobilevit_xxs" / {run_dir}/`.
- **Seed checkpoints** (well-trained intermediates for transfer learning) are
  copied to `cfg.data_dir / "results" / "mobilevit_xxs" / "seeds" /`.
