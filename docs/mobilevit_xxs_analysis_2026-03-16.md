# MobileViT-XXS Neural Hash Review (2026-03-16)

## Scope

Quick audit of the current MobileViT-XXS embedding pipeline and recent eval outputs.

## Structural status (after naming cleanup)

Canonical locations:
- `04_vectorize/mobilevit_xxs/` — neural embedding training + retrieval utilities
- `05_lookup_db/` — lookup-db staging (currently mostly placeholder)


Updated references/imports used by build/eval scripts:
- `05_lookup_db/01_precompute_mobilevit_vectors.py`
- `05_lookup_db/02_update_gallery_vectors.py`
- `06_eval/04_eval_mobilevit_xxs.py`

## Performance snapshot (from existing eval artifacts)

Primary file inspected:
- `/Volumes/carbonite/claw/data/ccg_card_id/results/eval/20260312_030642/summary.csv`

### Top-1 Artwork accuracy highlights

- **clint_backgrounds**
  - `phash_32x32_1024bit`: **0.749**
  - `phash_16x16_256bit`: **0.728**
  - best MobileViT (`illustration_id_e75_128d`): **0.654**

- **solring**
  - `phash_16x16_256bit`: **1.000**
  - `phash_32x32_1024bit`: **1.000**
  - best MobileViT (`illustration_id_e75_128d`): **0.896**

- **daniel / munchie**
  - many variants are near-perfect or perfect, suggesting easier domain conditions and weaker stress-testing signal.

## Likely reasons MobileViT underperforms in hard cases

1. **Domain gap dominates**
   - training is likely biased toward cleaner/aligned crops relative to cluttered backgrounds and capture artifacts.

2. **Objective mismatch**
   - illustration-only optimization helps artwork retrieval but not fine-grained edition separation.

3. **Resolution/detail bottleneck**
   - MobileViT-XXS + compact embedding dims can discard weak edition cues.

4. **Hard-negative pressure likely insufficient**
   - near-duplicate printings need persistent same-artwork/different-edition hard-negative mining.

5. **Eval skew toward easier sets**
   - near-100% subsets reduce sensitivity for model selection.

## Recommended next experiments (priority order)

1. Hard-negative curriculum by construction.
2. Multi-task objective with explicit edition-aware signal.
3. Stronger capture-realistic augmentations (glare/perspective/blur/clutter).
4. Backbone sweep beyond XXS with accuracy-per-ms reporting.
5. Retrieval recipe tuning (cosine/L2, calibration, rerank features).
6. Lock one hard split as primary model-selection benchmark.
