# CCG Card Identification — Welcome

If you're new here, this project answers one practical question:

> Given a real-world photo/video frame of a collectible card, which exact card is it?

That sounds simple until you hit real constraints:
- cards are seen at odd angles and lighting
- sleeves/glare/background clutter are common
- many printings share nearly identical artwork
- one-shot behavior matters (a card may appear once in a stream and still needs identification)

## Pipeline at a glance

1. **01_data_sources/** — ingest canonical card metadata + reference images
2. **02_data_sets/** — build curated train/eval datasets from real captures
3. **03_detector/** — localize card corners/warps in scene images
4. **04_vectorize/** — classical/baseline embeddings (pHash, DINOv2, BRIEF)
5. **04_vectorize/mobilevit_xxs/** — train + evaluate task-specific neural embeddings
6. **05_lookup_db/** — precompute/update vector caches for retrieval serving
7. **06_eval/** — benchmark retrieval quality across datasets

## Two retrieval targets (important)

- **Artwork match**: correct illustration (ignores printing differences)
- **Edition match**: exact printing/card_id (set code + collector number level)

Artwork accuracy can be high while edition accuracy is poor, especially for near-identical reprints.

## Current practical takeaway

On difficult real-world sets (e.g. cluttered backgrounds), higher-bit pHash baselines are still very competitive and can beat current MobileViT-XXS checkpoints. Neural embedding training is promising, but still needs targeted cleanup/tuning for robustness.

## Start here

- Accuracy framing tutorial: `docs/01_accuracy_what_do_you_mean.md`
- Neural embedding training docs: `04_vectorize/mobilevit_xxs/README.md`
- Evaluation runner: `06_eval/04_eval_mobilevit_xxs.py`
- Latest analysis snapshot: `docs/mobilevit_xxs_analysis_2026-03-16.md`
- Research notes: `docs/embedding-model-survey.md`
