# 04_vectorize — Card Embeddings for One-Shot Identification

This stage answers the core runtime question:

> Given one card image from a camera, what known card in the catalog is it most similar to?

Design target:
- **Raspberry Pi (1GB RAM)**
- mobile phones
- web browser inference
- dedicated embedded scanner/sorter devices

---

## Why not just use a classifier?

A fixed-class classifier is usually the wrong product fit here:
- output is tied to a fixed class list
- adding new cards tends to require retraining/fine-tuning
- head size grows with class count

For card ID, we prefer **embedding retrieval**:
- model outputs a compact vector
- lookup is nearest-neighbor search in a vector table/index
- adding new cards usually means vectorize-and-append, not retrain

---

## One-shot in practical terms

In this project, one-shot-ish behavior means:
- we may only have one canonical gallery image per new card
- real-world queries are noisy (angle, blur, glare, background clutter)
- we still want reliable lookup **without retraining every release**

Workflow:
1. Train a reusable embedding model.
2. Vectorize new catalog cards.
3. Add vectors to lookup store.
4. Keep serving.

---

## Why keep pHash/dHash baselines?

Because on edge hardware they are tiny, fast, and often competitive.

References:
- Hacker Factor — pHash: https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
- Hacker Factor — dHash: https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

---

## Why ArcFace as default over plain Triplet Loss?

- Triplet can work very well, but is highly sensitive to mining/batch construction.
- ArcFace generally gives more stable identity-style separation with cleaner margin control.

References:
- ArcFace: https://arxiv.org/abs/1801.07698
- FaceNet (Triplet): https://arxiv.org/abs/1503.03832
- Batch-hard triplet practice: https://arxiv.org/abs/1703.07737
- Proxy Anchor: https://arxiv.org/abs/2003.13911
- SupCon: https://arxiv.org/abs/2004.11362

---

## Contents

- `phash/` — perceptual hash vectors
- `dinov2/` — DINOv2 embeddings
- `brief/` — BRIEF descriptors
- `mobilevit_xxs/` — neural embedding training + retrieval utilities

For practical deployment details, see:
- `docs/04_vectorize_part2_edge_deployment.md`
