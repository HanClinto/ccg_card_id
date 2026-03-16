# Neural Embedding Survey for One-Shot Card Identification

This is a practical reading list + playbook focused on **instance-level visual retrieval** for card scanning.

If you felt earlier references were old: fair point. This update emphasizes **ViT-era embedding work (2023+)** and how it maps to this project.

---

## TL;DR (what changed in recent years)

Recent progress is less about “invent one new loss” and more about:
1. **better ViT pretraining recipes** (SSL and VLM)
2. **stronger curated data pipelines**
3. **scaling laws + compute-efficient tuning**
4. **embedding flexibility at deployment time** (e.g., variable dimensions)

For this repo, that means: keep ArcFace fine-tuning, but initialize from stronger ViT features and evaluate hard-split robustness, not just easy-set top-1.

---

## Recent ViT embedding directions worth studying

## 1) DINOv2 (2023) — robust all-purpose visual features
- Paper: **Learning Robust Visual Features without Supervision**
  - https://arxiv.org/abs/2304.07193
- Why relevant:
  - Strong frozen features across many tasks
  - Emphasizes curated large-scale data and stable ViT SSL training
- Practical use here:
  - Excellent backbone/init candidate for card embedding fine-tuning
  - Good baseline for "frozen encoder + light projection head"

## 2) I-JEPA (2023) — semantic predictive SSL for ViTs
- Paper: **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**
  - https://arxiv.org/abs/2301.08243
- Why relevant:
  - Learns semantic representations without heavy handcrafted augmentations
  - Predictive joint-embedding objective may transfer well to retrieval
- Practical use here:
  - Useful conceptual alternative to contrastive-only pretraining

## 3) SigLIP (2023) — better pairwise contrastive behavior for vision-language embeddings
- Paper: **Sigmoid Loss for Language Image Pre-Training**
  - https://arxiv.org/abs/2303.15343
- Why relevant:
  - Pairwise sigmoid loss decouples from global softmax normalization
  - Often strong embedding quality with scalable training
- Practical use here:
  - Strong open-weight embedding candidates for initialization (via SigLIP-family checkpoints)

## 4) EVA-CLIP (2023) — improved CLIP-scale training recipes
- Paper: **Improved Training Techniques for CLIP at Scale**
  - https://arxiv.org/abs/2303.15389
- Why relevant:
  - Recipe-level gains (optimization/augmentation/training strategy)
  - Shows how much performance is recipe+data, not just architecture
- Practical use here:
  - Borrow recipe ideas when training custom ViT embedding pipelines

## 5) DataComp (2023) — data curation matters as much as model details
- Paper: **In search of the next generation of multimodal datasets**
  - https://arxiv.org/abs/2304.14108
- Why relevant:
  - Systematically studies dataset construction impact for CLIP-like training
- Practical use here:
  - Reinforces our need for careful hard-case dataset curation and split design

## 6) Matryoshka Representation Learning (MRL)
- Paper: **Matryoshka Representation Learning**
  - https://arxiv.org/abs/2205.13147
- Why relevant:
  - One model supports multiple embedding sizes at inference
  - Great fit for edge constraints (Pi/mobile/browser)
- Practical use here:
  - Consider variable-dimension embeddings (e.g., 32/64/128) from one trained model

---

## Still-useful “older” foundations (keep, don’t discard)

These are older but still operationally important:
- ArcFace: https://arxiv.org/abs/1801.07698
- Proxy Anchor: https://arxiv.org/abs/2003.13911
- SupCon: https://arxiv.org/abs/2004.11362
- Triplet mining practice: https://arxiv.org/abs/1703.07737

Reason: your downstream problem is **fine-grained instance retrieval**, where these objectives remain strong and interpretable.

---

## What tends to work for custom ViT embedding models

1. Start from a strong pretrained ViT encoder (DINOv2 / SigLIP-family / CLIP-family)
2. Add task-specific embedding head + metric objective (ArcFace or SupCon/Proxy variants)
3. Mine hard negatives (same artwork, different edition)
4. Train with realistic capture augmentations (glare/warp/compression/background)
5. Select models on hard-split metrics (not easy-set saturation)
6. Deploy with compact embeddings + exact-search or shortlist+rereank strategy

---

## What still fails in practice

1. Random negatives only
2. Selecting by easy validation only
3. Assuming generic foundation embedding is enough for edition-level disambiguation
4. Ignoring deployment-time memory/latency (especially for 1GB Pi target)

---

## Recommended experiment matrix for this repo (ViT-focused)

Run a small, disciplined matrix first:

- **Backbone/init:**
  - MobileViT-XXS baseline
  - DINOv2-small frozen + train head
  - DINOv2-small full/partial fine-tune

- **Objective:**
  - ArcFace
  - SupCon

- **Embedding dim:**
  - 64, 128

- **Eval:**
  - Name ID + Edition ID
  - Include `solring` stress test + cluttered background split
  - Report p50/p95 latency + memory on target hardware

Pick winner by **edition accuracy on hard split per millisecond per MB**, not by one aggregate score.

---

## Bottom line

You’re right to push toward newer ViT literature. The modern edge is:
- better pretrained ViT features,
- better data curation,
- better evaluation discipline,
- and deployment-aware embeddings.

ArcFace vs Triplet is still useful, but now it should be framed as the *fine-tuning layer* on top of stronger ViT-era representation foundations.
