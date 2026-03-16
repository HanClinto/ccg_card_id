# Tutorial — Not All Pixels Are Equal (and How to Train for That)

You asked the exact right question:

> How do we train embeddings so subtle, meaningful regions (set symbol, copyright line, reprint markers) matter more than irrelevant noise (glare, signatures, sleeve artifacts)?

This is a core problem in card edition identification.

---

## 1) The problem with equal-vote similarity

Methods like pHash are useful, but they implicitly treat image evidence more uniformly than we want.

For edition ID, not all regions are equally informative:

**High-value regions** (often tiny):
- set symbol area
- collector number / copyright line
- reprint markers (e.g., The List icon)
- frame micro-variations

**Low-value regions** (often large):
- blank background around card
- glare streaks
- sleeve texture
- random occlusion/noise
- player-added marks/signatures

If model attention is not shaped, easy-but-wrong cues dominate.

---

## 2) Your heatmap intuition is strong

Your “overlay cards with same artwork and highlight statistically changing regions” idea is solid and implementable.

Conceptually:
1. Align cards sharing same `illustration_id` (you already have detector/warp infrastructure).
2. Compute per-pixel variance/frequency-difference maps across printings.
3. Turn that into a soft importance prior (heatmap).
4. Train embedding model to respect this prior (not hard mask, soft bias).

This gives the network prior knowledge about where edition signal is likely to live.

---

## 3) Three practical design patterns

## Pattern A — ROI-aware multi-crop embeddings (recommended first)

Use multiple crops per card:
- full card crop (global context)
- set-symbol ROI
- text line ROI (collector number/copyright)
- optional marker ROI

Encode each crop (shared or lightweight separate heads), then fuse embeddings.

Why this is good:
- easy to debug
- explicit control over what matters
- works with small models and edge deployment

---

## Pattern B — Soft spatial weighting inside ViT

Build a per-image importance map and apply it to patch tokens:
- token weight prior from heatmap
- learned gating modifies token contribution
- final embedding is weighted token pooling

This is close to your “internal warping lens” mental model.

Implementation options:
- patch-weighted pooling
- token gating modules
- learned token selection

Related ideas:
- TokenLearner (adaptive token selection): https://arxiv.org/abs/2106.11297
- DynamicViT (token sparsification): https://arxiv.org/abs/2106.02034

---

## Pattern C — Learn geometric focus (deformation/zoom)

Let model learn where/how to zoom/warp:
- Spatial Transformer Networks (differentiable warps)
- Deformable attention mechanisms

References:
- Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
- Deformable DETR: https://arxiv.org/abs/2010.04159

This can be powerful but is more complex to stabilize than Pattern A.

---

## 4) How to teach “what matters” (loss design)

You need objective pressure, not just architecture:

1. **Hard-negative mining by artwork**
   - anchors and negatives share artwork but differ edition.
2. **Region-consistency auxiliary losses**
   - enforce edition-separating behavior in high-value ROIs.
3. **Region-drop augmentations**
   - occasionally corrupt low-value areas so model stops over-relying on them.
4. **Multi-task supervision**
   - embedding loss + small heads for set_code / marker presence.

---

## 5) Suggested roadmap for this repo

## Phase 1 (low risk, high signal)
- Add ROI extraction (set symbol / text line / marker zones) after card warp.
- Train **global+ROI fused embeddings** (Pattern A).
- Evaluate on `solring` + cluttered-background set.

## Phase 2
- Add heatmap prior from same-artwork overlay stats.
- Use heatmap-weighted patch pooling (Pattern B).

## Phase 3 (optional)
- Test deformable focus module (Pattern C) if Phase 2 still misses edition cues.

---

## 6) Why ViT is still a good choice

ViTs are patch-native and naturally compatible with:
- token weighting
- token selection/sparsification
- multi-scale patch focus

So your instinct to use patch-based transformers is correct.
But for this project, start with ROI fusion + hard negatives before adding heavy architectural complexity.

---

## 7) Minimal experiment matrix (to keep this practical)

Compare these on identical splits:

1. Baseline global embedding (today)
2. Global + ROI fused embedding
3. Global + heatmap-weighted token pooling

For each, report:
- Name ID top-1/top-3
- Edition ID top-1/top-3
- Sol Ring stress performance
- p50/p95 latency + memory on target hardware

Pick by **edition gain per millisecond per MB**.

---

## 8) Key references (focus-aware embeddings & ViT-era practice)

- DINOv2 (robust ViT SSL features): https://arxiv.org/abs/2304.07193
- I-JEPA: https://arxiv.org/abs/2301.08243
- SigLIP: https://arxiv.org/abs/2303.15343
- TokenLearner: https://arxiv.org/abs/2106.11297
- DynamicViT: https://arxiv.org/abs/2106.02034
- Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Fine-grained ViT (part selection): https://arxiv.org/abs/2103.07976

---

## Bottom line

Yes: we should explicitly model pixel/patch importance.

The best near-term move is **ROI-aware embeddings + hard negatives + heatmap priors**.
That gives you most of the benefit of “smart focus” without jumping straight into fragile, high-complexity training.
