# Multi-Task Training Strategy + Edge Backbone Shortlist

This note answers two practical questions:

1. How do we train **one embedding** that works for both:
   - artwork/name separation, and
   - edition-level reprint separation?
2. Which backbone families should we test first under a 1GB Pi deployment constraint?

---

## Part A — One embedding for both artwork and edition

## Recommended training pattern (for this repo)

Use a **shared backbone + shared projection embedding + multi-task losses**:

- one embedding `z` (e.g., 128d) used at inference
- losses:
  - `L_artwork` (illustration_id supervision)
  - `L_edition` (card_id or set+collector proxy supervision)
  - optional `L_language`

Total loss:

`L = w1*L_artwork + w2*L_edition (+ w3*L_language)`

This preserves a single lookup embedding while forcing it to encode both coarse and fine distinctions.

---

## Curriculum that usually works better than static mixing

### Stage 1: global semantics first
- emphasize artwork/name discrimination
- strong inter-card negatives

### Stage 2: fine-grained push
- increase weight of edition objective
- mine hard negatives within same artwork/name family

### Stage 3: joint stabilization
- balanced loss weights
- continue mixed hard-negative mining

This is close to your intuition: solve broad separation, then focus on reprint deltas.

---

## Hard-negative mining policy (practical)

Use a mixed sampler each batch:
- 40% easy negatives (different card/artwork)
- 40% medium negatives (similar-looking across cards; pHash-near)
- 20% hard negatives (same artwork/name, different edition)

Then anneal toward more hard negatives over time.

---

## ArcFace vs Triplet (why ArcFace often trained better here)

Your recollection is directionally right. Triplet loss is not “bad,” but it is often harder to make stable in this exact setting.

Why Triplet can struggle in this domain:
- **Combinatorial sampling problem**: training quality depends heavily on which triplets are mined.
- **Many weak/easy triplets** contribute little gradient once broad separation is learned.
- **Over-focus risk**: aggressive hard mining can destabilize early training.
- **Single-image-per-edition issues**: for fine edition labels, positives can be sparse/weakly diverse.

Why ArcFace often behaves better for this repo:
- **Class-prototype supervision** gives denser, more stable gradients than sparse triplet comparisons.
- **Angular margin** explicitly enforces inter-class separation in normalized embedding space.
- **Easier optimization plumbing**: fewer moving parts than maintaining robust triplet mining policy end-to-end.
- **Good fit for hierarchical targets** when paired with multi-task heads (artwork + edition/set proxy + language).

Practical interpretation:
- Use ArcFace (or another class-margin objective) as the default “workhorse.”
- Add triplet/contrastive objectives as optional auxiliary losses once baseline is stable.

---

## Common misconceptions to avoid

1. **“One loss should solve everything.”**
   - In practice, artwork and edition often need different supervisory pressure.

2. **“If name accuracy is high, edition should follow.”**
   - Not true; edition cues are frequently tiny and localized.

3. **“More hard negatives is always better.”**
   - Too many too early can hurt convergence; curriculum matters.

4. **“card_id ArcFace directly is always ideal.”**
   - With near one-sample-per-class patterns, class-margin objectives can become brittle; use proxies (set_code, artwork families, grouped objectives) when needed.

5. **“Bigger model always wins.”**
   - Under 1GB Pi constraints, deployment efficiency is part of the objective, not an afterthought.

---

## Additional tactics worth adding

- **Proxy label design for edition**: combine set_code + collector_number buckets or hierarchical proxies to avoid extreme one-shot class sparsity.
- **Multi-positive construction**: synthetically generate stronger positive diversity (view augmentations, ROI crops, mild photometric variants).
- **Loss scheduling**: warm-start with higher artwork weight; ramp edition/language weights over epochs.
- **Teacher-student distillation**: train small deploy model against stronger teacher embeddings (DINOv2/SigLIP lane).
- **Calibration by task**: separate thresholds/confidence handling for name vs edition predictions.

---

## Part B — Backbone shortlist to test

Below are strong, well-known candidates with rough scale and practical fit.
(Exact params vary by variant; values below are representative.)

## Tier 1 (test first)

1. **MobileViT-XXS**
- Release: 2021 (ICLR 2022 paper)
- Params: ~1.3M (timm variant)
- Why try: current baseline, excellent edge footprint
- Risk: may miss tiny edition cues vs stronger backbones

2. **MobileViTv2 (small variants)**
- Release: 2022
- Params: ~1.5M–3.5M (variant-dependent)
- Why try: mobile-oriented update with better speed/latency behavior
- Risk: still capacity-limited for hardest edition distinctions

3. **EdgeNeXt-XXS / XS**
- Release: 2022
- Params: ~1.3M (XXS), ~5.6M (larger small variant)
- Why try: mobile hybrid CNN+Transformer, good efficiency/accuracy profile
- Risk: less ecosystem maturity than DeiT/ViT families

4. **EfficientFormerV2 (S0/S1 class)**
- Release: 2022
- Params: roughly MobileNet-size class (few million)
- Why try: explicitly optimized for mobile speed + size
- Risk: must validate feature quality for fine-grained retrieval (not just classification)

5. **LeViT (small variants e.g., 128S)**
- Release: 2021
- Params: ~8M class (variant-dependent)
- Why try: very strong speed/accuracy trade-off at inference
- Risk: older family; may underperform newer SSL-pretrained ViT features

## Tier 2 (quality lane / teacher lane)

6. **DeiT-Tiny / DeiT-Small**
- Release: 2020/2021
- Params: ~5.7M (Tiny), ~22M (Small)
- Why try: stable, well-understood ViT baselines
- Risk: not mobile-optimized by default

7. **TinyViT (smallest variants)**
- Release: 2022
- Params: variant-dependent (smallest still generally above ultra-tiny mobile class)
- Why try: strong distilled small-transformer family
- Risk: may still be heavy for strict 1GB Pi throughput targets

8. **DINOv2 ViT-S/14 backbone**
- Release: 2023
- Params: ~21M
- Why try: very strong pretrained visual features; likely large boost in retrieval quality
- Risk: heavy for edge; may be best as teacher or non-Pi profile

9. **SigLIP ViT-B-class embeddings**
- Release: 2023
- Params: typically much larger than tiny-edge models
- Why try: strong modern embedding priors
- Risk: usually too heavy for direct Pi runtime; excellent teacher/init candidate

10. **ConvNeXtV2-Atto/Pico (optional non-transformer control)**
- Release: 2023
- Params: very small variants exist
- Why try: strong compact baseline control; checks whether transformer complexity is necessary
- Risk: not patch-token native for advanced token weighting ideas

---

## What is likely best for your constraints

If target is ~5 FPS on 1GB Pi and high edition sensitivity:

- **Primary deploy candidates:**
  - MobileViTv2 small variant
  - EdgeNeXt-XXS/XS
  - EfficientFormerV2-S0/S1

- **Quality teacher / upper-bound candidates:**
  - DINOv2 ViT-S
  - SigLIP ViT-B family

Use teacher distillation into small deploy model if quality gap is large.

---

## Dataset + augmentation assumptions (lock these)

Training source assumptions:
- Train primarily on **aligned images from the packopening dataset**.
- Keep legacy sets (`solring`, `clint_backgrounds`, `daniel`, `munchie`) as evaluation/reporting sets.
- Two storage tiers are available:
  - **Primary/external cache** (large, slower): `cfg.data_dir`
  - **Fast SSD cache** (small, faster): `cfg.fast_data_dir`
- Prefer keeping active training images (`packopening/aligned_448`) in fast SSD cache when space allows.
- Resized cache folders must include explicit size suffixes (e.g., `aligned_448`, `aligned_320`) to avoid ambiguous names like plain `aligned`.
- Apply pHash quality filtering for packopening training rows to suppress bad matches.
  - Starting range: **max phash_dist 10–15** (default initial setting: 15).

Augmentation assumptions:
- Use **light augmentation** only (data is already real-camera and noisy).
- **Never** use horizontal flip.
- **Never** use vertical flip.
- Rotation, if used, should be **180° only** (0°/180°).
- Treat 180° rotation as an explicit experiment toggle (not guaranteed default).

Image size policy (locked for bakeoff v1):
- **Default train/eval resolution: 448×448** (quality-first for tiny edition cues).
- Optional fallback check: run winner at **320×320** if deployment pressure requires it.
- 224×224 is speed-floor only (not primary decision setting).

Rationale:
- Avoid introducing unrealistic transforms that dilute edition-critical cues.
- Encourage possible 180° rotation robustness without corrupting card-layout semantics.
- Preserve small but important edition markers (set symbols, micro text, reprint icons).

## Suggested experiment plan (minimal but informative)

1. Keep current MobileViT-XXS baseline.
2. Add 3 challengers: MobileViTv2, EdgeNeXt, EfficientFormerV2.
3. Train with shared embedding + two-task losses (artwork + edition).
4. Use curriculum + staged hard-negative mining.
5. Evaluate on:
   - normal sets
   - `solring` stress set
   - cluttered-background set
6. Track:
   - Name ID top-1/top-3
   - Edition ID top-1/top-3
   - latency + memory on target device

Select by edition gain per ms per MB, not just aggregate top-1.

---

## Reference links (starting points)

- MobileViT: https://arxiv.org/abs/2110.02178
- MobileViTv2: https://arxiv.org/abs/2206.02680
- EdgeNeXt: https://arxiv.org/abs/2206.10589
- EfficientFormerV2: https://arxiv.org/abs/2212.08059
- LeViT: https://arxiv.org/abs/2104.01136
- TinyViT: https://arxiv.org/abs/2207.10666
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: https://arxiv.org/abs/2303.15343
- Example HF model card (MobileViT-XXS stats): https://huggingface.co/timm/mobilevit_xxs.cvnets_in1k
