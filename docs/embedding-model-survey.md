# Neural Embedding Survey for One-Shot Card Identification

This is a practical reading list + playbook focused on **instance-level visual retrieval** (very close to the card-ID problem).

## What tends to work well

### 1) Metric learning with strong mining and large batches
- **ArcFace (additive angular margin)** — great identity separation with stable training.
  - https://arxiv.org/abs/1801.07698
- **Proxy Anchor Loss** — easier optimization for retrieval than pair/triplet-only objectives.
  - https://arxiv.org/abs/2003.13911
- **Supervised Contrastive Learning (SupCon)** — robust embeddings when labels are reliable.
  - https://arxiv.org/abs/2004.11362

Why it helps here: many cards are near-duplicates. Angular-margin + hard negatives gives cleaner decision boundaries.

### 2) Hard-negative sampling is not optional
- Sample negatives that share artwork/theme/layout but differ true card_id/edition.
- Keep mining active throughout training, not only at startup.

Related retrieval literature:
- In Defense of the Triplet Loss for Person Re-Identification (batch-hard mining)
  - https://arxiv.org/abs/1703.07737

### 3) Domain-specific augmentations
- Perspective warp, sleeve glare, camera noise, blur, compression, background clutter.
- If training crops are too clean, deployment mismatch will dominate.

### 4) Two-stage retrieval often beats pure embedding-only systems
- Stage A: fast ANN over embeddings (top-k candidates)
- Stage B: rerank using extra cues (OCR text, set symbol classifier, corner metadata)

Reference pattern (industry-scale visual search):
- DeepFashion retrieval (landmark + retrieval ideas)
  - https://arxiv.org/abs/1605.01354

## What often does NOT work (or disappoints)

1. **Only random negatives**
   - model looks good on easy eval, fails on confusing editions.

2. **Single-objective training for multi-objective problem**
   - illustration-only labels won't give strong edition discrimination.

3. **Small backbone + small embedding + no reranking**
   - compact and fast, but may throw away tiny distinguishing cues.

4. **Benchmarking mostly on easy splits**
   - can hide failure modes until late.

5. **Assuming pretrained features transfer perfectly**
   - generic ImageNet-style pretraining helps but does not solve instance-level card distinctions by itself.

## Practical recommendations for this repo

1. Keep MobileViT-XXS as a speed baseline, but add at least one stronger backbone lane for comparison.
2. Add explicit same-artwork/different-edition hard negatives.
3. Evaluate with a fixed hard split (e.g., cluttered backgrounds) as the primary model-selection metric.
4. Add reranking stage (OCR/set-symbol/collector-number) for top-20 candidates.
5. Log calibration curves and nearest-neighbor confusion groups, not just top-k.

## Additional useful references

- FaceNet (classic metric learning baseline)
  - https://arxiv.org/abs/1503.03832
- SimCLR (contrastive pretraining ideas)
  - https://arxiv.org/abs/2002.05709
- CLIP (zero-shot embedding transfer; useful baseline, usually not enough alone for fine print-level IDs)
  - https://arxiv.org/abs/2103.00020
- Image retrieval with DELG (local + global descriptors)
  - https://arxiv.org/abs/2001.05027
