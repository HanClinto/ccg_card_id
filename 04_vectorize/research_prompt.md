# Research Prompt — Edge-Ready Card Embedding Model for Name + Edition Identification

You are helping evaluate model architecture and training strategy for a collectible card scanner.

## Objective
Find practical model/training options that improve **both**:
1. **Card Name / Artwork Identification** (e.g., Sol Ring vs Giant Growth)
2. **Edition Identification** (e.g., C18 Sol Ring vs PLST Sol Ring)

We want a **single embedding** usable for both tasks at retrieval time.

## Key constraints
- Deployment target: **Raspberry Pi 1GB RAM** (edge-first)
- Throughput target: ~**5 FPS is acceptable** (not required to be super fast)
- Must work on noisy real captures (glare, blur, background clutter)
- Must detect very small but important visual cues:
  - set symbol
  - collector/copyright text
  - reprint markers (e.g., The List icon)
- We do **not** want retraining for every new release; prefer embedding retrieval pipelines where new cards are indexed by vectorization.

Augmentation policy constraints:
- Training data will be **aligned packopening images**.
- Keep augmentation conservative (real camera noise already present).
- **Do not use HFlip or VFlip.**
- If rotation augmentation is used, allow **180° only** (0°/180°), and evaluate as a separate ablation for rotation robustness.

Image-size policy constraints:
- Default training/evaluation resolution should be **448×448**.
- Add a fallback run at **320×320** only if compute/latency pressure requires it.
- Treat **224×224** as speed-floor baseline, not primary setting.

Storage/cache constraints:
- There are two storage tiers:
  - larger/slower external cache (`cfg.data_dir`)
  - limited-space fast SSD cache (`cfg.fast_data_dir`)
- Prefer storing active training images in SSD cache as `datasets/packopening/aligned_448/...`.
- Any resized cache directory must include size suffix in folder naming (`aligned_448`, `aligned_320`, etc.).

## Why this is hard (domain-specific challenges)
- Many edition-level differences are **tiny but business-critical**.
- The model must ignore lots of non-semantic variation (glare, sleeves, background clutter, blur) while still catching minute edition cues.
- Important cues often live in small regions:
  - set symbol shape/style
  - collector number / copyright line typography
  - small reprint markers (e.g., The List planeswalker icon)

### Concrete example: PLST vs C18 Sol Ring
These two printings can look very similar at a glance, but edition correctness still matters:
- C18 Sol Ring: https://scryfall.com/card/c18/222/sol-ring
- PLST Sol Ring: https://scryfall.com/card/plst/C18-222/sol-ring

Direct image links:
- C18 image: https://cards.scryfall.io/png/front/f/0/f082f4f4-c08a-42e5-8d70-53315e757d43.png
- PLST image: https://cards.scryfall.io/png/front/2/f/2f434f7a-2f20-4db3-bdc7-d152295636c8.png

This is a representative failure mode: wrong edition on a low-price card can still create expensive customer-service/reputation costs.

### Additional comparison cases to consider

1) **Alpha vs Beta (same card, early-print differences)**
- Alpha Island (LEA): https://scryfall.com/search?q=set%3Alea+name%3Aisland+unique%3Acards&unique=cards
- Beta Island (LEB): https://scryfall.com/search?q=set%3Aleb+name%3Aisland+unique%3Acards&unique=cards

2) **Revised vs 4th Edition (same card across adjacent core sets)**
- Revised Lightning Bolt (3ED): https://scryfall.com/search?q=set%3A3ed+name%3A%22lightning+bolt%22+unique%3Acards&unique=cards
- 4th Edition Lightning Bolt (4ED): https://scryfall.com/search?q=set%3A4ed+name%3A%22lightning+bolt%22+unique%3Acards&unique=cards

Use these as additional sanity-check pairs when evaluating whether a method is truly learning edition-level cues rather than just coarse card identity.

## Current baseline
- MobileViT-XXS embedding pipeline
- Multi-task concept: shared backbone with artwork + edition objectives
- Strong stress benchmark: `solring` dataset (20+ editions with near-identical artwork)

## What we need from you
Provide a ranked shortlist of architecture + training strategies suitable for this use case, with evidence.

### Evaluate and compare
- MobileViT variants (XXS / v2)
- Edge-optimized ViTs/hybrids (e.g., EdgeNeXt, EfficientFormerV2, LeViT, TinyViT)
- Strong teacher/upper-bound embedding backbones (e.g., DINOv2/SigLIP)
- Focus-aware methods (ROI fusion, token weighting/sparsification, attention to informative regions)

### Specifically answer
1. Best candidates for **on-device deployment** (1GB Pi)
2. Best candidates for **teacher/distillation** to a small deploy model
3. Best training setup for **single embedding, dual utility** (artwork + edition)
   - multi-task loss weighting and/or curriculum
   - hard-negative mining strategy (especially same-artwork/different-edition)
4. Whether focus-aware methods (heatmaps/ROI/token weighting) materially help edition accuracy
5. Expected quality/latency/memory tradeoffs

## Deliverable format
For each recommended option, include:
- Model name + paper/year + reference link (paper + code/model source)
- Approx model size/params and expected inference profile
- Why it fits (or doesn’t) for our constraints
- Suggested experiment configuration
- Risks/failure modes

Then provide:
- **Top 3 deploy-now candidates**
- **Top 2 teacher candidates**
- A minimal experiment matrix to test quickly

## Success criteria
Recommendations should optimize **edition accuracy on hard reprint cases** while staying viable for edge deployment.

Prefer concrete, reproducible, and pragmatic suggestions over generic model lists.
