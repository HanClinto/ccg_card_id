# Part 2 — From Zero to a Working Retrieval Model on a 1GB Raspberry Pi

This is a practical tutorial for getting a **real** card-identification pipeline running under tight memory and compute constraints.

Design target:
- Raspberry Pi with **1GB RAM**
- same design should scale *up* to mobile/web/embedded scanners

---

## 0) Success criteria first (before writing code)

For edge devices, define all three up front:

1. **Quality target** (example)
   - artwork top-1 >= 0.90 on your chosen hard set
2. **Latency target** (example)
   - <= 250ms median per query on-device
3. **Memory budget** (example)
   - <= 350MB RSS for inference + index + process overhead

If you don’t set these first, optimization work drifts.

---

## 1) Architecture that fits in 1GB

Use a two-track system:

- **Track A (always-on baseline):** pHash/dHash retrieval
  - tiny memory footprint
  - very fast and robust for near-duplicate cases
- **Track B (neural rerank):** compact embedding model (MobileViT-XXS)
  - higher semantic robustness
  - applied to top-k candidates or full ANN depending on budget

Why this works:
- baseline guarantees responsive low-power behavior
- neural path improves hard cases without forcing full expensive search each time

---

## 2) Memory math you should do early

### Embedding store size (float32)

Formula:

`bytes = num_cards × embedding_dim × 4`

Example with 120,000 cards:
- 64d: 120000 × 64 × 4 = **30.7 MB**
- 128d: 120000 × 128 × 4 = **61.4 MB**
- 320d: 120000 × 320 × 4 = **153.6 MB**

### Quantized store (uint8/int8)

Formula:

`bytes = num_cards × embedding_dim × 1`

For 120,000 cards:
- 128d int8: **15.4 MB**

This is why compact embeddings + quantization are usually worth it on Pi.

---

## 3) Latency budget template

Break query time into pieces:

- image decode/crop/normalize
- model forward pass
- ANN search or brute-force distance
- optional rerank + result formatting

Example target budget (250ms total):
- preprocess: 40ms
- inference: 90ms
- search: 70ms
- rerank/post: 50ms

You can’t improve what you don’t measure—log each stage separately.

---

## 4) Start simple: baseline-first rollout

### Step A — establish pHash baseline

- Build pHash vectors
- Evaluate on hard split and easy split
- Record top-1 + top-3 + latency

If pHash already solves a large fraction, keep it in production as fallback.

### Step B — add neural embeddings

- train MobileViT-XXS embedding model
- run eval with same datasets and reporting format
- compare **accuracy-per-millisecond**, not just accuracy

### Step C — introduce hybrid mode

- use pHash to shortlist top-N
- neural rerank those N candidates
- tune N for speed/accuracy tradeoff

---

## 5) Training recipe tuned for edge deployment

1. Keep model small first (XXS/tiny class)
2. Prefer 64d–128d embeddings
3. Use ArcFace-style margin objective as baseline
4. Add hard negatives early (same-artwork, different-edition)
5. Train/eval on at least one truly hard real-world split
6. Stop training when hard-split quality plateaus (avoid overfitting easy sets)

---

## 6) ArcFace vs Triplet in edge context

For this project profile:
- ArcFace is usually easier to converge reliably
- Triplet can be excellent but is highly sensitive to mining strategy

If engineering time is limited, ArcFace is often the faster path to dependable results.

A practical strategy:
- ArcFace baseline first
- only test Triplet/SupCon/Proxy losses once ArcFace baseline is stable and fully measured

---

## 7) Dataset strategy for one-shot card ID

You need **three** evaluation bands:

1. **clean/reference-like**
2. **typical real capture**
3. **hard clutter/glare/angle** (primary model-selection set)

Do not select models on easy splits alone; they can mask real deployment failures.

---

## 8) Deployment profiles

### Profile P1 — Pi safe mode (lowest risk)
- pHash/dHash only
- optional lightweight metadata filters
- best for guaranteed responsiveness

### Profile P2 — Pi hybrid (recommended)
- pHash shortlist + neural rerank
- balanced accuracy and speed

### Profile P3 — mobile/web accelerated
- neural full retrieval + optional rerank
- possible when device has stronger CPU/GPU/WebGPU

Use one code path with config flags; avoid maintaining separate logic trees per platform.

---

## 9) Operational checklist

Before shipping any model/index update:

- [ ] New cards added without retraining path verified
- [ ] Hard-split accuracy regression check passed
- [ ] Memory footprint measured on target device
- [ ] p50/p95 latency measured on target device
- [ ] Fallback path (pHash) still functional
- [ ] Versioned model + index manifest saved

---

## 10) Common failure modes

1. **Classifier creep**
   - team drifts toward closed-set training and frequent retrains
2. **Overfitting to easy validation**
   - looks great offline, fails in camera reality
3. **No stage-level profiling**
   - impossible to fix latency bottlenecks
4. **Unbounded embedding sizes**
   - quality gains too small for memory/latency cost
5. **No baseline fallback**
   - system degrades badly on weak hardware

---

## 11) References

Perceptual hashing:
- Hacker Factor — Looks Like It (pHash):
  - https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
- Hacker Factor — Kind of Like That (dHash):
  - https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

Metric learning / retrieval:
- ArcFace: https://arxiv.org/abs/1801.07698
- FaceNet (Triplet): https://arxiv.org/abs/1503.03832
- Batch-hard triplet practice: https://arxiv.org/abs/1703.07737
- Proxy Anchor Loss: https://arxiv.org/abs/2003.13911
- Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362

---

## 12) Suggested next experiment plan (small + useful)

1. Lock one hard benchmark set as primary.
2. Compare:
   - pHash 256-bit
   - pHash 1024-bit
   - MobileViT-XXS 64d
   - MobileViT-XXS 128d
   - Hybrid shortlist+rerank (N in {20, 50, 100})
3. Report for each:
   - top-1/top-3 artwork + edition
   - p50/p95 latency on Pi
   - memory footprint
4. Pick the best **accuracy-per-watt** point, not just highest absolute accuracy.

That gives a deployable answer quickly while preserving a path to future model upgrades.
