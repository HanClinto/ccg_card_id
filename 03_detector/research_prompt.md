# Research Prompt: Card Corner Detection — Architecture & Training Strategy

## Context

We are building a system to identify Magic: The Gathering cards from real-world images
(phone camera, pack-opening video frames). The full pipeline is:

```
Raw image → [Corner detection / dewarp] → [Card identification (ArcFace embedding)]
```

This document is specifically about the **corner detection** step. We need a model that,
given a raw image containing a card, outputs the (x, y) coordinates of the four card
corners in normalized [0, 1] space (TL → TR → BR → BL order).

---

## The Problem

Cards appear in a wide variety of conditions:
- Pack-opening video frames: cards held up to camera, various distances, angles, lighting
- Phone camera handheld scans: cleaner backgrounds, closer shots
- Flatbed scanner images: very clean, consistent lighting

The model must handle:
- Cards at varying scales (card may occupy 5%–95% of frame)
- Rotation up to ±45° from upright
- Partial occlusion (fingers, other cards)
- Varying backgrounds (cluttered desks, hands, packaging)
- Motion blur in video frames

### Sample Images

Training data (packopening video frames):
```
datasets/packopening/frames/_4rKseBy1rw_unlimited-booster-from-the/frame_0.jpg
datasets/packopening/frames/_4rKseBy1rw_unlimited-booster-from-the/frame_10005.jpg
```

Test data (clint_cards_with_backgrounds — handheld camera, varied real-world backgrounds):
```
datasets/clint_cards_with_backgrounds/data/04_data/good/03220cab-fc78-4323-bd34-b8dbebe35597_CityOfBrass_20221129_144100.mp4-0000.jpg
  corners: TL=(0.197, 0.187)  TR=(0.730, 0.173)  BR=(0.758, 0.592)  BL=(0.178, 0.600)

datasets/clint_cards_with_backgrounds/data/04_data/good/045abeeb-f5e5-4f3f-9836-5b1553e03f11_GrimBackwoods_20221129_144202.mp4-0405.jpg
  corners: TL=(0.078, 0.171)  ...

datasets/clint_cards_with_backgrounds/data/04_data/good/2be39749-ad6f-4160-99eb-c677eee7f1b2_ThrabenInspector_20221130_090707.mp4-1034.jpg
  corners: TL=(0.221, 0.299)  ...
```

All image paths are relative to the data root (`/Volumes/carbonite/claw/data/ccg_card_id/`).

---

## Training Data

**Primary source: packopening SQLite DB**
- ~420k frames from 768+ pack-opening YouTube videos
- Each frame has SIFT-verified corner coordinates (homography against Scryfall reference)
- Quality filter: pHash distance between dewarped frame and Scryfall reference
  - ph≤10: ~215k frames — near-perfect to good matches (45% of total)
  - ph≤20: ~420k frames — includes some partial occlusions and suspect matches
- Negatives: 10k frames from videos where SIFT found no match (card may or may not be present)

**Test set (held-out domain): clint_cards_with_backgrounds**
- 1,329 frames: 1,267 positives (39 cards, varied backgrounds) + 62 hard negatives
- Never seen during training — used to measure domain generalization

---

## Architectures Tried

### Architecture A: TinyCornerCNN (soft-argmax heatmap head)

A purpose-built ~48K parameter depthwise-separable CNN targeting edge deployment.

**Architecture:**
```
Input: (B, 3, 448, 448) ImageNet-normalized

Encoder (stride-16, produces 28×28 feature maps):
  Conv2d(3→16, stride=2)          448→224
  DW-Sep(16→32, stride=2)         224→112
  DW-Sep(32→64, stride=2)         112→56
  DW-Sep(64→64, stride=1)
  DW-Sep(64→128, stride=2)        56→28
  DW-Sep(128→128, stride=1)
  → (B, 128, 28, 28)

Corner head:
  Conv1x1(128→32) → BN → ReLU6 → Dropout2d(0.2) → Conv1x1(32→4)
  → (B, 4, 28, 28) raw heatmap logits (one channel per corner: TL/TR/BR/BL)
  → soft-argmax → (B, 8) corner coords in [0,1]

Presence head:
  AdaptiveAvgPool2d(1) → Flatten → Linear(128→1)
  → (B,) presence logit
```

**Soft-argmax** converts heatmaps to coordinates:
```python
weights = softmax(heatmap.flatten())           # (784,)
coords  = sum(weights * pixel_center_grid)     # weighted average position
```
This is fully differentiable and has theoretical advantages (handles occlusion via diffuse
peaks, spatial precision limited to ~16px grid).

**Loss function:**
```
L = BCE(presence) + λ_c * SmoothL1(corners, positives_only) + λ_h * BCE(heatmaps, gaussian_targets)
```
- `λ_c` = 5.0 (corner regression weight)
- `λ_h` = configurable (heatmap auxiliary supervision weight)
- Gaussian heatmap targets: σ=2.0 heatmap pixels (~32px at input scale)

**Status: conclusively stuck** — see run history below.

### Architecture B: MobileViTCornerDetector (spatial regression head)

MobileViT-XXS backbone (951K params, 3.6 MB fp32) + spatial-aware direct regression head.
Same backbone as the card-ID embedding model, so weights can be seeded from a pretrained
ArcFace checkpoint. No heatmaps — direct coordinate regression from pooled spatial features.

**Architecture:**
```
Input: (B, 3, 448, 448) ImageNet-normalized

Backbone: MobileViT-XXS (timm)
  forward_features(x) → (B, 320, 14, 14)  [448/32 = 14]

Spatial pool:
  AdaptiveAvgPool2d(2) → (B, 320, 2, 2)
  Flatten → (B, 1280)

Regression head:
  LayerNorm(1280) → Linear(1280→256) → GELU → Dropout(0.3) → Linear(256→9)
  → corners[:8] = (B, 8) corner coords, out[8] = (B,) presence logit
```

The 2×2 spatial pool preserves coarse quadrant structure (top-left, top-right, bottom-left,
bottom-right), which is critical for corner regression — without it, the model loses all
spatial information before predicting corner positions.

Backbone can be seeded from:
- ImageNet pretrained weights (timm default, `--seed imagenet`)
- ArcFace card-ID checkpoint (`--seed-checkpoint path/to/checkpoint.pt`, loads `backbone.*` keys)

**Run naming convention:**
`{backbone}_{head}_{loss_cfg}_{input}_{data_filter}_{seed}_{lr_cfg}`
e.g. `mvit_spatialreg_lc5_img448_ph10_seedin_blr10`
- `seedin` = ImageNet seed, `seedcid` = card-ID ArcFace seed
- `blr10` = backbone LR at 10% of head LR (differential LR)

---

## What We Tried and Why

### Metric: Corner Point Error (CPE)
Mean L2 distance in normalized [0,1] units between predicted and true corners, on positive
examples only. **Baseline: ~0.35** (equivalent to predicting all corners at image center,
given that pack-opening cards typically fill most of the frame). Lower is better.

### Run History

**Architecture A — TinyCornerCNN (soft-argmax):**

| run_name | bs | lr | epochs | e1 val_cpe | final val_cpe | verdict |
|---|---|---|---|---|---|---|
| `tcnn_softargmax_lc5lh10_img448_ph20` | 128 | 4e-3 | 8 | 0.611 | 0.634 | stuck at baseline |
| `tcnn_softargmax_lc5lh10_img448_ph20` | 128 | 1e-3 | 5 | 0.596 | 0.642 | stuck at baseline |
| `tcnn_softargmax_lc5lh0_img448_ph20`  | 32  | 1e-3 | 3 | 0.626 | 0.638 | stuck at baseline |
| `tcnn_softargmax_lc5lh0_img448_ph10`  | 32  | 1e-3 | **40** | 0.597 | **0.649** | conclusively stuck |
| `tcnn_softargmax_lc5lh1_img448_ph10`  | 32  | 1e-3 | **9** | 0.640 | 0.666 | stuck at baseline |

**Architecture A key observations:**
- `val_cpe` ≈ 0.63–0.65 throughout all runs — essentially the "predict image center" baseline
- `train_loss` drops nicely (e.g. 0.27→0.083 over 40 epochs for lh0/ph10) — the model
  IS learning something on the training set, but it doesn't generalize to corner localization
- `val_pres_acc` reaches 0.994–0.997 early — presence detection is working fine
- `test_cpe` (clint domain) stays at ~0.35–0.37 across all runs (center prediction)
- Increasing `λ_h` (heatmap weight) or decreasing it or removing it entirely: no difference

**Architecture B — MobileViTCornerDetector (spatial regression):**

| run_name | seed | backbone_lr | epochs | e1 val_cpe | final val_cpe | test_cpe | verdict |
|---|---|---|---|---|---|---|---|
| `mvit_spatialreg_lc5_img448_ph10_seedcid` | ArcFace | 1e-3 (full) | 4 | **0.571** | 0.662 | 0.534 | breaks center; catastrophic forgetting |
| `mvit_spatialreg_lc5_img448_ph10_seedin_blr10` | ImageNet | 1e-4 (10%) | 1 | — | — | — | killed (memory); **needs restart** |

**Architecture B key observations:**
- `seedcid` run broke below the center-prediction baseline on **epoch 1** (val_cpe 0.571 vs.
  0.635 for TinyCornerCNN) — the pretrained backbone provides immediately useful features
- By epoch 4 without differential LR: `test_pres_acc` collapses to **0.131** (catastrophic
  forgetting — backbone overwrites pretrained features at lr=1e-3)
- `test_cpe` rose from 0.384 (e1) to 0.534 (e4) as the backbone degraded
- Fix implemented: `--backbone-lr-scale 0.1` (`blr10`) — backbone at 1e-4, head at 1e-3.
  The `seedin_blr10` run (ImageNet seed + differential LR) was killed at epoch 1 due to
  memory pressure before producing useful results. **This is the current frontier.**

### Why Each Change Was Made

**lh=10 (original):** Default from model design. Heatmap BCE loss on 28×28 = 784 pixels
with only ~25 pixels near each corner (Gaussian σ=2). Suspected to overwhelm the
coordinate regression gradient (~31× more pixels contributing to BCE than are "near" the peak).

**lh=0:** Removed heatmap supervision entirely to test whether lh=10 was the problem.
Result: still stuck. Conclusion: soft-argmax without *any* heatmap supervision is too
indirect — gradient is spread across all 784 pixels equally, giving no strong signal to
form a peak at the right location.

**ph=10 (tighter quality filter):** Reduced from ph≤20 to ph≤10 to ensure training corners
are always fully visible (no partial occlusions). Halved dataset to ~215k frames. No effect
on val_cpe convergence behavior.

**bs=32 (smaller batch):** More gradient updates per epoch (12.8k vs 3.2k at bs=128),
same wall-clock throughput on Apple MPS (~4 b/s, ~284 img/s regardless of batch size).
No effect on the fundamental stuckness.

**lh=1:** Tried a small heatmap auxiliary signal — enough to give spatial structure to the
heatmaps without overwhelming coordinate regression. Result: still stuck at val_cpe 0.64–0.67
over 9 epochs. Conclusively, soft-argmax + TinyCornerCNN cannot break the center-prediction
local minimum regardless of the heatmap loss weight.

**MobileViT spatial regression (`seedcid`):** Replaced the CNN backbone with MobileViT-XXS
seeded from the card-ID ArcFace checkpoint. Replaced soft-argmax with direct spatial
regression (pool to 2×2 → flatten → FC → 9 outputs). **This immediately broke below center
baseline** (e1 val_cpe 0.571). However, training at full lr=1e-3 caused catastrophic forgetting
of the pretrained backbone features by epoch 4 (test_pres_acc → 0.131).

**MobileViT spatial regression (`seedin_blr10`):** ImageNet seed + differential LR (backbone
at 10% of head LR via `--backbone-lr-scale 0.1`). This is the architecturally correct fix for
catastrophic forgetting. Run was killed at epoch 1 due to memory pressure — no results yet.

---

## Root Cause Diagnosis

### TinyCornerCNN (Architecture A): confirmed soft-argmax gradient collapse

The soft-argmax architecture has a **sparse supervision / diffuse gradient problem**:

Without heatmap auxiliary loss, the only gradient path to the heatmap spatial values is:
```
SmoothL1(soft_argmax_output, true_corners)
  → ∂cx/∂h_k = w_k * (grid_x_k - cx)   [for each of 784 pixels]
```
Each pixel's gradient is small (~1/784 of total signal). The model has no incentive to
make any one pixel "peak" — the loss is equally satisfied by a diffuse uniform distribution
outputting (0.5, 0.5). The model converges to this local minimum and stays there.

With lh=10, the heatmap BCE loss provides explicit peak supervision, but at 784× the
magnitude of a single-pixel contribution, it drowns out the coordinate regression term.
Neither lh=0, lh=1, nor lh=10 escape the local minimum over 40 epochs. The TinyCornerCNN
soft-argmax architecture is **conclusively unable** to learn corner localization from this
training data.

### MobileViTCornerDetector (Architecture B): promising but needs fine-tuning stabilization

Replacing the backbone with pretrained MobileViT-XXS + direct spatial regression head
**immediately breaks the center-prediction local minimum** (e1 val_cpe 0.571). The pretrained
features already encode spatial structure sufficient to distinguish image quadrants.

The open problem is **catastrophic forgetting**: training at full backbone lr=1e-3 degrades
the pretrained features by epoch 4. Differential LR (`--backbone-lr-scale 0.1`) is the
implemented fix, but the corrected run was killed before producing results.

**The key unanswered question is no longer "can this learn at all" — it's "what is the right
architecture and training recipe to stabilize MobileViT fine-tuning for keypoint regression,
and is direct spatial regression the right output head or should we add a heatmap branch?"**

---

## Constraints

- **Edge deployment target:** The model should be small enough to run in a mobile app
  or browser. The current TinyCornerCNN at 48K params / 0.2 MB fp32 is ideal. A model
  up to ~1–2 MB fp32 (~250K–500K params) would be acceptable. MobileViT-XXS (951K, 3.6 MB)
  is the upper bound we've considered.
- **Input resolution:** 448×448 (chosen to match the card identification model)
- **Inference speed:** Should run in <100ms on a mobile CPU. Sub-50ms preferred.
- **Framework:** PyTorch. Apple MPS available for training (M-series Mac).
- **No external API dependencies** at inference time.

---

## Questions for Research

We are looking for recommendations on:

1. **What is the right output head for MobileViT spatial regression?** Soft-argmax on
   TinyCornerCNN is conclusively stuck. Direct spatial regression on MobileViT backbone
   breaks the center-prediction local minimum immediately (e1 val_cpe 0.571 vs. 0.635).
   The question now is: given a pretrained MobileViT-XXS backbone, is direct regression
   from pooled spatial features (current approach: pool to 2×2 → flatten → FC → 8 coords)
   the best head design? Or would a heatmap head on top of the MobileViT features work
   better (e.g. upsample the 14×14 feature map → 4-channel heatmap → soft-argmax)?
   Trade-offs: heatmap head gives occlusion confidence estimates but adds complexity;
   direct regression is simpler and already works on epoch 1. What does the literature
   recommend for keypoint regression heads on top of transformer backbones?

2. **Training strategies for heatmap-based localization with sparse peaks:** What is
   the state of the art for supervising heatmap localization models? Is there a better
   loss formulation than BCE on Gaussian targets? (e.g. focal loss on heatmaps, MSE on
   heatmaps, Wasserstein distance, offset supervision, etc.)

3. **Alternative lightweight architectures** that have been shown to work well for
   keypoint/corner detection at this model size and input resolution. Consider:
   - MobileNetV3 + keypoint head
   - EfficientNet-Lite + keypoint head
   - SuperPoint (self-supervised homographic adaptation — but is it too large?)
   - Any purpose-built tiny keypoint detectors
   - Classical + learned hybrid approaches

4. **Data augmentation strategies** specific to corner detection that we may be missing.
   We currently do: 90°/180°/270° rotation, ColorJitter (brightness/contrast/saturation/hue).
   Excluded: horizontal flip (MTG cards have asymmetric text/art — mirrored cards don't exist),
   fine rotation ±30° (slow, and real-world frames already contain the full range of card angles),
   GaussianBlur (real video frames already contain motion blur natively),
   RandomGrayscale (minimal value given camera captures are always color).

5. **Is there a canonical "card corner detection" solution** in the wild (mobile scanning
   apps, document scanners) that we should be adapting rather than training from scratch?
   DocScan, CamScanner-style approaches, etc.

6. **Sanity check:** Given the training setup described, does the stuck-at-center behavior
   make sense? Are there any obvious bugs or design flaws we may have missed?

---

## Deliverable Request

Please research the above questions and return:

1. **A diagnosis** of why our current approach is stuck (confirming, refining, or
   correcting our root cause hypothesis).

2. **A ranked list of 2–3 alternative approaches** with:
   - Architecture description
   - Why it would work better for this specific problem
   - Rough parameter count and inference speed estimate
   - Any known PyTorch implementations or papers to reference

3. **A concrete implementation plan** for the most promising approach:
   - What changes to make to `model.py` and `train.py`
   - What loss function to use
   - Recommended hyperparameters
   - Expected training behavior and convergence timeline

4. **Any quick experiments** (minimal code changes) that could be run on the existing
   architecture to further diagnose the problem before committing to a full rewrite.

---

## Codebase Reference

Key files:
```
03_detector/detectors/tiny_corner_cnn/
  model.py       TinyCornerCNN architecture, soft-argmax, Gaussian heatmap helpers
  dataset.py     Data loading (packopening DB + clint CSV), augmentation pipeline
  train.py       Training loop, loss function, history CSV logging
  precache_dataset.py   Pre-resizes training frames to 448×448 on local SSD

03_detector/README.md   Architecture overview and experiment log
```

Training history CSV (all runs, one row per epoch):
```
results/corner_detector/training_history.csv
```
Columns: `run_name, epoch, timestamp, arch, lambda_corners, lambda_heatmap,
batch_size, lr_start, train_source, max_phash_dist, train_loss, val_loss,
val_cpe, val_pres_acc, test_cpe, test_pres_acc, lr_end, checkpoint_saved`
