# Card Detection for MTG Card Scanning

Before a Magic: The Gathering card can be identified, it needs to be found. This project implements
stage one of the two-stage pipeline: given a raw image from a phone camera, flatbed scanner, or
video frame, locate the card and return its four corner points so the card can be dewarped into a
clean, rectified crop for the identification stage.

---

## The Pipeline

The full system works in two stages:

**Stage 1 — Detect and align** (this project). The detector takes a raw image and returns the four
corner coordinates of the most prominent card. A homography warp is then applied to produce a
flat, rectangular card image at a standard aspect ratio. This step compensates for perspective,
rotation, and moderate amounts of shear introduced by off-angle captures.

**Stage 2 — Identify** (see `04_build/` and `06_eval/`). The dewarped card image is passed to
whatever embedding model is used (whether pHash or ArcFace), which embeds it into an n-dimensional
vector for nearest-neighbor lookup against a gallery of ~108k Scryfall reference embeddings. The 
closest match is the card identity.

The two stages are deliberately separate. Detection is a geometric problem (where is the card?),
while identification is a semantic one (which card is it?). Keeping them separate makes each
component easier to evaluate, replace, and improve independently.

---

## Why Detection Is Hard

For clean studio shots — white background, good lighting, a single bordered card placed flat — the
problem is almost trivial. A simple Canny edge detector finds the rectangular outline and you are
done. However, this requires card scanning to be done against a plain (usually white) background
without anything touching the edges of the cards. This is often inconvenient, and it would be nice
to be able to scan against noiser backgrounds, or with cards held in a hand.

**Borderless and full-bleed art.** Modern Magic sets (especially Universes Beyond prints, showcase
frames, and many commander products) extend the artwork all the way to the card edge. There is no
high-contrast border to anchor edge detection. The card boundary blends into whatever is behind
it, and classical edge detectors have nothing to grip.

**Foil cards.** Specular reflections on foil create bright patches that can be as intense as any
real edge. These false edges fragment and confuse contour-based approaches. The reflection pattern
shifts with viewing angle, so it cannot be characterized with a fixed filter.

**Sleeves.** Deck-protection sleeves add a layer of plastic over the card. The plastic itself has
a slightly rounded surface that picks up glare, and the rounded corners change the card's apparent
shape. Double-sleeved cards add even more rounding. The sleeve may also introduce a thin gap
between itself and the card, creating a double-edge artifact.

**Varied and cluttered backgrounds.** Cards placed on wooden tables, playmats, carpet, or other
cards all provide edges and textures that compete with the card outline. A contour-based detector
that works on a white desk may completely fail on a dark playmat covered in other cards.

**Partial occlusion.** Cards partially outside the frame, or overlapping other cards, produce
incomplete outlines. A detector that requires a complete quad will miss these.

**Extreme perspective angles.** A card viewed nearly edge-on produces a very thin quadrilateral.
At extreme angles the foreshortened dimension can be smaller than many contour filters' minimum
area threshold, causing the card to be missed entirely.

**Multiple cards in frame.** Pack opening videos, draft picks, and collection sorting all place
multiple cards in frame simultaneously. The detector needs to find the "primary" card (usually the
most prominent or most central) reliably. It would be possible to build a detector that can find
multiple cards at the same time, but for our purposes, we will be focusing on the use-case of
finding only the single-most prominent card in the frame at a time.

The bottom line is that classical Canny/polygon detection is useful as a fast baseline on
controlled captures, but breaks down badly in the conditions that actually matter. The neural
approach is designed to handle all of the above.

---

## Coordinate Convention

All detectors return corners in a consistent order and coordinate system.

**Corner ordering** (clockwise from top-left):

```
corner 0 — Top-Left  (TL)
corner 1 — Top-Right (TR)
corner 2 — Bottom-Right (BR)
corner 3 — Bottom-Left  (BL)
```

**Coordinate system**: normalized `(x, y)` in the range `[0, 1]`, where `(0, 0)` is the top-left
pixel of the image and `(1, 1)` is the bottom-right. Normalized coordinates make results
resolution-independent: the same corner positions describe the same geometric location regardless
of whether the input was 720p, 1080p, or 4K.

To convert to pixel coordinates, multiply `x` by image width and `y` by image height:

```python
corners_px = result.corners_pixel(image_width=img.shape[1], image_height=img.shape[0])
```

The TL/BR pair shares the smallest and largest coordinate sums; TR/BL are separated by their
coordinate difference. This is the standard two-pass sort used throughout the codebase.

---

## Detector Implementations

Three detectors are provided, each with different trade-offs between speed, accuracy, and
prerequisites.

### CannyPolyDetector

A classical computer vision pipeline. The algorithm is:

1. Convert the BGR image to grayscale.
2. Optionally apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local
   contrast — useful when the card is in shadow.
3. Gaussian blur to suppress high-frequency noise.
4. Canny edge detection with configurable low and high thresholds.
5. Slight dilation of edges to close small gaps in the outline.
6. Find all contours, sort by area descending.
7. For each contour, compute the convex hull, then approximate with `approxPolyDP` using an
   epsilon of 2% of the perimeter. If the result has exactly 4 points and its area falls in the
   expected range (3%–95% of the image), accept it as the card quad.
8. Sort the winning quad's corners into TL/TR/BR/BL order and normalize.

Confidence is the area of the winning quad relative to the image area, capped at 1.0. A larger
card fill typically means the detector is more reliable.

**Strengths**: Fast (milliseconds on CPU), requires no training data, no GPU, no extra
dependencies. Works well on high-contrast bordered cards against plain backgrounds.

**Weaknesses**: Fails on borderless/full-bleed artwork, low-contrast backgrounds, foil glare, and
sleeves. These are exactly the cases that matter most in real-world use.

### SIFTHomographyDetector

Uses SIFT feature matching between the query image and a gallery of known reference images to
compute a homography, then maps the card's four known corners through it to get the query-image
corner locations.

The approach is extremely precise when it works: rather than trying to find the card outline from
scratch, it directly measures how the coordinate system of the known reference image has been
transformed in the query image. A clean dewarped result follows directly.

The major limitation is that it requires knowing the card identity in advance, which closes the
loop in the wrong direction for a general detector. You need to know what you are looking for
before you can look for it. In practice this detector is used as a ground-truth labeling tool (see
`02_data_sets/clint_cards_with_backgrounds/code/03_extract_corners.py` and the packopening
pipeline) rather than as a deployment detector. When `gallery=None` is passed, the detector
immediately returns `card_present=False`.

Thus, the SIFT homography detector is primarily used for building a ground-truth dataset useful
for training and evaluating the neural detectors, rather than a production detector itself.

Requires `opencv-contrib-python` or `opencv-contrib-python-headless` for SIFT.

### NeuralCornerDetector (TinyCornerCNN)

A learned detector that takes any image and predicts corners directly, with no gallery required.

**Architecture**: TinyCornerCNN — a purpose-built ~48K-parameter depthwise-separable CNN. It does
**not** use the MobileViT backbone. The encoder consists of four stride-2 depthwise-separable
blocks, reducing a 448×448 input to 28×28 feature maps (stride-16 total, 128 channels). This
spatial map is kept intact and fed to two heads:

```
Input: (B, 3, 448, 448)

Encoder (stride-16, 4× stride-2):
  Conv2d(3→16, s=2) → DW-Sep(16→32, s=2) → DW-Sep(32→64, s=2)
  → DW-Sep(64→64, s=1) → DW-Sep(64→128, s=2) → DW-Sep(128→128, s=1)
  → (B, 128, 28, 28)

Corner head:
  Conv2d(128→32, 1×1) → BN → ReLU6 → Dropout2d → Conv2d(32→4, 1×1)
  → (B, 4, 28, 28) heatmaps (one per corner: TL, TR, BR, BL)
  → soft-argmax → (B, 8) corner coordinates

Presence head:
  AdaptiveAvgPool2d(1) → Flatten → Linear(128→1) → (B,) logit
```

**Heatmap regression with soft-argmax**: Rather than compressing the spatial feature map to a
vector and regressing coordinates directly (which loses spatial precision), the model outputs a
4-channel spatial heatmap. One channel per corner, at 28×28 resolution. The predicted corner
coordinate is the softmax-weighted average of the spatial grid positions — a fully differentiable
operation. This preserves spatial precision all the way to the output, and handles occlusion
naturally: an occluded corner produces a low-confidence, diffuse heatmap peak rather than a
confident wrong prediction.

**Per-corner confidence**: Available from the peak value of each heatmap channel after sigmoid:
`sigmoid(heatmaps).amax(dim=(-2,-1))` → 4-element vector of confidence scores per corner.

**Loss**: Three terms, applied per-batch:

- `BCEWithLogitsLoss` on the presence logit (all frames including negatives)
- `SmoothL1Loss` on corner coordinates from soft-argmax (positives only), weight `λ_c = 5.0`
- `BCEWithLogitsLoss` on raw heatmap logits vs Gaussian blob targets at GT corner positions
  (positives only), weight `λ_h = 10.0`, sigma = 2.0 heatmap pixels (~32px at input scale)

The Gaussian heatmap loss provides dense spatial supervision that guides the model toward localized
peaks, complementing the coordinate regression loss.

**Strengths**: Handles all the difficult cases (borderless art, foil, sleeves, clutter, partial
occlusion) once trained. No gallery required. Single forward pass. Runs well on MPS/GPU. The
heatmap architecture gives ~16px spatial precision at 448×448 input — substantially better than
global regression approaches at the same parameter count.

**Weaknesses**: Requires training data and a GPU for practical training. Requires a pre-built
cache of 448×448 images for practical training throughput (see `precache_dataset.py`).

---

## Training Data

### Packopening DB (primary training source)

Training data comes exclusively from the packopening pipeline:

```
datasets/packopening/packopening.db
```

The database stores SIFT-verified corner labels for frames extracted from hundreds of pack-opening
videos on YouTube. Every row in the `frames` table is a frame where SIFT homography successfully
matched a known card reference, producing reliable normalized corner coordinates. The dataset
contains ~409k such labeled frames across ~768 videos, covering a wide range of card sets,
lighting conditions, camera angles, and card conditions.

Labels are filtered by **pHash distance** (Hamming distance between the dewarped frame's pHash and
the Scryfall reference image pHash). Low pHash distance confirms that the SIFT homography matched
the right card and the corners are trustworthy. Training uses frames with `phash_dist ≤ 20`, which
retains ~89% of frames (~340k) while excluding borderline matches where labels may be unreliable.

Negative examples (frames where no card was visible) are sampled from videos in
`frames_extracted` status — videos that had frames extracted but no SIFT matches found. These
provide realistic "no card present" examples drawn from the same video domain as the positives.

The packopening domain closely mirrors the `clint_backgrounds` evaluation domain: both are
handheld captures from video, with similar card orientations, lighting variation, and background
clutter. This domain alignment is the main reason packopening was chosen as the training source
rather than more controlled datasets.

### clint_cards_with_backgrounds (test set only — never used in training)

```
datasets/clint_cards_with_backgrounds/data/04_data/corners.csv
```

This 1,271-frame dataset is held out exclusively as a **test set** for domain-generalization
evaluation. It is never included in training or validation splits. Using it for training would
leak test information and make evaluation results meaningless. See the Evaluation section for how
it is used.

### Augmentation

The `CornerDataset` class applies the following augmentations during training:

- **90°/180°/270° rotation** (75% chance each training pass) — covers all card orientations
- **Fine rotation ±30°** (50% chance) — simulates angled captures
- **Color jitter** (brightness/contrast/saturation/hue) — lighting variation
- **Random grayscale** (5% chance) — partial grayscale robustness
- **Gaussian blur** (20% chance) — slight defocus robustness

Cards are not horizontally flipped. Flipping would produce mirror-image text and artwork that
does not exist in the real world and would train the model on physically impossible examples.

All spatial augmentations update the corner coordinates alongside the image. After each spatial
augmentation, corners are re-sorted into canonical TL→TR→BR→BL order via `sort_corners_canonical`,
so the heatmap channel assignments (channel 0 = TL, 1 = TR, 2 = BR, 3 = BL) remain consistent
regardless of what rotation was applied.

---

## Evaluation Metrics

Five metrics are tracked in the benchmark:

**Corner Point Error (CPE)** is the mean Euclidean distance between predicted and ground-truth
corners, in normalized units. A CPE of 0.02 means the average corner is off by 2% of the image
dimension — about 14 pixels in a 720p image.

**PCK@k (Percentage of Correct Keypoints)** measures how often each predicted corner falls within
a threshold distance of the ground-truth corner. The threshold is expressed as a fraction of the
image diagonal. PCK@5% means the corner must be within 5% of the diagonal, or about 40 pixels in
a 720p frame. Reported as a fraction in [0, 1].

**Quad IoU** is the intersection-over-union of the predicted quadrilateral against the ground-truth
quad, computed in pixel space. A perfect prediction scores 1.0; a completely wrong prediction
scores 0.0. `metrics.py` provides both a fast bounding-box approximation and an exact polygon IoU
via shapely (used when shapely is installed).

**Detection Rate** (recall on positives) is the fraction of frames containing a card where the
detector correctly predicts `card_present=True`. A detector that always says no card would score
0.0 here.

**False Positive Rate** (FPR on negatives) is the fraction of hard-negative frames where the
detector incorrectly predicts `card_present=True`. Lower is better. A detector that always says a
card is present would score 1.0 here.

---

## Evaluation Datasets

Two datasets are used for evaluation. They represent complementary points on the difficulty spectrum.

### Sol Ring (best-case / Canny baseline)

307 frames extracted from pack-opening or scanning videos, all showing Sol Ring editions against
a plain white or near-white background. Cards are fully visible with no occlusion, held flat, and
well-lit. These conditions are as close to ideal as real-world captures get.

This is the **best-case scenario for the Canny detector**: high-contrast card borders against a
clean background make edge detection trivial. Canny performs near its ceiling here. Any neural
model must match or exceed Canny on this dataset to be considered production-worthy — failing on
easy cases is unacceptable even if hard cases improve.

### Clint Backgrounds (worst-case / neural model target)

1,271 frames from phone video, covering 39 different cards held in hands against a variety of
highly-colored, cluttered backgrounds. The dataset includes:

- **Partial finger occlusion** — one or more corners are partially covered by the holder's fingers
- **Borderless and full-art cards** — the artwork extends to the card edge, leaving no border for
  edge detectors to grip
- **Highly varied backgrounds** — colored playmats, wooden tables, other cards, dark surfaces
- **Perspective and rotation** — cards tilted or at moderate angles

This is the **worst-case scenario for the Canny detector** and the **primary target for the neural
model**. Canny's performance degrades substantially here. This is the domain the neural model is
designed to handle, and where improvements matter most in practice.

The domain of clint_backgrounds closely matches the packopening training domain: both involve
handheld cards from video with similar capture conditions. The key difference is that clint is held
out completely from training — it is used only for evaluation, giving an honest estimate of
generalization to real-world conditions the model has never seen during training.

## Running the Benchmark

```bash
# Activate the venv first
source .venv312/bin/activate

# Evaluate CannyPolyDetector on the val split (default)
python 03_detector/eval/benchmark.py

# Evaluate both Canny and TinyCornerCNN
python 03_detector/eval/benchmark.py \
    --detectors canny,tinycornercnn \
    --neural-checkpoint /path/to/corner_detector_tiny/last.pt

# Evaluate on all data (not just val split)
python 03_detector/eval/benchmark.py --split all

# Quick smoke-test: only 20 images per split
python 03_detector/eval/benchmark.py --limit 20

# Override data directory
python 03_detector/eval/benchmark.py \
    --data-dir /Volumes/carbonite/claw/data/ccg_card_id

# Two-stage evaluation (stage 1 rough detection → crop → stage 2 refinement)
python 03_detector/eval/two_stage_test.py \
    --checkpoint /path/to/last.pt --split all
```

Expected output format:

```
Detector             | CPE↓   | PCK@5%↑ | IoU↑   | Det.Rate↑ | FPR↓
CannyPolyDetector    | 0.042  | 71.3%   | 0.81   | 88.4%     | 12.5%
NeuralCornerDetector | 0.018  | 93.7%   | 0.94   | 97.2%     |  3.1%
```

---

## Experiments

### v0.1 — Baseline TinyCornerCNN (run1/run2, epochs 1–73)

The first training attempt established a working pipeline and surfaced several important failure
modes.

**Setup:**

| Parameter | Value |
|---|---|
| Architecture | TinyCornerCNN (44k parameters) |
| Training data | packopening DB, phash ≤ 20, ~394k positives |
| Negatives | None (presence classifier disabled — see below) |
| Augmentation | Horizontal flip + ±30° fine rotation + color jitter |
| Loss | SmoothL1 on fixed TL/TR/BR/BL corner order |
| Learning rate | 3e-3 cosine decay to 3e-5 |
| Batch size | 64 |
| Epochs | 73 (early stop — architecture replaced) |

**In-domain training curve:**

| Epoch | Train Loss | Val Loss | Val CPE | Test CPE (clint) |
|---|---|---|---|---|
| 33 | 0.0200 | 0.0144 | 0.0750 | 0.230 |
| 40 | 0.0124 | 0.0090 | 0.0552 | 0.237 |
| 50 | 0.0110 | 0.0079 | 0.0514 | 0.279 |
| 60 | 0.0105 | 0.0076 | 0.0499 | 0.238 |
| 73 | 0.0101 | 0.0070 | 0.0477 | 0.254 |

In-domain val CPE improved steadily. The clint test CPE (domain-generalization) plateaued around
0.23–0.28 with no consistent improvement trend despite the in-domain gains.

**Benchmark results vs Canny (all frames, `benchmark.py --split all`):**

*Sol Ring — 307 frames, 21 card editions, aligned video captures:*

| Detector | CPE↓ | PCK@5%↑ | IoU↑ | ID-able↑ | Det.Rate↑ |
|---|---|---|---|---|---|
| CannyPolyDetector | **0.022** | **98.9%** | **0.885** | **48.2%** | 100% |
| TinyCornerCNN e32 | 0.187 | 8.1% | 0.352 | 0.0% | 100% |
| TinyCornerCNN e40 | 0.194 | 4.9% | 0.311 | 0.0% | 100% |

*Clint backgrounds — 1,267 frames, 39 cards, varied handheld captures with backgrounds:*

| Detector | CPE↓ | PCK@5%↑ | IoU↑ | ID-able↑ | Det.Rate↑ |
|---|---|---|---|---|---|
| CannyPolyDetector | **0.160** | **50.2%** | **0.470** | **6.9%** | 86.6% |
| TinyCornerCNN e32 | 0.229 | 4.4% | 0.316 | 0.0% | 100% |
| TinyCornerCNN e40 | 0.237 | 3.6% | 0.216 | 0.0% | 100% |

The neural detector was not competitive with Canny at 73 epochs on either dataset. The domain gap
between packopening training data and real-world captures was roughly 5× (val CPE 0.047 vs test
CPE 0.23–0.25). Corner accuracy was low enough that no dewarped crops passed pHash verification
(ID-able = 0%).

**Issues discovered and resolved:**

*Presence classifier collapse.* When trained with negative examples (frames with no visible card),
the model learned to minimize loss by predicting "no card" for everything. Root cause: the
packopening dataset is ~34:1 positive-to-negative, so a near-constant "absent" prediction achieves
very low BCE loss. Fix: disable negative sampling (`--neg-sample-n 0`) and set
`ignore_presence=True` in inference. The identification stage (pHash / ArcFace) acts as the real
presence filter — if the dewarped crop does not match any gallery card, the frame is discarded
there.

*External drive I/O bottleneck.* Loading 394k JPEG frames from a spinning external drive gave
~4–7 batch/s and ~26 min/epoch. Pre-resizing all frames to 224×224 on a local SSD (`precache_dataset.py`)
raised throughput to ~18 batch/s and ~5–6 min/epoch.

*Ordered loss is an unnecessary constraint.* The SmoothL1 loss compared predicted corners against
ground truth in a fixed TL/TR/BR/BL order. This means a card rotated 90° has a completely
different "correct" answer, adding gradient noise and making 90°/270° augmentation awkward. See
v0.2 for the fix.

---

### v0.2 — Dihedral-invariant loss + geometric corner sorting

**Motivation:**

The v0.1 loss required the model to predict corners in a specific winding direction *and* a
specific starting corner. This is an unnecessary constraint. The model's job is to locate four
points; the labeling of which point is "top-left" is a post-processing detail that can be resolved
geometrically at inference time.

Forcing the model to learn both the corner locations *and* their canonical labeling under a single
SmoothL1 loss creates two problems:

1. **Gradient noise from equivalent configurations.** A card seen at 0° and the same card seen at
   90° are geometrically equivalent, but the v0.1 loss assigns them different target vectors. The
   model has to learn to output different orderings for the same physical arrangement, which wastes
   capacity and slows convergence.

2. **Augmentation brittleness.** A horizontal flip reverses the winding order of the corners
   (clockwise becomes counter-clockwise). Compensating in augmentation code requires every
   label-generating step to maintain a canonical clockwise order — a silent violation anywhere
   silently corrupts labels.

**Changes in v0.2:**

*Loss — full dihedral invariance.* The corner loss evaluates all 8 orderings in D4 (4 cyclic
rotations of the ground-truth quad, plus 4 cyclic rotations of the reversed quad) and uses the
minimum. The model only needs to locate four points correctly.

*Augmentation — 90°/180°/270° rotations.* Applied a random multiple of 90° followed by a fine
random rotation of ±30°. Combined with horizontal flip, the model sees cards in every orientation.

*Inference — geometric corner sorting.* Predicted corners are sorted by centroid-relative angle
and rotated so the smallest `x + y` point is index 0, giving TL→TR→BR→BL always.

**Summary:**

| | v0.1 | v0.2 |
|---|---|---|
| Loss | SmoothL1, fixed corner order | Min-over-D4 SmoothL1 |
| Augmentation | h-flip, ±30° rotation | h-flip, 90°/180°/270°, ±30° rotation |
| Corner ordering at inference | As predicted by model | Geometric sort (centroid + angle) |
| Training frames | ~409k (phash ≤ 20) | ~316k (phash ≤ 15) |
| Epoch time | ~26 min (external drive) | ~5–6 min (SSD cache) |

---

### v0.3 — Heatmap architecture, 448×448 input, canonical corner order (current)

**Motivation:**

v0.1 and v0.2 both used Global Average Pooling after the encoder, compressing the 28×28 spatial
feature map to a 128-dimensional vector before regressing corner coordinates. This was the primary
bottleneck: all spatial information had to be encoded into 128 numbers, making precise corner
localization fundamentally difficult regardless of how many training epochs were run.

Additionally, horizontal flip augmentation was producing unrealistic training examples. MTG cards
are asymmetric — text, mana symbols, and artwork all have a natural orientation. A flipped card
image does not exist in the real world.

**Changes in v0.3:**

*Architecture — heatmap head replaces global regression.* The encoder output (28×28 at 448×448
input) feeds a 1×1 conv head that produces a 4-channel heatmap (one channel per corner). Soft-
argmax extracts differentiable (x, y) coordinates as the softmax-weighted mean of the spatial
grid. An auxiliary Gaussian BCE loss supervises the raw heatmap logits against Gaussian blob
targets at GT corner positions, providing dense spatial guidance.

*Input resolution — 224×224 → 448×448.* Doubling the input resolution doubles the effective
spatial resolution of the 28×28 feature maps at zero architectural cost. The 224×224 SSD cache
was deleted and rebuilt at 448×448.

*Augmentation — horizontal flip removed.* Flipping cards produces mirror-image text and artwork
that does not exist in the real world. The model should not train on physically impossible
examples.

*Loss — canonical order replaces dihedral invariance.* With heatmap channels fixed to specific
corners (channel 0 = TL, 1 = TR, 2 = BR, 3 = BL), the direct SmoothL1 loss on channel-matched
coordinates is correct and sufficient. Dihedral D4 permutation search is no longer needed.
After each spatial augmentation, `sort_corners_canonical` re-assigns corners to channels, keeping
channel assignments consistent.

**Summary:**

| | v0.2 | v0.3 |
|---|---|---|
| Input resolution | 224×224 | 448×448 |
| Architecture | Encoder → GAP → Linear(128→8+1) | Encoder → heatmap head → soft-argmax |
| Corner output | Direct regression (8 values) | Soft-argmax from 4-channel heatmap |
| Spatial precision | ~16px (after GAP: unlimited drift) | ~16px grid (28×28 at 448 input) |
| Heatmap aux loss | No | BCEWithLogitsLoss vs Gaussian targets, λ=10.0 |
| Occlusion handling | Confident wrong prediction | Diffuse heatmap peak (low confidence) |
| Loss | Min-over-D4 SmoothL1 | Direct SmoothL1 on canonical order |
| Augmentation | h-flip, 90°/180°/270°, ±30° | 90°/180°/270°, ±30° (no h-flip) |
| Per-corner confidence | No | sigmoid(heatmaps).amax(dim=(-2,-1)) |
| Parameters | ~44K | ~48K |
| Epoch time | ~5–6 min (224 SSD cache) | ~23 min (448 SSD cache) |

---

## Directory Structure

```
03_detector/
  README.md                     This file
  base.py                       DetectionResult dataclass + CardDetector ABC

  detectors/
    __init__.py                 Exports CannyPolyDetector, SIFTHomographyDetector
    canny_poly.py               Classical Canny + polygon detector
    sift_homography.py          SIFT feature-matching wrapper
    tiny_corner_cnn/
      __init__.py               Package marker
      model.py                  TinyCornerCNN (heatmap head + soft-argmax) + make_gaussian_heatmaps
      dataset.py                CornerDataset + load_from_packopening_db + load_clint_as_test
      train.py                  Training script (argparse, AdamW, cosine LR, checkpoints)
      predict.py                Inference wrapper implementing CardDetector ABC
      precache_dataset.py       Pre-resize all training frames to 448×448 on local SSD

  eval/
    __init__.py                 Package marker
    metrics.py                  CPE, PCK, IoU, exact polygon IoU, pHash distance
    benchmark.py                Evaluation harness (loads data, runs detectors, prints table)
    two_stage_test.py           Two-stage eval: full-image detect → crop → refine
    visualize_corners.py        Overlay predicted vs GT corners on sample images

  tests/
    test_geometry.py            Unit tests for coordinate conventions and sort_corners_canonical
```
