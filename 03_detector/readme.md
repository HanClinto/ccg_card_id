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

**Stage 2 — Identify** (see `04_build/` and `06_eval/`). The dewarped card image is passed to the
ArcFace metric-learning model, which embeds it into a 128-dimensional vector and performs nearest-
neighbour lookup against a gallery of ~108k Scryfall reference embeddings. The closest match is
the card identity.

The two stages are deliberately separate. Detection is a geometric problem (where is the card?),
while identification is a semantic one (which card is it?). Keeping them separate makes each
component easier to evaluate, replace, and improve independently.

---

## Why Detection Is Hard

For clean studio shots — white background, good lighting, a single bordered card placed flat — the
problem is almost trivial. A simple Canny edge detector finds the rectangular outline and you are
done. Real-world captures are far less cooperative.

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
most prominent or most central) reliably.

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

Requires `opencv-contrib-python` or `opencv-contrib-python-headless` for SIFT.

### NeuralCornerDetector

A learned detector that takes any image and predicts corners directly, with no gallery required.

**Architecture**: MobileViT-XXS backbone (the same one used in the card identification model in
`04_build/`) followed by a regression head. The backbone produces a 384-dimensional feature vector
via global average pooling. The head is:

```
LayerNorm(384) → Linear(384, 256) → GELU → Dropout(0.3) → Linear(256, 9)
```

The 9 output logits are split: the first 8 pass through sigmoid to produce 4 corner `(x, y)`
pairs, and the ninth is a raw card-presence logit (apply sigmoid for the probability).

**Loss**: Two terms. A BCE loss on the presence logit (applied to all frames, including negatives)
and a SmoothL1 regression loss on the corner predictions (applied only to positive frames where a
card is present). The corner loss is weighted by `lambda_corners=5.0`.

**Transfer learning**: Because the card-ID model and the detection model share the same backbone,
the detection model can be seeded from a pre-trained card-ID checkpoint. The backbone has already
learned to extract rich card-related features; the detection head then only needs to learn the
regression mapping. In practice this substantially reduces the number of epochs needed to reach a
good result. Pass `--seed-checkpoint` at training time.

**Strengths**: Handles all the difficult cases (borderless art, foil, sleeves, clutter) once
trained. No gallery required. Single forward pass. Runs well on MPS/GPU.

**Weaknesses**: Requires training data and a GPU for practical training. Currently trained only on
`clint_cards_with_backgrounds` (1,267 frames) — small by neural-network standards. Performance
will improve substantially as more labeled data is added from the packopening pipeline.

---

## Training Data

The primary labeled dataset is `corners.csv`, located at:

```
datasets/clint_cards_with_backgrounds/data/04_data/corners.csv
```

This file contains 1,267 rows, each one a phone-video frame from the
`clint_cards_with_backgrounds` dataset. Each row stores the `img_path` (relative to `cfg.data_dir`),
`card_id`, and the four normalized corner coordinates produced by the SIFT homography detector.
Since SIFT was used to generate these labels, they are highly accurate where available — the SIFT
pipeline requires a minimum of 20 good feature matches and a geometric consistency check before
accepting a result.

Negative examples (frames where SIFT failed or no card is present) live in:

```
datasets/clint_cards_with_backgrounds/data/04_data/bad/
```

There are 62 such frames. They are used as hard negatives during training (the model should output
`card_present=False` for these).

The training split is small (roughly 1,000 positives plus 62 negatives), so augmentation is
critical. The `CornerDataset` class applies random horizontal flips, rotations up to ±30°, color
jitter, random grayscale, and Gaussian blur. Spatial augmentations update the corner coordinates
alongside the image, so labels stay correct.

The packopening pipeline (see `02_data_sets/packopening/`) processes hundreds of pack opening
videos and will eventually yield tens of thousands of additional labeled frames. Once the neural
detector is accurate enough to bootstrap its own labels (predict corners, verify with SIFT on easy
frames, propagate to hard frames), this dataset can grow quickly. That bootstrapping loop is
planned but not yet implemented.

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

## Running the Benchmark

```bash
# Activate the venv first
source .venv312/bin/activate

# Evaluate CannyPolyDetector on the val split (default)
python 03_detector/eval/benchmark.py

# Evaluate both Canny and a trained neural detector
python 03_detector/eval/benchmark.py \
    --detectors canny,neural \
    --neural-checkpoint /path/to/corner_detector/last.pt

# Evaluate on all data (not just val split)
python 03_detector/eval/benchmark.py --split all

# Quick smoke-test: only 20 images per split
python 03_detector/eval/benchmark.py --limit 20

# Override data directory
python 03_detector/eval/benchmark.py \
    --data-dir /Volumes/carbonite/claw/data/ccg_card_id
```

Expected output format:

```
Detector             | CPE↓   | PCK@5%↑ | IoU↑   | Det.Rate↑ | FPR↓
CannyPolyDetector    | 0.042  | 71.3%   | 0.81   | 88.4%     | 12.5%
NeuralCornerDetector | 0.018  | 93.7%   | 0.94   | 97.2%     |  3.1%
```

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
    neural/
      __init__.py               Package marker
      model.py                  NeuralCornerDetector (MobileViT-XXS + regression head)
      dataset.py                CornerDataset + load_dataset
      train.py                  Training script (argparse, AdamW, cosine LR, checkpoints)
      predict.py                Inference wrapper implementing CardDetector ABC

  eval/
    __init__.py                 Package marker
    metrics.py                  CPE, PCK, IoU, exact polygon IoU
    benchmark.py                Evaluation harness (loads data, runs detectors, prints table)
```
