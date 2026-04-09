# Democratic Feature Voting for Card Centering and Authentication

## Problem Statement

Traditional card centering measurement relies on border widths: measure the gap between the printed image and the card edge on all four sides, compare left-to-right and top-to-bottom ratios. This works well for bordered cards where the border is the intended reference frame.

Borderless cards break this assumption entirely. A borderless card has no defined margin to measure against. Even semi-borderless treatments (showcase frames, extended art, full-art lands) often have borders so thin and stylistically irregular that pixel-counting them produces noisy, meaningless numbers. For these cards, "centered" has no clear definition — until now.

---

## Core Insight: Canonical Feature Placement via Democratic Voting

For any large print run of the same card, every copy *should* have its printed features (title text, mana cost symbols, power/toughness box, reminder text icons, set symbol, collector number, etc.) in exactly the same position relative to the card edge. In practice they don't, because of printing sheet misalignment — and that variance *is* the centering error we want to measure.

The key observation: **across a large population of the same card, the modal feature position is the intended position**.

If you photograph 500 copies of a borderless card and extract the pixel location of (say) the first letter of the card title, the resulting distribution will cluster tightly around the press-intended position. Outliers are miscut or miscentered copies. The cluster center *is* the canonical correct position — derived democratically from the evidence, not assumed from a spec sheet.

This is analogous to how GPS receivers use multiple satellite signals to triangulate a position more accurately than any single measurement. Each card copy is a noisy measurement of the same ground truth; voting averages out the noise.

---

## Feature Candidates

Features should be:
- **High-contrast and geometrically stable** — reliably detectable via keypoint matching or template matching
- **Spread across the card face** — features near the top and bottom of the card together enable rotational skew measurement (a single central feature only gives translation)
- **Present on every card in the category** — or at least every card in a well-defined subcategory

### Strong candidates for Magic: The Gathering

| Feature | Location on card | Notes |
|---|---|---|
| Title text first glyph | Upper area | Works for grouping by first letter; very high contrast |
| Mana cost pip cluster | Upper right | Distinctive shape; position encodes color identity |
| Set symbol | Lower right | Present on nearly all cards; shape is set-specific |
| Collector number digits | Lower left | High-contrast numerals; stable position |
| Power/toughness box | Lower right | Creatures only; very geometrically stable box shape |
| Artist credit line | Lower center | Thin but present; useful for vertical skew measurement |
| Text box border | Center | The border between artwork and rules text; strong horizontal line |
| Reminder text bullet | Varies | Less reliable, present only when reminder text exists |

For maximum rotational accuracy, pair a feature near the **top** of the card (e.g., title glyph centroid) with one near the **bottom** (e.g., collector number centroid or P/T box). The vector between the two feature positions defines the card's vertical axis; deviation from true vertical is the rotational skew.

---

## Algorithm Sketch

### Phase 1: Feature Extraction

For each card image in the population:

1. Dewarp and normalize the card to a canonical rectangle (the existing homography pipeline handles this).
2. Run a lightweight detector for each feature type. Options:
   - **Template matching** (OpenCV `matchTemplate`) for geometrically rigid features like the P/T box or mana pip shapes
   - **OCR bounding boxes** (e.g., Tesseract or a custom text detector) for title and collector number glyphs
   - **Keypoint detection** (SIFT/ORB) for set symbols, then match against a reference set-symbol template
3. Record the centroid of each detected feature in normalized card coordinates (0–1 in each axis), relative to the card bounding box.

### Phase 2: Canonical Position Estimation

For each (card group, feature type) pair:

1. Collect all per-copy feature centroids.
2. Compute the **robust cluster center** — use median rather than mean to resist outlier influence, or fit a Gaussian and take the mean of the inlier set (RANSAC-style).
3. Store this as the **canonical position** for that feature in that card group.

Card grouping strategies (from coarse to fine):
- **By card name**: every printing of "Black Lotus" forms one group
- **By set + frame treatment**: e.g., "Foundations borderless" as a group — useful when layout differs meaningfully between frame treatments
- **By first letter of title**: a practical bootstrap strategy when full name matching isn't yet available; all "S" cards share similar title-glyph horizontal offset even if their exact name differs

The first-letter grouping is particularly clever as a bootstrapping technique: it doesn't require recognizing the card first. You can cluster unlabeled card images by their first-glyph shape (a cheap classification problem — 26 classes), establish canonical horizontal position for each letter, and use that to score centering *before* full card identification.

### Phase 3: Per-Copy Centering Measurement

For a given card copy with known canonical positions:

1. Extract each feature centroid from the copy (same detectors as Phase 1).
2. Compute the residual vector: `actual_position - canonical_position` for each feature.
3. **Translation error**: the mean residual across all features gives the horizontal (left/right) and vertical (top/bottom) offset in normalized card units. Convert to millimeters using the card's physical dimensions (MTG standard: 63 × 88 mm).
4. **Rotational skew**: use the angle between the vector connecting a top feature and a bottom feature on this copy vs. the same vector in canonical space. Report in degrees.
5. **Axis-independent centering ratio** (traditional grading language): convert the translation error back to a PSA-style ratio like `50/50` (centered) vs. `55/45` (slightly off) by expressing the offset as a shift of the midpoint within the nominal border region.

### Phase 4: Grading Output

Report per-copy:
- Left/right centering ratio (front and back if both are photographed)
- Top/bottom centering ratio
- Rotational skew in degrees
- Per-feature confidence scores (how cleanly was each feature detected?)
- A composite centering grade, e.g., mapping to PSA/BGS thresholds

---

## Rotational Skew in Detail

Rotational skew ("diamond cutting" in collector parlance) is distinct from pure translation. A card can be perfectly centered left/right and top/bottom but have its printed image rotated relative to the card edge — the printed image is a rotated rectangle within the card rectangle.

To detect this:

```
top_feature_centroid    = (x_t, y_t)   # e.g., title glyph center
bottom_feature_centroid = (x_b, y_b)   # e.g., collector number center

card_axis_angle = atan2(x_b - x_t, y_b - y_t)   # angle of the print axis

canonical_card_axis_angle = atan2(x_b_canon - x_t_canon,
                                   y_b_canon - y_t_canon)

skew_degrees = card_axis_angle - canonical_card_axis_angle
```

A skew of even 0.5° is noticeable on high-grade cards. PSA 10 centering requires borders within 55/45 — a complementary skew threshold (e.g., < 0.3°) could be part of a holistic grade.

---

## Extension: Print Layer Authentication via Rosette Analysis

The same feature-alignment pipeline opens a second, more forensic application: **distinguishing print runs by their color registration signatures**.

### Background

Color offset printing (CMYK) lays down four or more ink layers in sequence. Even with precise press registration, each layer lands with a characteristic systematic offset — a fingerprint of the specific press, plate generation, and press run. This manifests as the **rosette pattern** visible under magnification: the overlapping halftone dots form a flower-like grid whose exact geometry varies by printing.

For Magic: The Gathering, this is already used by experienced collectors to distinguish:
- **Alpha/Beta**: very fine rosettes, slightly different dot pitch
- **Collector's Edition (CE)**: known registration signature distinct from Alpha/Beta
- **International Collector's Edition (ICE)**: different press, measurably different rosette alignment than CE — this is how rebacked CE/ICE cards (front from one printing, back from another) are detected even when the card itself looks visually perfect

### Mechanized Rosette Analysis

At a high zoom level (macro lens or flatbed scanner at 1200+ DPI):

1. Isolate a region of solid or near-solid color (a large mana symbol, a black border area).
2. Apply a 2D FFT to the region — the rosette lattice produces sharp peaks in frequency space at the halftone screen frequency and its harmonics.
3. Measure the **angle and spacing** of the dominant frequency peaks.
4. Compare against a reference library of known printing signatures.

This is essentially the same democratic-voting idea: build a reference database by scanning known authentic examples of each printing, extract their rosette signatures, and cluster them. Any new card can then be matched against the reference clusters.

### Practical Implementation Notes

- Minimum scan resolution: **1200 DPI** for rosette analysis; 2400 DPI preferred for high-value cards
- Region selection: avoid artwork areas (continuous tone, no halftone structure); prefer solid-color borders, mana symbols, or set symbol fills
- FFT window size: 256×256 or 512×512 pixels to get enough frequency resolution while staying within a color-uniform region
- Signature vector: peak frequency, peak angle, secondary peak ratio, dot shape descriptor (round vs. elongated)
- Classification: cosine similarity against stored signatures, or train a small classifier on a labeled dataset of known printings

### Authentication Use Cases

| Claim | Method | Observable |
|---|---|---|
| "This Alpha card" | Rosette angle + dot pitch vs. Alpha reference | Alpha has specific screen angle combination |
| "CE vs. ICE" | Rosette signature comparison | Different presses, different layer offsets |
| "Rebacked card" | Front rosette ≠ back rosette reference for same claimed set | Mismatch in registration fingerprint |
| "Counterfeit" | Rosette absent (inkjet/laser), or wrong pattern | Most counterfeits use wrong printing tech |
| "Trimmed border" | Centering + physical measurement discrepancy | Border ratios inconsistent with press data |

---

## Implementation Roadmap

### Stage 1 — Bootstrap (weeks 1–4)

- [ ] Build a feature extractor for 2–3 high-confidence features (title glyph, P/T box, set symbol)
- [ ] Collect a small labeled dataset: 20–50 copies each of 5–10 cards spanning bordered and borderless treatments
- [ ] Run the voting algorithm, visualize canonical positions, sanity-check against ground truth
- [ ] Measure translation error on copies with known (manually measured) centering

### Stage 2 — Scale (weeks 5–10)

- [ ] Integrate with the existing card-identification pipeline (dewarp → identify → look up canonical positions)
- [ ] Expand the feature detector to all major feature types
- [ ] Build the canonical position database for the top ~1000 most-graded cards
- [ ] Automate ingestion: every scan through the pipeline votes toward the canonical database

### Stage 3 — Grading Output (weeks 11–16)

- [ ] Implement the centering ratio and skew computation
- [ ] Design the grading rubric (thresholds for PSA 10 / 9 / 8 equivalent)
- [ ] Build a reporting UI or API endpoint

### Stage 4 — Rosette Authentication (parallel research track)

- [ ] Procure a high-DPI scanner (1200–2400 DPI flatbed)
- [ ] Build and label a rosette reference library for Alpha, Beta, CE, ICE
- [ ] Implement FFT-based signature extractor and similarity classifier
- [ ] Validate on known authentic / rebacked cards

---

## Why This Approach Is Better Than Manual Measurement

| Dimension | Manual (caliper/scan + ruler) | This approach |
|---|---|---|
| Borderless cards | Undefined / arbitrary | Fully defined via feature voting |
| Throughput | ~1–2 cards/min with skill | Automated, 1000s/day |
| Rotational skew | Hard to measure manually | Falls out naturally from two-feature vector |
| Consistency | Human variance, fatigue | Deterministic, reproducible |
| Authentication | Requires expert loupe work | Automated rosette analysis |
| Ground truth | Assumed from spec sheet | Derived from the population itself |

The "spec sheet" assumption is subtle but important: press-intended positions are not always publicly documented, and they may vary between print runs in ways not captured by any specification. Democratic voting derives the *empirical* standard directly from what was actually printed — which is the only standard that matters for grading.

---

## Open Questions

1. **How many samples are needed for a stable canonical estimate?** Preliminary expectation: 20–50 copies per card for < 0.1% uncertainty on feature position. Rarer cards may need interpolation from similar-frame cards.

2. **Does canonical position vary across print runs of the same card?** Possibly — a card reprinted in two different sets may have slightly different layout even if the art is the same. The grouping strategy should account for this (group by set + card name, not just card name).

3. **How to handle foil cards?** Foil introduces glare that disrupts feature detection. May need polarized lighting or multi-exposure HDR capture.

4. **How to handle card backs?** The MtG card back is the same on every card — an ideal reference. Back centering is simpler to measure but front/back agreement is also a grading criterion.

5. **Integration with physical grading standards?** PSA, BGS, and CGC each have proprietary centering criteria. This system should be designed to be *calibrated against* graded submissions, not to replace the graders — at least initially.
