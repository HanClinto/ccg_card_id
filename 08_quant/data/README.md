# CCG Card ID — Edge Inference Package

Identifies Magic: The Gathering cards from images using two ONNX models:

1. **Corner detector** — locates card corners in a raw photo/video frame
2. **Identifier** — embeds the card image and looks it up in a 108 k-card gallery

No PyTorch required. Runs on Raspberry Pi, desktop, or any platform with `onnxruntime`.

---

## Install

```bash
pip install -r requirements.txt
```

Minimum Python 3.9. On Raspberry Pi, prefer the `onnxruntime` wheel from PyPI
(not the Debian package) for the latest version:

```bash
pip install onnxruntime numpy opencv-python psutil
```

---

## Files

| File | Size | Description |
|---|---|---|
| `benchmark.py` | — | CLI benchmark tool |
| `detector_*.onnx` + `.onnx.data` | 8 MB | SimCC corner detector (384×384 input) |
| `identifier_*.onnx` + `.onnx.data` | 5 MB | ArcFace identifier (448×448 input) |
| `mobilevit_xxs_ft_*_gallery.npz` | 53 MB | 108,354 × 128-d gallery embeddings |

The `.onnx` and `.onnx.data` files for each model **must stay in the same directory** —
the `.data` file contains the weight tensors referenced by the `.onnx` graph.

---

## Usage

### Pre-cropped images (phone scans, flatbed scans)

If your images are already cropped to just the card face, skip corner detection:

```bash
python benchmark.py \
    --identifier identifier_*.onnx \
    --gallery    mobilevit_xxs_ft_*_gallery.npz \
    --images     /path/to/images/ \
    --skip-detect
```

### Raw camera frames (full pipeline)

For unprocessed photos or video frames where the card appears in a scene:

```bash
python benchmark.py \
    --detector   detector_*.onnx \
    --identifier identifier_*.onnx \
    --gallery    mobilevit_xxs_ft_*_gallery.npz \
    --images     /path/to/frames/ \
    --top-k 3 \
    --csv results.csv
```

### Detector only (no identification)

To measure detection speed or test corner output without running the identifier:

```bash
python benchmark.py \
    --detector detector_*.onnx \
    --images   /path/to/frames/
```

---

## Output columns

| Column | Description |
|---|---|
| `detect_ms` | Corner detection time |
| `sharpness` | SimCC mean peak (0–1): how focused the corner distributions are; low = blurry/no card |
| `presence` | Sigmoid of the raw card-presence logit (0–1) |
| `dewarp_ms` | Perspective warp time |
| `identify_ms` | Embedding inference time |
| `lookup_ms` | Gallery dot-product search time |
| `total_ms` | End-to-end time |
| `rss_mb` | Process RSS memory at time of image |
| `gt_card_id` | Ground-truth Scryfall UUID (from filename, if present) |
| `top1_card_id` | Top-1 match Scryfall UUID |
| `top1_score` | Cosine similarity to top-1 match (higher = more confident) |
| `correct` | 1 if top-1 matches ground truth, 0 otherwise (only if GT present) |

---

## Measuring accuracy — do images need to be labelled?

**Labelling is optional.** The benchmark always runs inference and reports timings
regardless of how files are named.

The `correct` column is only computed when the image filename contains a
Scryfall UUID (the `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` format). The UUID can
appear anywhere in the name:

```
# These all produce a correct/incorrect result:
e7ee2fa1-5aed-4612-bfa9-cfbf5d282c9b.jpg
evolvingwilds_C15_nosleeve_e7ee2fa1-5aed-4612-bfa9-cfbf5d282c9b_001.jpg
scan_e7ee2fa1-5aed-4612-bfa9-cfbf5d282c9b_front.png

# These run inference but skip the accuracy column:
my_card.jpg
unknown_001.png
```

To look up a card's Scryfall UUID, search [scryfall.com](https://scryfall.com) and
copy the UUID from the card URL:
`https://scryfall.com/card/c15/237/evolving-wilds` → click the card image →
the UUID appears in the full-resolution URL.

Alternatively, the Scryfall bulk JSON (`all_cards.json`) maps every card to its UUID.

---

## Expected performance

Measured on Raspberry Pi 4 (ARM Cortex-A72, 4-core, FP32):

| Stage | Approx. time |
|---|---|
| Corner detection | 300–600 ms |
| Perspective dewarp | < 5 ms |
| Embedding inference | 300–600 ms |
| Gallery lookup (108 k) | 5–15 ms |
| **Total (full pipeline)** | **~700–1200 ms** |

On Raspberry Pi 5 (Cortex-A76) expect roughly 2× faster.
INT8 quantised models (`*_int8.onnx`, produced by `quantize_int8.py`) give a
further 2–4× speedup on CPU.

Accuracy on clean phone scans (daniel_scans dataset):
- Top-1 artwork match: **100%**
- Top-1 edition (exact printing): **60%**
- Top-3 edition: **~75%**
