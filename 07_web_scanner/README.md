# CCG Card ID — Web Scanner

A real-time Magic: The Gathering card scanner with a REST API backend and a
single-page browser client. Point your phone or webcam at a card; the server
detects corners, dewarps the image, and identifies the card against an 81k
Scryfall reference gallery.

---

## Architecture

### MVP (current) — server-side everything

```
Browser
  └─ JPEG frame (base64)
       └─> POST /v1/identify
             ├── decode image
             ├── detect corners  (Canny or Neural CNN)
             ├── dewarp          (cv2.getPerspectiveTransform)
             └── identify card   (pHash or ArcFace gallery search)
                   └─> card_name, set_code, price ...
```

All heavy work is on the server.  The client is a thin HTML/JS page that
grabs camera frames, sends them, and displays results.  Good for a local
development machine where server and browser are on the same WiFi network.

### Final version — client-side preprocessing, server-side lookup only

Planned future evolution:

```
Browser (ONNX/WASM)
  ├── corner detection     (tiny_corner_cnn exported to ONNX)
  ├── dewarp               (WebGL perspective transform)
  ├── pHash computation    (pure JS or WASM)
  └─> POST /v1/lookup      (compact hash only — no raw image upload)
        └── Hamming search against server vector DB
```

Benefits: lower bandwidth (hash vs full JPEG), privacy (image never leaves
device), works on slow connections.  ArcFace embedding export is also
possible once the model is small enough for WASM inference.

---

## API Reference (Ximilar-compatible format)

The `/v1/identify` endpoint mirrors the Ximilar Visual AI request/response
shape so that client code can be swapped between services.

### POST /v1/identify

Request body:
```json
{
  "records": [
    {
      "_base64": "<base64-encoded JPEG or PNG>",
      "detector":   "canny",
      "identifier": "phash_16x16"
    }
  ]
}
```

`_base64` — also accepted as `base64` (no leading underscore).
`detector` and `identifier` are optional; the server's configured defaults
are used when omitted (see `GET /v1/defaults`).

Response:
```json
{
  "records": [
    {
      "_status":           {"code": 200, "text": "OK"},
      "card_present":      true,
      "corners":           [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
      "card_name":         "Sol Ring",
      "set_code":          "ltr",
      "set_name":          "The Lord of the Rings: Tales of Middle-earth",
      "scryfall_id":       "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "tcgplayer_id":      123456,
      "tcgplayer_price_usd": 1.50,
      "confidence":        0.92,
      "identifier_used":   "phash_16x16",
      "detector_used":     "canny"
    }
  ],
  "_status": {"code": 200, "text": "OK"}
}
```

Corner coordinates are normalized [0, 1] relative to image width/height,
in TL → TR → BR → BL order.

When no card is detected: `card_present: false` and all identification
fields are absent.

### GET /v1/health

```json
{"status": "ok", "gallery_loaded": true, "detectors_available": 2, "identifiers_available": 3}
```

### GET /v1/detectors

```json
{"detectors": [{"name": "canny", "label": "Canny edge detector"}, ...]}
```

### GET /v1/identifiers

```json
{
  "identifiers": [
    {"name": "phash_16x16",  "label": "pHash 16x16 (256-bit, 32 B/card)", "bytes_per_card": 32},
    {"name": "phash_32x32",  "label": "pHash 32x32 (1024-bit, 128 B/card)", "bytes_per_card": 128},
    {"name": "arcface_illustration_id_e75", "label": "ArcFace epoch 75 (128-d)", "bytes_per_card": 512}
  ]
}
```

### GET /v1/defaults

```json
{"detector": "canny", "identifier": "phash_16x16"}
```

---

## How to run

### 1. Install dependencies

```bash
cd 07_web_scanner/server
pip install -r requirements.txt
```

All packages except `torch` install quickly.  If you already have the
project's `.venv312` activated you probably have most of them already.

### 2. Populate prices (one-time)

Reads `default_cards.json` from `cfg.data_dir` and adds a `prices` table to
the catalog SQLite database:

```bash
python 07_web_scanner/server/populate_prices.py
```

Re-running is safe — it drops and recreates the table.

### 3. Start the server

```bash
# Default: Canny detector, pHash 16x16 identifier, port 8000
python 07_web_scanner/server/app.py

# Specify detector and identifier
python 07_web_scanner/server/app.py \
    --detector canny \
    --identifier phash_16x16 \
    --port 8000

# Different host (e.g. bind to all interfaces for phone access)
python 07_web_scanner/server/app.py --host 0.0.0.0 --port 8000
```

### 4. Open the client

Open `07_web_scanner/client/index.html` in your browser, or navigate to
`http://localhost:8000/app` (the server also serves the client directory).

Set the Server URL in the settings panel to match your host and port (default
`http://localhost:8000`).

---

## File structure

```
07_web_scanner/
├── README.md                  This file
├── server/
│   ├── requirements.txt       Python dependencies
│   ├── populate_prices.py     One-time price migration from default_cards.json
│   ├── card_lookup.py         CardLookup — SQLite lookup by scryfall_id
│   ├── search.py              PHashSearch, ArcFaceSearch, GallerySearchManager
│   └── app.py                 FastAPI server + CLI entrypoint
└── client/
    ├── index.html             Single-page app shell
    ├── scanner.js             Camera capture, ScanBucket deduplication, UI
    └── style.css              Dark theme, two-column layout
```

---

## Bucket deduplication — the "grocery scanner" model

The scanner sends frames at 1–10 FPS.  A single card appears in many
consecutive frames.  We do not want to log it dozens of times.

`ScanBucket` implements a simple state machine:

```
State: {candidate: null | {scryfall_id, count}, cooldowns: Set}

For each incoming result:
  null result (no card detected)
    → drain count by drainPerMiss (default 1)
    → if count reaches 0, reset candidate

  result.scryfall_id is in cooldowns
    → ignore (already logged recently)

  result.scryfall_id matches current candidate
    → increment count
    → if count >= fillAt (default 3): CONFIRM card, add to log, enter cooldown

  result.scryfall_id is different card
    → reset candidate to new card, count = 1
```

**fillAt=3** means we need 3 consecutive matching identifications before
logging.  This rejects spurious single-frame detections.

**cooldownMs=4000** means the same card won't be logged again for 4 seconds
after confirmation, preventing repeated scans of a stationary card.

**drainPerMiss=1** means one missed frame increments the counter back toward
zero — the candidate is abandoned only after `fillAt` consecutive misses.

Tune these via the settings panel in the client.

---

## Gallery NPZ layout

The server expects pre-computed gallery NPZ files built by the project's
`05_build/` or `06_eval/` pipelines.

**pHash galleries** live at:
```
{data_dir}/vectors/phash/gallery_manifest_artwork_id_manifest/
    phash_8x8_64bit_gallery.npz
    phash_16x16_256bit_gallery.npz
    phash_32x32_1024bit_gallery.npz
```
Each NPZ contains `embeddings` (n, bytes) uint8.

**ArcFace galleries** live at:
```
{data_dir}/vectors/mobilevit_xxs/img224/gallery_manifest_artwork_id_manifest/
    mobilevit_xxs_ft_illustration_id_e75_128d_gallery.npz
```
Each NPZ contains `embeddings` (n, 128) float32, already L2-normalized.

Both gallery types are indexed in the same order as the rows of
`{data_dir}/mobilevit_xxs/artwork_id_manifest.csv`.
