# Pack Opening Video Dataset

Ingests YouTube pack-opening videos as training data for the CCG card
identification model. Each video is a natural source of real-world card images
with varied lighting, angles, glare, camera quality, and partial occlusion —
exactly the hard cases the model needs to learn.

The primary source channel is **[OpenBoosters](https://www.youtube.com/@OpenBoosters)**,
which produces high-quality single-set pack openings across nearly the entire
history of MTG.

---

## How It Works

```
YouTube video
    │
    ▼
01  Extract clean keyframes (ffmpeg I-frames, then blur filter)
    │
    ▼
02  For each frame: run SIFT homography against every card in the target set
    │  (set is known from the video title — typically 150–400 cards)
    │
    ├─ Match with 4 corners in-frame → KEEP
    └─ No match / corners off-frame   → DISCARD
    │
    ▼
03  Store aligned crop + corner coords + card_id in dataset
```

The key design choice is to use **slow but accurate SIFT homography** rather
than a fast approximate hash. Because the target set is known from the video
title, we only search ~150–400 reference images rather than the full 108k
gallery, making the per-frame search tractable. We pre-compute SIFT features
for every card in the set once (cached on disk), then match each frame against
that pre-built index.

We deliberately throw away a lot of frames. Any frame that doesn't produce a
clean 4-corner homography match is discarded. A conservative filter that keeps
only confident results is better than a permissive one that keeps noisy labels.

---

## Video Registry (Google Sheets)

The master list of videos to process lives in a public Google Sheet:

**[YouTube Card Recognition Datasets](https://docs.google.com/spreadsheets/d/1mgBQxvJa_GMrGSuhx5qzGJBFLM8GJ4A53JbximUdAxQ)**

### Sheet structure

**Tab 1 — Video Registry** (one row per video):

| Column | Description |
|---|---|
| `video_id` | YouTube video ID (the part after `?v=`) |
| `url` | Full YouTube URL |
| `channel` | Channel name (e.g., `OpenBoosters`) |
| `title` | Video title (auto-populated by `00_fetch_channel.py`) |
| `set_codes` | Comma-separated Scryfall set codes (e.g., `lea` or `otj,otp,big`) |
| `set_notes` | Freeform (e.g., "pre-release kit — includes buy-a-box promo") |
| `status` | `new` → `pending` → `downloading` → `downloaded` → `frames_extracted` → `processing` → `done` / `error` / `skip` / `needs_review` |
| `frames_extracted` | Count of clean frames extracted |
| `frames_matched` | Count of frames with confirmed card match |
| `contributor` | Who added this row |
| `added_date` | ISO date |
| `processed_date` | ISO date when pipeline last ran |
| `notes` | Freeform |

**Tab 2 — Processing Log** (auto-written by the pipeline): one row per
processing run, with timestamp, video_id, counts, and any error messages.
Humans do not edit this tab.

The Sheet is intentionally small (one row per video, not one row per frame).
Frame-level data lives in the local SQLite database (see below), not in the
Sheet. This keeps the Sheet fast and collaborative while the heavy data stays
local.

### Contributing videos

Anyone with the link can suggest a video by adding a row to Tab 1 with status
`pending` and filling in `url`, `set_codes`, and `contributor`. If you're not
sure of the exact set codes, leave a note in `set_notes` and someone will
confirm. Set codes follow [Scryfall's convention](https://scryfall.com/sets)
(lowercase 3–5 letters).

---

## Local Data Storage (SQLite)

Frame-level data is stored in a SQLite database at:

```
{cfg.data_dir}/datasets/packopening/packopening.db
```

Schema:

```sql
-- One row per video (mirrors Sheet Tab 1)
CREATE TABLE videos (
    video_id        TEXT PRIMARY KEY,
    slug            TEXT UNIQUE NOT NULL,   -- filesystem-safe name
    url             TEXT NOT NULL,
    channel         TEXT,
    title           TEXT,
    set_codes       TEXT,                   -- comma-separated
    status          TEXT DEFAULT 'pending',
    added_date      TEXT,
    processed_date  TEXT,
    notes           TEXT
);

-- One row per good matched frame
CREATE TABLE frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT REFERENCES videos(video_id),
    frame_path      TEXT NOT NULL,          -- relative to cfg.data_dir
    aligned_path    TEXT,                   -- relative to cfg.data_dir
    card_id         TEXT NOT NULL,          -- Scryfall UUID
    illustration_id TEXT,
    set_code        TEXT,
    num_matches     INTEGER,                -- SIFT inlier count
    corner0_x REAL, corner0_y REAL,
    corner1_x REAL, corner1_y REAL,
    corner2_x REAL, corner2_y REAL,
    corner3_x REAL, corner3_y REAL,
    matching_area_pct REAL,
    blur_score      REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_frames_card ON frames(card_id);
CREATE INDEX idx_frames_video ON frames(video_id);
```

This database is the authoritative store. The manifest CSV for training is
generated from it on demand by `05_build_manifest.py`.

---

## Directory Layout

All paths relative to `cfg.data_dir`:

```
datasets/packopening/
  packopening.db                    SQLite database (all results)

  sift_cache/{set_code}/            Pre-computed SIFT features per card
    {card_id}.npz                   keypoints + descriptors for one reference image

  raw/{slug}/                       Downloaded video files
    {slug}.mp4
    {slug}.info.json                yt-dlp metadata

  frames/{slug}/                    Extracted keyframes
    frame_{pos:06d}.jpg             Named by source frame position (idempotent)

  aligned/{slug}/                   Dewarped card crops (the training images)
    {card_id}_{frame_pos:06d}.jpg

  manifest.csv                      Generated training manifest (ManifestRow-compatible)
  corners.csv                       Generated corner data (same schema as clint_backgrounds)
```

---

## Pipeline Scripts

Run from the project root. All scripts accept `--help`.

Scripts are split into three stages:

```
02_data_sets/packopening/code/
  db.py                   Shared SQLite helpers (schema, open_db, upsert_video, …)

  # Stage 1 — Registry (no API key needed)
  01_fetch_channel.py     Enumerate all videos on a YouTube channel → SQLite (yt-dlp)

  # Stage 2 — Annotation (requires ANTHROPIC_API_KEY + credits)
  02_annotate.py          Classify video titles with Claude: is it MTG? which sets?
                          Writes set_codes + status back to SQLite.

  # Stage 3 — Per-video ingestion (CPU-intensive; run after annotation)
  pipeline/
    01_download.py        Download a video (or all pending) with yt-dlp
    02_extract_frames.py  Extract clean I-frames; filter by blur score
    03_precompute_sift.py Pre-compute SIFT features for all cards in a set (cached)
    04_match_frames.py    Run SIFT homography; keep 4-corner matches; write to DB
    05_build_manifest.py  Export manifest.csv + corners.csv from DB for training
```

### Typical workflow

```bash
source .venv312/bin/activate

# 1. Fetch the full video list for OpenBoosters (no API key needed)
python 02_data_sets/packopening/code/01_fetch_channel.py
# → adds ~2000 videos to SQLite with status 'pending', empty set_codes

# 2. Annotate with Claude (needs ANTHROPIC_API_KEY in .env)
python 02_data_sets/packopening/code/02_annotate.py
# → fills in set_codes; marks confidently non-MTG videos as 'skip'
# → low-confidence results get status 'needs_review' for manual checking

# 3. Download one video to test the pipeline
python 02_data_sets/packopening/code/pipeline/01_download.py --slug <slug>

# 4. Extract frames
python 02_data_sets/packopening/code/pipeline/02_extract_frames.py --slug <slug>

# 5. Pre-compute SIFT for the set (once per set, then cached)
python 02_data_sets/packopening/code/pipeline/03_precompute_sift.py --set-code lea

# 6. Match frames to cards
python 02_data_sets/packopening/code/pipeline/04_match_frames.py --slug <slug>

# 7. Rebuild training manifest
python 02_data_sets/packopening/code/pipeline/05_build_manifest.py
```

### Batch processing all pending videos

```bash
python 02_data_sets/packopening/code/pipeline/01_download.py --all
python 02_data_sets/packopening/code/pipeline/02_extract_frames.py --all
# (SIFT precompute per set as needed)
python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all
python 02_data_sets/packopening/code/pipeline/05_build_manifest.py
```

All scripts support resuming interrupted runs. Re-running a completed step is
safe (idempotent).

---

## Frame Extraction Details

### Why I-frames only

FFmpeg can extract frames at fixed time intervals, but for compressed video this
often produces partially-decoded frames at B/P-frame boundaries, resulting in
interlacing artifacts or ghosting. Using `-skip_frame nokey` forces FFmpeg to
output only fully self-contained I-frames (GOP keyframes), which are always
clean.

The trade-off: I-frames are sparser (typically every 0.5–5 seconds depending on
the encoder settings). For pack-opening videos where each card is held still for
2–10 seconds, this is usually sufficient — we expect 2–8 I-frames per card.

```bash
# What the script runs under the hood:
ffmpeg -skip_frame nokey -i input.mp4 -vsync vfr -frame_pts true \
  -q:v 2 frames/frame_%06d.jpg
```

### Blur filtering

After extraction, each frame is scored for sharpness using the Laplacian
variance method (`cv2.Laplacian(gray, cv2.CV_64F).var()`). Frames below a
threshold (default: 100.0) are excluded from SIFT matching. This removes motion
blur from card reveals and focus transitions.

---

## SIFT Matching Details

### Pre-computation (once per set)

Running SIFT feature extraction on a 1040×745 Scryfall reference PNG takes
~0.3 s. For a set of 300 cards that's 90 s total. These features are saved as
`.npz` files in `sift_cache/{set_code}/` and reused across all videos from
that set.

### Per-frame matching

For each frame:

1. Extract SIFT features from the frame (~0.3 s).
2. For each card in the set (~150–400 cards):
   - Run FLANN matching + Lowe's ratio test against pre-loaded descriptors.
   - If `num_good_matches >= MIN_MATCHES` (default: 20), attempt homography.
   - Homography accepted if all 4 corners fall within the frame bounds.
   - Record `num_good_matches` and corner coordinates.
3. Keep the card with the highest `num_good_matches` that passes the 4-corner
   test. Discard the frame entirely if no card passes.

**Typical timing:** ~0.05 s per card × 300 cards = ~15 s per frame.
A video yielding 60 usable frames takes ~15 minutes to match.

### Quality thresholds (conservative by design)

| Check | Threshold | Rationale |
|---|---|---|
| `blur_score` | `>= 100` | Exclude motion-blurred frames |
| `num_good_matches` | `>= 20` | Require strong feature agreement |
| Corners in-frame | all 4 | Partial cards give poor training signal |
| `matching_area_pct` | `0.05 – 0.95` | Card too small or homography degenerate |
| Margin over 2nd-best | `>= 5 matches` | Avoid ambiguous frames (multiple cards) |

These thresholds favour precision over recall. It is better to discard a valid
frame than to keep a mislabelled one.

---

## Set Code Inference from Video Title

`00_fetch_channel.py` attempts to infer set codes from video titles
automatically using a mapping of known set names and abbreviations against
the Scryfall sets list. Examples:

- "Opening an Alpha booster pack" → `lea`
- "Legends Pack Opening" → `leg`
- "OTJ Pre-Release Kit" → `otj, otp, big` (main + associated promo sets)

The inferred codes are written as a suggestion to the Sheet with status
`needs_review`. A human confirms or corrects before the video is processed.
This prevents wrong-set matches from contaminating the training data.

---

## Suggestions for Improving Coverage

### Multi-set videos

Some videos open pre-release kits, which include a main set plus associated
promo packs (e.g., OTJ + OTP + BIG + SPG). Set multiple codes in `set_codes`
(comma-separated). The SIFT matcher will search against the union of all listed
sets.

### Non-English videos

If the video contains non-English cards, add a `lang` field to the video row.
The pipeline will search against the appropriate language card images from
Scryfall (when available). Most early sets only have English scans in Scryfall.

### Verified vs. unverified frames

Consider adding a `verified` boolean column to the `frames` table. High-match
frames (e.g., `num_matches >= 50`) can be auto-verified; lower-confidence
matches are flagged for manual spot-checking. Only verified frames enter the
training manifest. This allows the confidence threshold to be lowered slightly
to capture more data without reducing overall quality.

---

## Open Data

If you process videos and would like to contribute the resulting SQLite database
(frame metadata, not the video files themselves), open a GitHub issue or PR.
The frame images themselves cannot be redistributed (YouTube ToS / copyright),
but the extracted corner coordinates and card ID mappings are factual metadata
with no copyright claim.

---

## Prerequisites

```bash
# yt-dlp (video downloader)
pip install yt-dlp

# Python packages (already in .venv312)
pip install opencv-python-headless gspread imagehash

# ffmpeg (must be in PATH)
brew install ffmpeg   # macOS
```

Google Sheets sync requires a service account JSON key. See
`code/SHEETS_SETUP.md` for instructions on obtaining one (read/write access to
the registry sheet only).
