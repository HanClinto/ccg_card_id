# Phase 2 Data Plan (MTG confusing-card disambiguation)

Date: 2026-02-20

## 1) What is already available locally (usable now)

Inspected: `/Users/hanclaw/claw/data/ccg_card_id/datasets`

### High-value datasets for immediate use

1. **`clint_cards_with_backgrounds`**
   - Size: ~2.6 GB
   - Real-camera frames with homography/alignment pipeline outputs.
   - `04_data/good`: **1270** images (usable positives)
   - `04_data/bad`: **62** images (hard negatives / failure cases)
   - `03_reference`: **40** reference card IDs (UUID filenames)
   - `04_data/masks/{224,512}` + `resized/{224,512}` each have **1270** files (useful for synthetic compositing / controlled augmentations)
   - Notes:
     - Filename format includes UUID + card token (e.g. `..._CityOfBrass_...jpg`), so labels are recoverable.
     - Existing `dataset.tsv` currently only has header, so manifest must be rebuilt from filenames.

2. **`clint_cards_solring`**
   - Size: ~1.2 GB
   - `04_data/good`: **307** images across **21** Sol Ring printings/set variants (strong for “same card, different print” confusion)
   - `04_data/bad`: **12** negatives
   - Useful as a tightly controlled confusion benchmark.

3. **`daniel_scans`**
   - Size: ~329 MB
   - `images_processed`: **150** images
   - Contains 4 cards (`evolvingwilds`, `terramorphicexpanse`, `giantgrowth`, `fireball`) across many set codes + sleeve conditions.
   - Very useful for confusing cards and condition/sleeve variation.

### Potential but not immediate

4. **`tcgplayer_listos`**
   - Size: ~76 MB
   - Contains metadata/scripts and xml, plus `listos_dataset.tsv` with **46,954** rows (**45,113** unique `tcgid`).
   - **No corresponding local image files** currently present for those rows (`img_path` targets missing).
   - `listos_info.db` has listing metadata (including listing URLs, conditions, image IDs), but this may include user text and noisy labels.

5. **`munchie`**
   - Size: ~5.5 GB, mostly zipped assets and JSON
   - `raw.zip` (1659 files), `munchie_fronts.zip` (838), `backs.zip` (768)
   - Potentially useful later, but needs curation/extraction and label quality review.

---

## 2) TCGplayer Listos feasibility (legal/ToS-safe)

## Practical conclusion
Direct unattended scraping of TCGplayer listing pages/images is **high risk** from compliance and operational perspectives.

### Why risky
- Existing legacy scripts rely on crawling listing pages and deriving image URLs from internal patterns.
- TCGplayer endpoints may be protected/rate-limited and terms may restrict automated extraction/reuse.
- Listing content may include user-generated text/data and potentially personal/seller content, adding policy/compliance overhead.

### Safer path for Listos-style data
1. **Permission-first**: request written approval from TCGplayer for research/training usage and collection method.
2. **If approval is not available**: avoid broad scraping. Use opt-in, user-provided captures/exports instead.
3. **Treat Listos metadata as candidate index only** until legal access is clarified.

---

## 3) Alternatives if direct scraping is risky

## A) Scryfall API + owned capture pipeline (best legal baseline)
- Pros: clear API documentation, stable, high quality canonical card images, easy scale.
- Cons: canonical scans are not real-camera photos.
- Use for: synthetic positives, reference images, retrieval candidates.

## B) User-contributed capture workflow (recommended medium-term real positives)
- Build a tiny upload/capture flow (phone web UI):
  - capture front image(s)
  - choose card from candidate list (or scan fallback)
  - record set/finish/sleeve/lighting metadata
  - require consent checkbox for ML training
- Pros: legally clean if consented, directly matches deployment domain.
- Cons: slower bootstrapping; requires lightweight tooling.

## C) Controlled in-house capture sessions
- Run 2-3 short sessions with target confusion sets (e.g., 10–20 hard card groups).
- Standardized backgrounds + random lighting/sleeves/angles.
- Pros: high label quality, fast iteration, low legal ambiguity.

## D) Marketplace partner/export workflow
- Ask sellers/shops to export images + labels under explicit data-sharing agreement.
- Pros: genuine “in-the-wild” quality and volume.
- Cons: relationship/ops overhead.

---

## 4) Recommended 1–2 week execution plan

## Week 1 (bootstrap now)
1. **Build local real manifest from current datasets**
   - Use helper script added in this task:
     - `04_vectorize/mobilevit_xxs/x_build_local_real_manifest.py`
   - Includes:
     - positives: `clint_cards_with_backgrounds/good`, `clint_cards_solring/good`, `daniel_scans/images_processed`
     - negatives: `clint_cards_with_backgrounds/bad`

2. **Train baseline with mixed data**
   - Positives:
     - synthetic/card-scan positives from existing Scryfall pipeline
     - real-camera positives from local datasets above
   - Negatives:
     - hard negatives from `.../bad`
     - inter-class negatives from confusing card groups

3. **Hard-negative mining loop (2–3 cycles)**
   - Evaluate confusion matrix
   - Append top misclassifications as weighted negatives
   - Retrain quickly with increased sampling weight for hard pairs

4. **Create target confusion packs**
   - Start with known confusable pairs/groups (same art/similar names/reprints)
   - Ensure each group has at least:
     - 20+ train images/class where possible
     - 5+ validation images/class

## Week 2 (real-data expansion)
1. **Launch opt-in capture workflow** (lightweight)
   - 1-page uploader with consent + structured label form.
2. **Collect 1k–3k real positives** focused on hardest confusion buckets.
3. **Quality gate**
   - auto checks: blur, card area %, OCR sanity
   - manual review for 5–10% sample
4. **Retrain + evaluate on “hard set” benchmark**
   - keep a frozen benchmark slice from `clint_*` + `daniel_scans`
   - track top-1/top-k and per-confusion-pair error deltas.

---

## 5) Added helper script

## `04_vectorize/mobilevit_xxs/x_build_local_real_manifest.py`
Builds a CSV manifest from currently available local real-camera datasets.

### Output schema
- `image_path`
- `class_id`
- `class_name`
- `label_type` (`positive` or `negative`)
- `source`
- `split`
- `quality`

### Example usage
```bash
python3 04_vectorize/mobilevit_xxs/x_build_local_real_manifest.py \
  --datasets-root /Users/hanclaw/claw/data/ccg_card_id/datasets \
  --out /Users/hanclaw/claw/projects/ccg_card_id/cache/phase2/local_real_manifest.csv
```

---

## 6) Concrete recommendation summary

1. **Use current local real-camera data immediately** (strong enough for phase2 bootstrap).
2. **Do not run broad TCGplayer Listos scraping now** without explicit permission.
3. **Use Scryfall + synthetic augmentation for coverage**, and real local data for domain grounding.
4. **Stand up opt-in capture workflow this week** to unlock legally safe real-world scaling.
5. **Drive model improvements via hard-negative mining**, not just raw volume.
