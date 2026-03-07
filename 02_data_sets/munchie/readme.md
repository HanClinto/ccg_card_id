# Munchie Dataset

Real-world scanner images of Magic: The Gathering cards, sourced from Munchie / Access Granted Online.

## Contents

- **812 front-face scans** of individual MTG cards, stored in `munchie_fronts.zip`
- **metadata** in `data/master.json` — each record has `ocrName`, `set`, and a `sfcards` list of candidate Scryfall IDs

## Purpose

These scans bridge the domain gap between clean Scryfall reference images (used for gallery building)
and real-world card photos (used at query time). Training on munchie scans teaches the model that
a slightly worn, lit-unevenly, flatbed-scanned card should embed near its clean Scryfall reference.

## Training phases

The resolved manifest carries several ID columns, one per training phase:

| Phase | Label column | Goal |
|---|---|---|
| 1 — Artwork ID | `illustration_id` | All printings sharing the same art cluster together |
| 2 — Edition ID | `card_id` (scryfall UUID) | Distinguish between different printings of the same card |
| 3 — Language ID | `lang` | (future) Distinguish language variants |

## Pipeline

```
01_extract.py          Extract munchie_fronts.zip → datasets/munchie/images/fronts/
02_resolve.py          Resolve ocrName+set → scryfall card_id, illustration_id, oracle_id
03_build_manifest.py   Write datasets/munchie/manifest.csv (training-ready)
```

Run in order from `02_data_sets/munchie/code/`.

## Coverage (after resolution)

- ~528/812 fully resolved (unique English name+set match)
- ~75/812 ambiguous-but-same-artwork (multiple set variants, identical illustration_id → usable)
- ~209/812 not resolved (tokens, full-art numbered names, non-English)

## Known issues

See `data/munchie_notes.txt` for specific image quality notes (off-center scans, upside-down
images, Japanese cards, promos). The pipeline flags these in `resolved.jsonl`; they are excluded
from the manifest by default.

## License

These images are sourced from Munchie / Access Granted Online and are intended for internal
research and training use only.
