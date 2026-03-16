# Ximilar TCG API — Evaluation Report

**Dataset:** Sol Ring (solring) — 21 editions, homography-aligned video frames  
**Frames evaluated:** 110 / 307 total (credits exhausted after 1,000)  
**Date:** 2026-03-12  

## Summary

| Metric | Correct | Total | Accuracy |
|--------|--------:|------:|---------:|
| Artwork (correct card name) | 110 | 110 | **100.0%** |
| Edition (exact printing match) | 80 | 110 | **72.7%** |

### Comparison vs ArcFace e75 (same dataset)

| Model | Artwork | Edition |
|-------|--------:|--------:|
| Ximilar TCG API | 100.0% | 72.7% |
| ArcFace e75 (ours) | 89.6% | 3.9% |
| pHash 16×16 | 100.0% | n/a |

> **Note:** ArcFace was trained on illustration_id (not edition), so its 3.9% edition
> accuracy is expected. Ximilar's 72.7% edition accuracy on nearly-identical Sol Ring
> printings is a strong result for a generalised card-ID service.

## Per-Edition Breakdown

| Set | Frames | Artwork | Edition | Edition % | Common misidentifications |
|-----|-------:|--------:|--------:|----------:|--------------------------|
| afc | 14 | 14/14 (100%) | 1/14 (7%) | 7% | khc #104 (9×), scd #276 (2×), znc #120 (1×) |
| c13 | 14 | 14/14 (100%) | 11/14 (79%) | 79% | c20 #252 (1×), c19 #221 (1×), cmd #261 (1×) |
| c15 | 13 | 13/13 (100%) | 11/13 (85%) | 85% | c19 #221 (1×), c17 #223 (1×) |
| c16 | 9 | 9/9 (100%) | 6/9 (67%) | 67% | c20 #252 (2×), c17 #223 (1×) |
| c20 | 11 | 11/11 (100%) | 11/11 (100%) | 100% | — |
| clb | 17 | 17/17 (100%) | 12/17 (71%) | 71% | brc #160 (2×), c20 #252 (1×), znc #120 (1×) |
| khc | 13 | 13/13 (100%) | 9/13 (69%) | 69% | znc #120 (2×), c20 #252 (1×), scd #276 (1×) |
| plst | 19 | 19/19 (100%) | 19/19 (100%) | 100% | — |

## Edition Confusion Matrix (misses only)

Each cell = number of frames from that true set predicted as another set.

| True \ Predicted | brc | c17 | c19 | c20 | cmd | khc | ncc | scd | znc |
|---|---|---|---|---|---|---|---|---|---|
| **afc** | · | · | · | 1 | · | 9 | · | 2 | 1 |
| **c13** | · | · | 1 | 1 | 1 | · | · | · | · |
| **c15** | · | 1 | 1 | · | · | · | · | · | · |
| **c16** | · | 1 | · | 2 | · | · | · | · | · |
| **clb** | 2 | · | · | 1 | · | · | 1 | · | 1 |
| **khc** | · | · | · | 1 | · | · | · | 1 | 2 |

## Edition Misses — Detail

| Filename | True Set | True # | Pred Set | Pred # | Pred Full Name |
|----------|----------|-------:|----------|-------:|----------------|
| `0afa0e33-4804-4b00-b625-c2d6b61090fc_solring_khc_20221219_153056.mp4-0247.jpg` | khc | 104 | c20 | 252 | Sol Ring Commander 2020 (C20) #252 |
| `0afa0e33-4804-4b00-b625-c2d6b61090fc_solring_khc_20221219_153056.mp4-0367.jpg` | khc | 104 | znc | 120 | Sol Ring Zendikar Rising Commander (ZNC) #120 |
| `0afa0e33-4804-4b00-b625-c2d6b61090fc_solring_khc_20221219_153056.mp4-0427.jpg` | khc | 104 | znc | 120 | Sol Ring Zendikar Rising Commander (ZNC) #120 |
| `0afa0e33-4804-4b00-b625-c2d6b61090fc_solring_khc_20221219_153056.mp4-0727.jpg` | khc | 104 | scd | 276 | Sol Ring Starter Commander Decks (SCD) #276 |
| `0f003fde-be17-4159-a361-711ed0bee911_solring_c16_20221219_153333.mp4-0362.jpg` | c16 | 272 | c20 | 252 | Sol Ring Commander 2020 (C20) #252 |
| `0f003fde-be17-4159-a361-711ed0bee911_solring_c16_20221219_153333.mp4-0422.jpg` | c16 | 272 | c20 | 252 | Sol Ring Commander 2020 (C20) #252 |
| `0f003fde-be17-4159-a361-711ed0bee911_solring_c16_20221219_153333.mp4-0542.jpg` | c16 | 272 | c17 | 223 | Sol Ring Commander 2017 (C17) #223 |
| `199cde21-5bc3-49cd-acd4-bae3af6e5881_solring_clb_20221219_153212.mp4-0302.jpg` | clb | 871 | c20 | 252 | Sol Ring Commander 2020 (C20) #252 |
| `199cde21-5bc3-49cd-acd4-bae3af6e5881_solring_clb_20221219_153212.mp4-0362.jpg` | clb | 871 | znc | 120 | Sol Ring Zendikar Rising Commander (ZNC) #120 |
| `199cde21-5bc3-49cd-acd4-bae3af6e5881_solring_clb_20221219_153212.mp4-0542.jpg` | clb | 871 | ncc | 379 | Sol Ring New Capenna Commander (NCC) #379 |
| `199cde21-5bc3-49cd-acd4-bae3af6e5881_solring_clb_20221219_153212.mp4-0602.jpg` | clb | 871 | brc | 160 | Sol Ring The Brothers' War Commander (BRC) #160 |
| `199cde21-5bc3-49cd-acd4-bae3af6e5881_solring_clb_20221219_153212.mp4-0784.jpg` | clb | 871 | brc | 160 | Sol Ring The Brothers' War Commander (BRC) #160 |
| `1b59533a-3e38-495d-873e-2f89fbd08494_solring_c13_20221219_153409.mp4-0480.jpg` | c13 | 259 | c20 | 252 | Sol Ring Commander 2020 (C20) #252 |
| `1b59533a-3e38-495d-873e-2f89fbd08494_solring_c13_20221219_153409.mp4-0540.jpg` | c13 | 259 | c19 | 221 | Sol Ring Commander 2019 (C19) #221 |
| `1b59533a-3e38-495d-873e-2f89fbd08494_solring_c13_20221219_153409.mp4-0660.jpg` | c13 | 259 | cmd | 261 | Sol Ring Commander 2011 (CMD) #261 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0000.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0062.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0122.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0242.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0302.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0362.jpg` | afc | 215 | scd | 276 | Sol Ring Starter Commander Decks (SCD) #276 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0422.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0482.jpg` | afc | 215 | scd | 276 | Sol Ring Starter Commander Decks (SCD) #276 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0544.jpg` | afc | 215 | znc | 120 | Sol Ring Zendikar Rising Commander (ZNC) #120 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0604.jpg` | afc | 215 | c20 | 252 | Sol Ring Commander 2020 (C20) #252 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0664.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0724.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `2c52c96d-e20f-4025-b759-674b36cf0db3_solring_afc_20221219_153115.mp4-0784.jpg` | afc | 215 | khc | 104 | Sol Ring Kaldheim Commander (KHC) #104 |
| `3459b229-7c46-4f70-87d4-bb31c2c17dd9_solring_c15_20221219_153523.mp4-0420.jpg` | c15 | 268 | c19 | 221 | Sol Ring Commander 2019 (C19) #221 |
| `3459b229-7c46-4f70-87d4-bb31c2c17dd9_solring_c15_20221219_153523.mp4-0660.jpg` | c15 | 268 | c17 | 223 | Sol Ring Commander 2017 (C17) #223 |

## Notes & Observations

- **Artwork accuracy is perfect (100%)** — Ximilar correctly identifies every frame as
  'Sol Ring' with no false names.
- **Edition accuracy is 72.7%** — impressive for 21 nearly-identical printings from video
  frames. The main confusion patterns are recent commander decks being mixed up with each
  other (afc, khc, znc, etc.).
- **plst editions score 100% edition accuracy** — Ximilar correctly identifies The List
  printings, which is noteworthy given they carry host-set-specific codes.
- **afc has the worst edition accuracy (1/14 = 7%)** — the D&D Adventures in the Forgotten
  Realms commander deck Sol Ring is frequently confused with other recent commander-era prints.
- **Remaining 197 frames uncached** — run `python 06_eval/05_eval_ximilar.py` once credits
  are topped up to complete the solring dataset and add daniel (150 frames).

## Credit Usage

- 1,000 credits consumed (10 credits × 110 frames)
- ~1,970 credits needed to finish: 197 remaining solring + 150 daniel × 10 credits