# pHash Baseline Accuracy Report (cached)

This report is generated from existing cached eval artifacts (no rerun).

## Source of truth
- Eval summary CSV:
  - `/Volumes/carbonite/claw/data/ccg_card_id/results/eval/20260312_030642/summary.csv`
- Filter used:
  - `algorithm_variant` in `{phash_8x8_64bit, phash_16x16_256bit, phash_32x32_1024bit}`
  - `topk = 1`

---

## Top-1 Artwork (Name/Artwork ID)

| Dataset | Best pHash variant | Top-1 |
|---|---|---:|
| solring | phash_16x16_256bit (tie with 32x32) | **1.000** (307/307) |
| clint_backgrounds | phash_32x32_1024bit | **0.749** (951/1270) |
| daniel | phash_16x16_256bit | **0.993** (149/150) |
| munchie | phash_8/16/32 (all tie) | **1.000** (564/564) |

## Top-1 Edition (exact printing)

| Dataset | Best pHash variant | Top-1 |
|---|---|---:|
| solring | phash_32x32_1024bit | **0.440** (135/307) |
| clint_backgrounds | phash_32x32_1024bit | **0.640** (813/1270) |
| daniel | phash_32x32_1024bit | **0.247** (37/150) |
| munchie | phash_16x16_256bit (tie with 32x32) | **1.000** (564/564) |

---

## Sol Ring focus (the key stress benchmark)

### Artwork Top-1
- phash_8x8_64bit: 0.912 (280/307)
- phash_16x16_256bit: **1.000 (307/307)**
- phash_32x32_1024bit: **1.000 (307/307)**

### Edition Top-1
- phash_8x8_64bit: 0.078 (24/307)
- phash_16x16_256bit: 0.179 (55/307)
- phash_32x32_1024bit: **0.440 (135/307)**

Interpretation:
- pHash is effectively solved for **name/artwork** on `solring` at 16x16 and 32x32.
- pHash is far from solved for **edition** on `solring`.
- Therefore: beating **0.440 Top-1 edition on solring** (32x32 pHash) is a clear neural milestone.

---

## Practical baseline target for neural bake-off

At minimum, a contender should aim to:
1. Stay competitive with pHash on Name/Artwork Top-1
2. Exceed **0.440 Top-1 edition on `solring`**

That is a concrete “cracked it” threshold relative to cached pHash behavior.
