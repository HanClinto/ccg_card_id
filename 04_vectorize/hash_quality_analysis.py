#!/usr/bin/env python3
"""
hash_quality_analysis.py  —  Nearest-Neighbor (NN) Purity test
================================================================
Measures hash quality by asking: "For each gallery card, is its nearest
neighbor in hash-space from the same artwork (illustration_id)?"

This is formally called *Nearest-Neighbor Purity* (NN Precision@1) in
metric learning.  The informal 'birthday analysis' framing comes from
cryptography: a birthday attack finds collisions in ~√N probes to a random
function.  Here we do the perceptual equivalent — systematically finding
'structural collisions': pairs of different-artwork cards that hash so close
together that the hash would confuse one for the other.

Three pHash variants (8×8 / 16×16 / 32×32, Hamming distance) and the best
neural embedding (ArcFace e15, cosine similarity) are evaluated from their
existing vector caches — no re-embedding needed.

Outputs:
  • NN purity per variant (% of cards whose nearest neighbour shares the artwork)
  • Distance distributions: intra-class (same art) vs inter-class (diff art)
  • Top-N confounders: different-artwork pairs closest in hash space

Usage:
    python 04_vectorize/hash_quality_analysis.py
    python 04_vectorize/hash_quality_analysis.py --top-confounders 50
    python 04_vectorize/hash_quality_analysis.py --output report.md
    python 04_vectorize/hash_quality_analysis.py --subsample 10000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ccg_card_id.config import cfg  # noqa: E402

DATA_DIR = cfg.data_dir
FAST_DATA_DIR = Path("/Users/hanclaw/claw/fast_data/ccg_card_id")

MANIFEST_PATH = DATA_DIR / "mobilevit_xxs" / "artwork_id_manifest.csv"
SCRYFALL_DB   = FAST_DATA_DIR / "catalog" / "scryfall" / "cards.db"

PHASH_CACHE_DIR = DATA_DIR / "vectors" / "phash" / "gallery_manifest_artwork_id_manifest"
NN_CACHE_DIR    = DATA_DIR / "vectors" / "nn_analysis_cache"
PHASH_VARIANTS = {
    "pHash 8×8 (64-bit)":    ("phash_8x8_64bit_gallery.npz",    8),
    "pHash 16×16 (256-bit)": ("phash_16x16_256bit_gallery.npz", 16),
    "pHash 32×32 (1024-bit)":("phash_32x32_1024bit_gallery.npz",32),
}

NEURAL_CACHE = (
    DATA_DIR / "vectors" / "mobilevit_xxs" / "img448"
    / "gallery_manifest_manifest"
    / "mobilevit_xxs_ft_illustration_id+set_code_e15_128d_gallery.npz"
)

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)

_POPCOUNT_U8 = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)

# ---------------------------------------------------------------------------
# NN result cache
# ---------------------------------------------------------------------------

def _file_fingerprint(*paths: Path) -> str:
    """Stable fingerprint from the mtime + size of one or more files."""
    h = hashlib.sha1()
    for p in paths:
        try:
            st = p.stat()
            h.update(str(p).encode())
            h.update(str(st.st_mtime).encode())
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            h.update(b"missing")
    return h.hexdigest()


def save_nn_cache(
    cache_path: Path,
    fingerprint: str,
    nn_any_dist:  np.ndarray,
    nn_any_idx:   np.ndarray,
    nn_diff_dist: np.ndarray,
    nn_diff_idx:  np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fingerprint=np.array([fingerprint]),
        nn_any_dist=nn_any_dist,
        nn_any_idx=nn_any_idx,
        nn_diff_dist=nn_diff_dist,
        nn_diff_idx=nn_diff_idx,
    )
    print(f"  NN cache saved → {cache_path}")


def load_nn_cache(
    cache_path: Path,
    expected_fp: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return cached arrays if fingerprint matches, else None."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if str(data["fingerprint"][0]) != expected_fp:
            print(f"  NN cache stale (source changed) → {cache_path}")
            return None
        print(f"  NN cache hit → {cache_path}")
        return (
            data["nn_any_dist"],
            data["nn_any_idx"],
            data["nn_diff_dist"],
            data["nn_diff_idx"],
        )
    except Exception as e:
        print(f"  NN cache unreadable ({e}) → recomputing")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def scryfall_url(card_id: str, set_code: str = "", collector_number: str = "") -> str:
    """Return the most direct Scryfall page URL for a card."""
    if set_code and collector_number:
        return f"https://scryfall.com/card/{set_code}/{collector_number}"
    return f"https://scryfall.com/search?q=id%3A{card_id}"


def load_manifest() -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (card_ids, illustration_ids, card_names, set_codes) aligned to manifest rows."""
    card_ids, illust_ids, card_names, set_codes = [], [], [], []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            card_ids.append(row["card_id"].lower())
            illust_ids.append(row["illustration_id"].lower())
            card_names.append(row["card_name"])
            set_codes.append(row.get("set_code", "").lower())
    return card_ids, illust_ids, card_names, set_codes


def load_phash_cache(npz_path: Path) -> np.ndarray:
    """Load packed uint8 pHash vectors from an npz cache (legacy or modern format)."""
    data = np.load(npz_path, allow_pickle=False)
    return data["embeddings"].astype(np.uint8, copy=False)


def load_neural_cache(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load float32 embedding vectors and their source paths."""
    data = np.load(npz_path, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float32, copy=False)
    paths = data["paths"].tolist() if "paths" in data.files else []
    return embeddings, [str(p) for p in paths]


def build_cardid_to_illust_map_from_db() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Returns (illust_map, name_map, set_map, cnum_map) from Scryfall SQLite."""
    conn = sqlite3.connect(SCRYFALL_DB)
    rows = conn.execute(
        "SELECT id, illustration_id, name, set_code, collector_number FROM cards"
    ).fetchall()
    conn.close()
    illust_map = {r[0].lower(): (r[1] or "").lower() for r in rows}
    name_map   = {r[0].lower(): (r[2] or "")         for r in rows}
    set_map    = {r[0].lower(): (r[3] or "").lower()  for r in rows}
    cnum_map   = {r[0].lower(): (r[4] or "")          for r in rows}
    return illust_map, name_map, set_map, cnum_map


# ---------------------------------------------------------------------------
# Core analysis: NN purity
# ---------------------------------------------------------------------------

def nn_analysis_hamming(
    vectors: np.ndarray,
    illust_ids: list[str],
    chunk_size: int = 200,
    desc: str = "Hamming NN",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each card find (a) nearest any neighbour and (b) nearest
    different-illustration neighbour.

    Returns:
        nn_any_dist   (n,) int32   — Hamming dist to nearest neighbour (excl self)
        nn_any_idx    (n,) int32   — index of that neighbour
        nn_diff_dist  (n,) int32   — Hamming dist to nearest *different-illust* neighbour
        nn_diff_idx   (n,) int32   — index of that neighbour
    """
    n = len(vectors)
    illust_arr = np.array(illust_ids)
    HUGE = 999999

    nn_any_dist  = np.full(n, HUGE, dtype=np.int32)
    nn_any_idx   = np.full(n, -1,   dtype=np.int32)
    nn_diff_dist = np.full(n, HUGE, dtype=np.int32)
    nn_diff_idx  = np.full(n, -1,   dtype=np.int32)

    for i_start in tqdm(range(0, n, chunk_size), desc=desc, unit="chunk"):
        i_end = min(i_start + chunk_size, n)
        q = vectors[i_start:i_end]            # (c, bytes)

        # XOR and popcount: (c, n)
        # Broadcast: q[:, None, :] ^ vectors[None, :, :] = (c, n, bytes)
        xor  = q[:, np.newaxis, :] ^ vectors[np.newaxis, :, :]
        dists = _POPCOUNT_U8[xor].sum(axis=2, dtype=np.int32)   # (c, n)

        # Exclude self
        for k in range(i_end - i_start):
            dists[k, i_start + k] = HUGE

        # Nearest any
        j_any = dists.argmin(axis=1)
        nn_any_idx[i_start:i_end]  = j_any
        nn_any_dist[i_start:i_end] = dists[np.arange(i_end - i_start), j_any]

        # Nearest different-illustration
        for k in range(i_end - i_start):
            i_global = i_start + k
            same_mask = illust_arr == illust_arr[i_global]
            dists[k, same_mask] = HUGE
            # Also keep self excluded
            dists[k, i_global]  = HUGE
        j_diff = dists.argmin(axis=1)
        nn_diff_idx[i_start:i_end]  = j_diff
        nn_diff_dist[i_start:i_end] = dists[np.arange(i_end - i_start), j_diff]

    return nn_any_dist, nn_any_idx, nn_diff_dist, nn_diff_idx


def nn_analysis_cosine(
    vectors: np.ndarray,
    illust_ids: list[str],
    chunk_size: int = 500,
    desc: str = "Cosine NN",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same contract as nn_analysis_hamming but for L2-normalised float vectors."""
    n = len(vectors)
    illust_arr = np.array(illust_ids)
    NEG_HUGE = -9.0

    nn_any_sim  = np.full(n, NEG_HUGE, dtype=np.float32)
    nn_any_idx  = np.full(n, -1,       dtype=np.int32)
    nn_diff_sim = np.full(n, NEG_HUGE, dtype=np.float32)
    nn_diff_idx = np.full(n, -1,       dtype=np.int32)

    # Ensure L2-normalised (model output should already be, but just in case)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vecs_n = (vectors / norms).astype(np.float32)

    for i_start in tqdm(range(0, n, chunk_size), desc=desc, unit="chunk"):
        i_end = min(i_start + chunk_size, n)
        q = vecs_n[i_start:i_end]        # (c, d)
        sims = q @ vecs_n.T              # (c, n)

        # Exclude self
        for k in range(i_end - i_start):
            sims[k, i_start + k] = NEG_HUGE

        # Nearest any
        j_any = sims.argmax(axis=1)
        nn_any_idx[i_start:i_end]  = j_any
        nn_any_sim[i_start:i_end]  = sims[np.arange(i_end - i_start), j_any]

        # Nearest different-illustration
        for k in range(i_end - i_start):
            i_global = i_start + k
            same_mask = illust_arr == illust_arr[i_global]
            sims[k, same_mask]  = NEG_HUGE
            sims[k, i_global]   = NEG_HUGE
        j_diff = sims.argmax(axis=1)
        nn_diff_idx[i_start:i_end]  = j_diff
        nn_diff_sim[i_start:i_end]  = sims[np.arange(i_end - i_start), j_diff]

    return nn_any_sim, nn_any_idx, nn_diff_sim, nn_diff_idx


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def distance_stats(arr: np.ndarray, label: str) -> str:
    """Return a one-line percentile summary."""
    ps = np.percentile(arr, [0, 10, 25, 50, 75, 90, 100]).astype(int if arr.dtype != np.float32 else float)
    fmt = ".3f" if arr.dtype == np.float32 else "d"
    vals = [f"{v:{fmt}}" for v in ps]
    return f"  {label:30s}  min={vals[0]}  p10={vals[1]}  p25={vals[2]}  p50={vals[3]}  p75={vals[4]}  p90={vals[5]}  max={vals[6]}"


def make_confounders_table(
    nn_diff_dist: np.ndarray,
    nn_diff_idx:  np.ndarray,
    card_ids:     list[str],
    illust_ids:   list[str],
    card_names:   list[str],
    set_codes:    list[str],
    top_n:        int,
    mode:         str,  # "hamming" or "cosine"
    cnum_map:     dict[str, str] | None = None,
) -> list[dict]:
    """
    Return top_n worst confounders: cards whose nearest different-artwork
    neighbour is closest in hash space.
    """
    # For hamming, smallest distance = worst confounder
    # For cosine, largest similarity = worst confounder
    if mode == "hamming":
        order = np.argsort(nn_diff_dist)[:top_n * 2]  # oversample to dedupe pairs
    else:
        order = np.argsort(-nn_diff_dist)[:top_n * 2]

    rows = []
    seen_pairs: set[frozenset] = set()
    for i in order:
        if len(rows) >= top_n:
            break
        j = int(nn_diff_idx[i])
        if j < 0:
            continue
        pair = frozenset([i, j])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        cnum_a = (cnum_map or {}).get(card_ids[i], "")
        cnum_b = (cnum_map or {}).get(card_ids[j], "")
        rows.append({
            "dist":      float(nn_diff_dist[i]),
            "card_a":    card_names[i],
            "card_id_a": card_ids[i],
            "set_a":     set_codes[i],
            "url_a":     scryfall_url(card_ids[i], set_codes[i], cnum_a),
            "illust_a":  illust_ids[i][:8],
            "card_b":    card_names[j],
            "card_id_b": card_ids[j],
            "set_b":     set_codes[j],
            "url_b":     scryfall_url(card_ids[j], set_codes[j], cnum_b),
            "illust_b":  illust_ids[j][:8],
        })
    return rows


def print_confounders_table(rows: list[dict], mode: str, lines: list[str]) -> None:
    """Emit a GitHub-Flavored Markdown table with clickable Scryfall links."""
    score_hdr = "Hamming dist" if mode == "hamming" else "Cosine sim"
    lines.append(
        f"| {score_hdr} | Card A (set) | Card B (set) | Illust A | Illust B |"
    )
    lines.append("| ---: | --- | --- | --- | --- |")
    for r in rows:
        score_str = f"{int(r['dist'])}" if mode == "hamming" else f"{r['dist']:.4f}"
        name_a = r["card_a"].replace("|", "\\|")
        name_b = r["card_b"].replace("|", "\\|")
        set_a  = f" ({r['set_a']})" if r["set_a"] else ""
        set_b  = f" ({r['set_b']})" if r["set_b"] else ""
        link_a = f"[{name_a}{set_a}]({r['url_a']})"
        link_b = f"[{name_b}{set_b}]({r['url_b']})"
        lines.append(
            f"| {score_str} | {link_a} | {link_b} | {r['illust_a']} | {r['illust_b']} |"
        )


def analyze_phash_variant(
    label: str,
    npz_path: Path,
    hash_size: int,
    card_ids:   list[str],
    illust_ids: list[str],
    card_names: list[str],
    set_codes:  list[str],
    top_n: int,
    subsample: int | None,
    lines: list[str],
    nn_cache_path: Path | None = None,
    rebuild_nn_cache: bool = False,
) -> None:
    lines.append(f"\n{'='*70}")
    lines.append(f"  {label}")
    lines.append(f"{'='*70}")

    vectors = load_phash_cache(npz_path)
    n_total = len(vectors)
    max_dist = hash_size * hash_size

    if len(vectors) != len(card_ids):
        lines.append(
            f"  WARNING: cache has {len(vectors)} rows but manifest has {len(card_ids)} rows. "
            "Alignment may be wrong — results unreliable."
        )
        n = min(len(vectors), len(card_ids))
        vectors   = vectors[:n]
        c_ids     = card_ids[:n]
        i_ids     = illust_ids[:n]
        c_names   = card_names[:n]
        s_codes   = set_codes[:n]
    else:
        c_ids, i_ids, c_names, s_codes = card_ids, illust_ids, card_names, set_codes
        n = n_total

    # Optional subsample
    if subsample and subsample < n:
        idx = np.random.choice(n, subsample, replace=False)
        idx.sort()
        vectors = vectors[idx]
        c_ids   = [c_ids[i]   for i in idx]
        i_ids   = [i_ids[i]   for i in idx]
        c_names = [c_names[i] for i in idx]
        s_codes = [s_codes[i] for i in idx]
        lines.append(f"  (subsampled to {subsample} cards)")
        n = subsample

    n_illust = len(set(i_ids))
    lines.append(f"  Gallery: {n:,} cards | {n_illust:,} unique illustration_ids")
    lines.append(f"  Hash size: {hash_size}×{hash_size} = {max_dist} bits")

    # Try NN cache (only when not subsampling — subsampled runs are exploratory)
    cached = None
    nn_fp = _file_fingerprint(npz_path, MANIFEST_PATH)
    if nn_cache_path is not None and subsample is None:
        if rebuild_nn_cache:
            print(f"  Rebuilding NN cache for {label}")
        else:
            cached = load_nn_cache(nn_cache_path, nn_fp)

    if cached is not None:
        nn_any_dist, nn_any_idx, nn_diff_dist, nn_diff_idx = cached
    else:
        nn_any_dist, nn_any_idx, nn_diff_dist, nn_diff_idx = nn_analysis_hamming(
            vectors, i_ids, desc=f"  {label[:25]}"
        )
        if nn_cache_path is not None and subsample is None:
            save_nn_cache(nn_cache_path, nn_fp, nn_any_dist, nn_any_idx, nn_diff_dist, nn_diff_idx)

    # NN purity
    nn_same = np.array([i_ids[nn_any_idx[i]] == i_ids[i] for i in range(n)])
    purity = nn_same.mean() * 100
    lines.append(f"\n  Nearest-Neighbour Purity:  {purity:.2f}%  ({nn_same.sum():,} / {n:,})")

    # Zero-distance collisions between different artworks
    zero_diff = int((nn_diff_dist == 0).sum())
    lines.append(f"  Identical-hash collisions (dist=0, diff artwork):  {zero_diff:,}")

    # Distribution stats
    # Intra-class: for each card, dist to nearest same-illustration card
    # We derive this from the NN search: when NN is same-illust, nn_any_dist IS the intra dist
    intra_dists = nn_any_dist[nn_same]
    # For cards whose NN is different-illust, the intra dist is at least nn_any_dist
    # (we don't have the exact same-illust NN for those — would need a second pass)
    # So we report intra dist for "well-behaved" cards only, and note the caveat
    lines.append(f"\n  Distance distributions (bits, max={max_dist}):")
    if len(intra_dists):
        lines.append(distance_stats(intra_dists,   "Intra-class (same art, NN purity subset)"))
    lines.append(distance_stats(nn_diff_dist,      "Inter-class (diff art, nearest confounder)"))

    # Separation: what fraction of inter-class NNs are within the median intra distance?
    if len(intra_dists):
        med_intra = float(np.median(intra_dists))
        pct_overlap = float((nn_diff_dist <= med_intra).mean()) * 100
        lines.append(f"  Overlap: {pct_overlap:.1f}% of diff-art NNs ≤ median same-art dist ({med_intra:.0f} bits)")

    # Confounders
    lines.append(f"\n#### Top-{top_n} confounders (different artwork, closest in hash space)\n")
    confounders = make_confounders_table(
        nn_diff_dist, nn_diff_idx, c_ids, i_ids, c_names, s_codes, top_n, mode="hamming"
    )
    print_confounders_table(confounders, mode="hamming", lines=lines)


def analyze_neural_variant(
    label: str,
    npz_path: Path,
    illust_map:  dict[str, str],
    name_map:    dict[str, str],
    set_map:     dict[str, str],
    cnum_map:    dict[str, str],
    top_n: int,
    subsample: int | None,
    lines: list[str],
    nn_cache_path: Path | None = None,
    rebuild_nn_cache: bool = False,
) -> None:
    lines.append(f"\n{'='*70}")
    lines.append(f"  {label}")
    lines.append(f"{'='*70}")

    vectors, paths = load_neural_cache(npz_path)
    n_total = len(vectors)

    # Join paths → card_id → illustration_id / set_code / etc.
    card_ids   = []
    illust_ids = []
    card_names = []
    set_codes  = []
    valid_mask = []
    for p in paths:
        m = UUID_RE.search(p)
        if m:
            cid = m.group(0).lower()
            card_ids.append(cid)
            illust_ids.append(illust_map.get(cid, ""))
            card_names.append(name_map.get(cid, cid[:8]))
            set_codes.append(set_map.get(cid, ""))
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_idx = [i for i, v in enumerate(valid_mask) if v]
    vectors = vectors[valid_idx]
    n = len(vectors)
    missing_illust = sum(1 for iid in illust_ids if not iid)

    n_illust = len(set(illust_ids))
    lines.append(f"  Gallery: {n:,} cards | {n_illust:,} unique illustration_ids")
    if missing_illust:
        lines.append(f"  WARNING: {missing_illust:,} cards missing illustration_id (not in catalog)")

    if subsample and subsample < n:
        idx = np.random.choice(n, subsample, replace=False)
        idx.sort()
        vectors    = vectors[idx]
        card_ids   = [card_ids[i]   for i in idx]
        illust_ids = [illust_ids[i] for i in idx]
        card_names = [card_names[i] for i in idx]
        set_codes  = [set_codes[i]  for i in idx]
        lines.append(f"  (subsampled to {subsample} cards)")
        n = subsample

    # Try NN cache (only when not subsampling)
    cached = None
    nn_fp = _file_fingerprint(npz_path)
    if nn_cache_path is not None and subsample is None:
        if rebuild_nn_cache:
            print(f"  Rebuilding NN cache for {label}")
        else:
            cached = load_nn_cache(nn_cache_path, nn_fp)

    if cached is not None:
        nn_any_sim, nn_any_idx, nn_diff_sim, nn_diff_idx = cached
    else:
        nn_any_sim, nn_any_idx, nn_diff_sim, nn_diff_idx = nn_analysis_cosine(
            vectors, illust_ids, desc=f"  {label[:25]}"
        )
        if nn_cache_path is not None and subsample is None:
            save_nn_cache(nn_cache_path, nn_fp, nn_any_sim, nn_any_idx, nn_diff_sim, nn_diff_idx)

    # NN purity
    nn_same = np.array([illust_ids[nn_any_idx[i]] == illust_ids[i] for i in range(n)])
    purity = nn_same.mean() * 100
    lines.append(f"\n  Nearest-Neighbour Purity:  {purity:.2f}%  ({nn_same.sum():,} / {n:,})")

    # Identical-direction collisions (cosine sim >= 0.9999, different artwork)
    near_perfect = int((nn_diff_sim >= 0.9999).sum())
    lines.append(f"  Near-identical embeddings (cos≥0.9999, diff artwork):  {near_perfect:,}")

    # Distributions (using sim, higher = more similar)
    intra_sims = nn_any_sim[nn_same]
    lines.append(f"\n  Cosine similarity distributions (higher = more similar):")
    if len(intra_sims):
        lines.append(distance_stats(intra_sims,   "Intra-class (same art, NN purity subset)"))
    lines.append(distance_stats(nn_diff_sim,       "Inter-class (diff art, nearest confounder)"))

    if len(intra_sims):
        med_intra = float(np.median(intra_sims))
        pct_overlap = float((nn_diff_sim >= med_intra).mean()) * 100
        lines.append(f"  Overlap: {pct_overlap:.1f}% of diff-art NNs ≥ median same-art sim ({med_intra:.4f})")

    lines.append(f"\n#### Top-{top_n} confounders (different artwork, most similar in embedding space)\n")
    confounders = make_confounders_table(
        nn_diff_sim, nn_diff_idx, card_ids, illust_ids, card_names, set_codes, top_n,
        mode="cosine", cnum_map=cnum_map,
    )
    print_confounders_table(confounders, mode="cosine", lines=lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hash quality / NN purity analysis")
    parser.add_argument("--top-confounders", type=int, default=20,
                        help="How many worst confounders to show per variant (default: 20)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write markdown report to this file")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Analyse a random subsample of N cards per variant (for speed)")
    parser.add_argument("--skip-neural", action="store_true",
                        help="Skip the neural embedding variant (faster)")
    parser.add_argument("--rebuild-nn-cache", action="store_true",
                        help="Ignore existing NN result cache and recompute from scratch")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for subsampling (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    lines: list[str] = []
    lines.append("# Hash Quality Analysis — Nearest-Neighbour Purity")
    lines.append("")
    lines.append(
        "For each hash variant we find every gallery card's nearest neighbour "
        "in hash-space and check whether they share the same artwork (illustration_id). "
        "A perfect hash = 100% purity. 'Confounders' are different-artwork pairs "
        "that hash suspiciously close together."
    )
    lines.append("")

    # Load manifest labels for pHash variants
    print("Loading manifest...")
    card_ids_mf, illust_ids_mf, card_names_mf, set_codes_mf = load_manifest()
    print(f"  {len(card_ids_mf):,} rows, {len(set(illust_ids_mf)):,} unique illustration_ids")

    # pHash variants
    phash_cache_names = {
        "pHash 8×8 (64-bit)":    "phash_8x8.npz",
        "pHash 16×16 (256-bit)": "phash_16x16.npz",
        "pHash 32×32 (1024-bit)":"phash_32x32.npz",
    }
    for label, (fname, hash_size) in PHASH_VARIANTS.items():
        npz_path = PHASH_CACHE_DIR / fname
        if not npz_path.exists():
            lines.append(f"\n[SKIP] {label} — cache not found: {npz_path}")
            continue
        analyze_phash_variant(
            label, npz_path, hash_size,
            card_ids_mf, illust_ids_mf, card_names_mf, set_codes_mf,
            top_n=args.top_confounders,
            subsample=args.subsample,
            lines=lines,
            nn_cache_path=NN_CACHE_DIR / phash_cache_names[label],
            rebuild_nn_cache=args.rebuild_nn_cache,
        )

    # Neural variant
    if not args.skip_neural:
        if not NEURAL_CACHE.exists():
            lines.append(f"\n[SKIP] Neural e15 — cache not found: {NEURAL_CACHE}")
        else:
            print("Loading Scryfall catalog for illustration_id lookup...")
            illust_map, name_map, set_map, cnum_map = build_cardid_to_illust_map_from_db()
            print(f"  {len(illust_map):,} cards in catalog")
            analyze_neural_variant(
                "Neural ArcFace e15 (128-d cosine, 108k gallery)",
                NEURAL_CACHE,
                illust_map, name_map, set_map, cnum_map,
                top_n=args.top_confounders,
                subsample=args.subsample,
                lines=lines,
                nn_cache_path=NN_CACHE_DIR / "neural_e15.npz",
                rebuild_nn_cache=args.rebuild_nn_cache,
            )

    report = "\n".join(lines)
    print()
    print(report)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
