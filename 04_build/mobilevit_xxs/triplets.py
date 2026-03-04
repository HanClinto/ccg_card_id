from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ccg_card_id.config import cfg


@dataclass(frozen=True)
class CardRec:
    card_id: str
    oracle_id: str
    name: str
    set_code: str
    lang: str
    artwork_id: str
    image_path: str


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_phash_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    out: dict[str, str] = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("card_id") or row.get("id") or "").strip()
            hv = str(row.get("phash") or row.get("hash") or "").strip()
            if cid and hv:
                out[cid] = hv
    return out


def _hamming_hex(a: str, b: str) -> int:
    try:
        ai = int(a, 16)
        bi = int(b, 16)
    except Exception:
        return 10**9
    return (ai ^ bi).bit_count()


def _build_image_index(images_root: Path) -> dict[str, list[Path]]:
    """One-time scan of image tree: card_id stem -> matching files."""
    idx: dict[str, list[Path]] = {}
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in tqdm(images_root.rglob("*"), desc=f"index images {images_root.name}", unit="file"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        idx.setdefault(stem, []).append(p)
    return idx


def _find_image_from_index(image_index: dict[str, list[Path]], card_id: str) -> Path | None:
    cand = image_index.get(card_id)
    if not cand:
        return None
    for p in cand:
        if "/front/" in p.as_posix():
            return p
    return cand[0]


def _is_commander_legal(card: dict[str, Any]) -> bool:
    legalities = card.get("legalities") or {}
    return legalities.get("commander") == "legal"


def _is_world_championship_blank(card: dict[str, Any]) -> bool:
    n = str(card.get("name", "")).lower()
    layout = str(card.get("layout", "")).lower()
    return "world championship" in n or "blank" in n or layout == "token"


def load_card_records(
    default_cards_json: Path,
    images_root: Path,
    *,
    commander_only: bool = True,
    exclude_world_blank: bool = True,
    image_index: dict[str, list[Path]] | None = None,
) -> list[CardRec]:
    raw = _load_json(default_cards_json)
    out: list[CardRec] = []
    if image_index is None:
        image_index = _build_image_index(images_root)
    for c in tqdm(raw, desc=f"load cards {default_cards_json.name}", unit="card"):
        if commander_only and not _is_commander_legal(c):
            continue
        if exclude_world_blank and _is_world_championship_blank(c):
            continue
        card_id = str(c.get("id", "")).strip()
        if not card_id:
            continue
        img = _find_image_from_index(image_index, card_id)
        if img is None:
            continue
        out.append(
            CardRec(
                card_id=card_id,
                oracle_id=str(c.get("oracle_id", "")).strip(),
                name=str(c.get("name", "")).strip(),
                set_code=str(c.get("set", "")).strip(),
                lang=str(c.get("lang", "")).strip(),
                artwork_id=str(c.get("illustration_id", "") or "").strip(),
                image_path=str(img),
            )
        )
    return out


def _load_resume_hard_negs(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}
    except Exception:
        pass
    return {}


def _save_hard_negs(path: Path | None, hard_negs: dict[str, list[str]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hard_negs, indent=2), encoding="utf-8")


def build_triplets(
    *,
    card_id_cards_json: Path = cfg.scryfall_default_cards,
    all_cards_json: Path = cfg.scryfall_all_cards,
    images_root: Path = cfg.scryfall_images_dir,
    phash_json: Path | None = None,
    out_csv: Path,
    out_hard_negs_json: Path | None = None,
    seed: int = 42,
    hard_k: int = 24,
    resume: bool = True,
    checkpoint_every: int = 200,
    prefix_chars: int = 2,
    hard_candidate_cap: int = 2500,
) -> dict[str, int]:
    rng = random.Random(seed)
    image_index = _build_image_index(images_root)

    recs_card = load_card_records(card_id_cards_json, images_root, image_index=image_index)
    recs_all = load_card_records(all_cards_json, images_root, image_index=image_index)
    by_id_card = {r.card_id: r for r in recs_card}

    if phash_json is None:
        phash_json = cfg.vectors_file("phash", 8)  # 8x8 grid = 64-bit hash
    phash_map = _load_phash_map(phash_json)

    ids_with_hash = [r.card_id for r in recs_card if r.card_id in phash_map]

    # Bucket by hash prefix to avoid O(N^2) full scan.
    prefix_buckets: dict[str, list[str]] = {}
    for cid in ids_with_hash:
        pref = phash_map[cid][:prefix_chars]
        prefix_buckets.setdefault(pref, []).append(cid)

    hard_negs: dict[str, list[str]] = _load_resume_hard_negs(out_hard_negs_json) if resume else {}

    processed = 0
    pbar = tqdm(ids_with_hash, desc="build card_id hard negatives", unit="anchor")
    for cid in pbar:
        if cid in hard_negs and hard_negs[cid]:
            continue

        a = by_id_card[cid]
        hv_a = phash_map[cid]

        pref = hv_a[:prefix_chars]
        pool = [x for x in prefix_buckets.get(pref, []) if x != cid]

        # If bucket too small, widen with random global samples.
        if len(pool) < max(hard_k * 4, 128):
            need = max(hard_k * 8, 256) - len(pool)
            if need > 0:
                extra_src = [x for x in ids_with_hash if x != cid and x not in set(pool)]
                if extra_src:
                    if len(extra_src) > need:
                        extra = rng.sample(extra_src, need)
                    else:
                        extra = extra_src
                    pool.extend(extra)

        if hard_candidate_cap > 0 and len(pool) > hard_candidate_cap:
            pool = rng.sample(pool, hard_candidate_cap)

        cands: list[tuple[int, str]] = []
        for other in pool:
            b = by_id_card[other]
            if b.oracle_id and a.oracle_id and b.oracle_id == a.oracle_id:
                continue
            d = _hamming_hex(hv_a, phash_map[other])
            cands.append((d, other))

        cands.sort(key=lambda x: x[0])
        hard_negs[cid] = [x[1] for x in cands[:hard_k]]

        processed += 1
        if out_hard_negs_json is not None and processed % max(1, checkpoint_every) == 0:
            _save_hard_negs(out_hard_negs_json, hard_negs)
            pbar.set_postfix({"saved": processed, "done": len(hard_negs)})

    if out_hard_negs_json is not None:
        _save_hard_negs(out_hard_negs_json, hard_negs)

    # -------- build triplets --------
    triplets: list[dict[str, str]] = []

    # (1) Card identification
    for cid in tqdm(ids_with_hash, desc="triplets card_id", unit="triplet"):
        anchor = by_id_card[cid]
        negs = hard_negs.get(cid, [])
        if not negs:
            continue
        neg_id = rng.choice(negs)
        neg = by_id_card[neg_id]
        triplets.append(
            {
                "task": "card_id",
                "anchor_id": anchor.card_id,
                "anchor_path": anchor.image_path,
                "positive_id": anchor.card_id,
                "positive_path": anchor.image_path,
                "negative_id": neg.card_id,
                "negative_path": neg.image_path,
                "meta": f"name={anchor.name}|neg_name={neg.name}",
            }
        )

    # Indexes for (2) and (3)
    by_oracle_art_set: dict[tuple[str, str, str], list[CardRec]] = {}
    by_oracle_art: dict[tuple[str, str], list[CardRec]] = {}
    by_set_lang: dict[tuple[str, str], list[CardRec]] = {}
    for r in tqdm(recs_all, desc="index all_cards", unit="card"):
        if not r.oracle_id or not r.artwork_id:
            continue
        by_oracle_art_set.setdefault((r.oracle_id, r.artwork_id, r.set_code), []).append(r)
        by_oracle_art.setdefault((r.oracle_id, r.artwork_id), []).append(r)
        by_set_lang.setdefault((r.set_code, r.lang), []).append(r)

    # (2) Set identification
    for (oracle_id, artwork_id, set_code), rows in tqdm(by_oracle_art_set.items(), desc="triplets set_id", unit="group"):
        langs = {r.lang for r in rows if r.lang}
        if len(langs) < 2:
            continue
        rows_by_lang: dict[str, list[CardRec]] = {}
        for r in rows:
            rows_by_lang.setdefault(r.lang, []).append(r)
        l1, l2 = sorted(langs)[:2]
        a = rng.choice(rows_by_lang[l1])
        p = rng.choice(rows_by_lang[l2])

        neg_pool = [
            r
            for r in by_oracle_art.get((oracle_id, artwork_id), [])
            if r.set_code != set_code and r.lang == l1
        ]
        if not neg_pool:
            neg_pool = [r for r in by_oracle_art.get((oracle_id, artwork_id), []) if r.set_code != set_code]
        if not neg_pool:
            continue
        n = rng.choice(neg_pool)

        triplets.append(
            {
                "task": "set_id",
                "anchor_id": a.card_id,
                "anchor_path": a.image_path,
                "positive_id": p.card_id,
                "positive_path": p.image_path,
                "negative_id": n.card_id,
                "negative_path": n.image_path,
                "meta": f"oracle={oracle_id}|art={artwork_id}|set={set_code}",
            }
        )

    # (3) Language identification
    by_oracle_set: dict[tuple[str, str], list[CardRec]] = {}
    for r in recs_all:
        if r.oracle_id:
            by_oracle_set.setdefault((r.oracle_id, r.set_code), []).append(r)

    for (set_code, lang), rows in tqdm(by_set_lang.items(), desc="triplets lang_id", unit="group"):
        if len(rows) < 2:
            continue
        a, p = rng.sample(rows, 2)
        neg_pool = [r for r in by_oracle_set.get((p.oracle_id, p.set_code), []) if r.lang and r.lang != lang]
        if not neg_pool:
            continue
        n = rng.choice(neg_pool)

        triplets.append(
            {
                "task": "lang_id",
                "anchor_id": a.card_id,
                "anchor_path": a.image_path,
                "positive_id": p.card_id,
                "positive_path": p.image_path,
                "negative_id": n.card_id,
                "negative_path": n.image_path,
                "meta": f"set={set_code}|lang={lang}",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "anchor_id",
                "anchor_path",
                "positive_id",
                "positive_path",
                "negative_id",
                "negative_path",
                "meta",
            ],
        )
        w.writeheader()
        w.writerows(triplets)

    return {
        "records_card": len(recs_card),
        "records_all": len(recs_all),
        "triplets_total": len(triplets),
        "triplets_card_id": sum(1 for t in triplets if t["task"] == "card_id"),
        "triplets_set_id": sum(1 for t in triplets if t["task"] == "set_id"),
        "triplets_lang_id": sum(1 for t in triplets if t["task"] == "lang_id"),
        "hard_neg_anchors": len(hard_negs),
    }
