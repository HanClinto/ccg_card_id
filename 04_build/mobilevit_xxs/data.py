from __future__ import annotations

import csv
import hashlib
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

# catalog is imported lazily inside build_manifest_from_scryfall so this module
# can be imported without requiring the DB to exist yet.

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    card_id: str
    card_name: str
    set_code: str
    split: str
    # Extended fields — empty string when not present (backwards-compatible)
    illustration_id: str = ""
    oracle_id: str = ""
    lang: str = ""
    source: str = ""


def deterministic_split(card_id: str, train: float, val: float, seed: int) -> str:
    key = f"{seed}:{card_id}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < (train + val):
        return "val"
    return "test"


def _build_image_index(images_root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for p in tqdm(images_root.rglob("*"), desc=f"index images {images_root.name}", unit="file"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        idx.setdefault(p.stem, []).append(p)
    return idx


def _find_image_for_card(image_index: dict[str, list[Path]], card_id: str) -> Path | None:
    candidates = image_index.get(card_id)
    if not candidates:
        return None
    for c in candidates:
        if "/front/" in c.as_posix():
            return c
    return candidates[0]


def build_manifest_from_scryfall(
    *,
    images_root: Path,
    out_csv: Path,
    train_ratio: float = 0.85,
    val_ratio: float = 0.1,
    seed: int = 42,
    lang: str = "en",
    # Legacy parameter kept for call-site compatibility — ignored when catalog is available
    default_cards_json: Path | None = None,
    english_only: bool = True,
) -> dict[str, int]:
    from ccg_card_id.catalog import catalog as _catalog

    # Resolve effective language filter
    effective_lang = lang if lang else ("en" if english_only else None)

    data = _catalog.all_cards(lang=effective_lang)
    image_index = _build_image_index(images_root)

    rows: list[ManifestRow] = []
    missing_image = 0

    for card in tqdm(data, desc="build manifest", unit="card"):
        card_id = str(card.get("id", "")).strip()
        if not card_id:
            continue

        img = _find_image_for_card(image_index, card_id)
        if img is None:
            missing_image += 1
            continue

        rows.append(
            ManifestRow(
                image_path=str(img),
                card_id=card_id,
                card_name=str(card.get("name", "")).strip(),
                set_code=str(card.get("set_code", "")).strip(),
                split=deterministic_split(card_id, train_ratio, val_ratio, seed),
                illustration_id=str(card.get("illustration_id", "") or ""),
                oracle_id=str(card.get("oracle_id", "") or ""),
                lang=str(card.get("lang", "") or ""),
                source="scryfall",
            )
        )

    _FIELDNAMES = ["image_path", "card_id", "card_name", "set_code", "split",
                   "illustration_id", "oracle_id", "lang", "source"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in _FIELDNAMES})

    by_split = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        by_split[r.split] += 1

    return {
        "rows": len(rows),
        "missing_image": missing_image,
        "train": by_split["train"],
        "val": by_split["val"],
        "test": by_split["test"],
    }


def load_manifest(path: Path, split: str | None = None) -> list[ManifestRow]:
    out: list[ManifestRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = ManifestRow(
                image_path=row["image_path"],
                card_id=row["card_id"],
                card_name=row.get("card_name", ""),
                set_code=row.get("set_code", ""),
                split=row.get("split", "train"),
                illustration_id=row.get("illustration_id", ""),
                oracle_id=row.get("oracle_id", ""),
                lang=row.get("lang", ""),
                source=row.get("source", ""),
            )
            if split is None or r.split == split:
                out.append(r)
    return out


def extract_card_id_from_filename(name: str) -> str | None:
    m = UUID_RE.search(name)
    return m.group(0) if m else None


def read_hard_negatives(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            true_id = str(row.get("true_id", "")).strip()
            pred_id = str(row.get("predicted_id", "")).strip()
            if true_id and pred_id and true_id != pred_id:
                out.setdefault(true_id, set()).add(pred_id)
    return {k: sorted(v) for k, v in out.items()}


def random_choice(rng: random.Random, items: Iterable[str]) -> str | None:
    items = list(items)
    if not items:
        return None
    return items[rng.randrange(len(items))]
