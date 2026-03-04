from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
UUID_PREFIX_RE = re.compile(r"^([0-9a-fA-F-]{36})_([^_]+)")
DANIEL_RE = re.compile(r"^([^_]+)_([^_]+)_([^_]+)_([0-9a-fA-F-]{36})")


@dataclass
class Row:
    image_path: str
    class_id: str
    class_name: str
    label_type: str  # positive | negative
    source: str
    split: str
    quality: str


def deterministic_split(key: str, train: float, val: float, seed: int) -> str:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def parse_clint_name(name: str) -> tuple[str, str] | None:
    m = UUID_PREFIX_RE.match(name)
    if not m:
        return None
    return m.group(1).lower(), m.group(2)


def parse_daniel_name(name: str) -> tuple[str, str] | None:
    m = DANIEL_RE.match(name)
    if not m:
        return None
    card_name, set_code, _, card_id = m.groups()
    return f"{card_name.lower()}:{set_code.lower()}:{card_id.lower()}", f"{card_name}_{set_code}"


def build_rows(
    datasets_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[Row]:
    rows: list[Row] = []

    clint_bg_good = datasets_root / "clint_cards_with_backgrounds" / "data" / "04_data" / "good"
    clint_bg_bad = datasets_root / "clint_cards_with_backgrounds" / "data" / "04_data" / "bad"
    clint_sol_good = datasets_root / "clint_cards_solring" / "data" / "04_data" / "good"
    daniel = datasets_root / "daniel_scans" / "images_processed"

    for p in iter_images(clint_bg_good):
        parsed = parse_clint_name(p.name)
        if not parsed:
            continue
        class_id, class_name = parsed
        rows.append(
            Row(
                image_path=str(p),
                class_id=class_id,
                class_name=class_name,
                label_type="positive",
                source="clint_cards_with_backgrounds/good",
                split=deterministic_split(class_id, train_ratio, val_ratio, seed),
                quality="real_camera",
            )
        )

    for p in iter_images(clint_sol_good):
        parsed = parse_clint_name(p.name)
        if not parsed:
            continue
        class_id, class_name = parsed
        rows.append(
            Row(
                image_path=str(p),
                class_id=class_id,
                class_name=class_name,
                label_type="positive",
                source="clint_cards_solring/good",
                split=deterministic_split(class_id, train_ratio, val_ratio, seed),
                quality="real_camera",
            )
        )

    for p in iter_images(clint_bg_bad):
        parsed = parse_clint_name(p.name)
        if not parsed:
            continue
        class_id, class_name = parsed
        rows.append(
            Row(
                image_path=str(p),
                class_id=class_id,
                class_name=class_name,
                label_type="negative",
                source="clint_cards_with_backgrounds/bad",
                split=deterministic_split(f"neg:{p.name}", train_ratio, val_ratio, seed),
                quality="hard_negative",
            )
        )

    for p in iter_images(daniel):
        parsed = parse_daniel_name(p.name)
        if not parsed:
            continue
        class_id, class_name = parsed
        rows.append(
            Row(
                image_path=str(p),
                class_id=class_id,
                class_name=class_name,
                label_type="positive",
                source="daniel_scans/images_processed",
                split=deterministic_split(class_id, train_ratio, val_ratio, seed),
                quality="real_camera",
            )
        )

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build local real-image manifest for MobileViT-XXS training.")
    p.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/Users/hanclaw/claw/data/ccg_card_id/datasets"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/hanclaw/claw/projects/ccg_card_id/cache/mobilevit_xxs/local_real_manifest.csv"),
    )
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = build_rows(args.datasets_root, args.train_ratio, args.val_ratio, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "class_id", "class_name", "label_type", "source", "split", "quality"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
