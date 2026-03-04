from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccg_card_id.config import cfg

from data import build_manifest_from_scryfall


def main() -> None:
    p = argparse.ArgumentParser(description="Build reproducible MobileViT-XXS training manifest")
    p.add_argument("--default-cards-json", type=Path, default=cfg.scryfall_all_cards)
    p.add_argument("--images-root", type=Path, default=cfg.scryfall_images_dir)
    p.add_argument("--out", type=Path, default=cfg.data_dir / "mobilevit_xxs" / "manifest.csv")
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-languages", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true", help="Rebuild manifest even if output CSV already exists")
    args = p.parse_args()

    if args.out.exists() and not args.rebuild_cache:
        with args.out.open("r", newline="", encoding="utf-8") as f:
            rows = sum(1 for _ in csv.DictReader(f))
        print(json.dumps({
            "manifest": str(args.out),
            "rows": rows,
            "cached": True,
            "hint": "Use --rebuild-cache to regenerate manifest",
        }, indent=2))
        return

    stats = build_manifest_from_scryfall(
        default_cards_json=args.default_cards_json,
        images_root=args.images_root,
        out_csv=args.out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        english_only=not args.all_languages,
    )
    print(json.dumps({"manifest": str(args.out), **stats}, indent=2))


if __name__ == "__main__":
    main()
