from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ccg_card_id.config import cfg

from triplets import build_triplets


def main() -> None:
    p = argparse.ArgumentParser(description="Build phase-2 triplets (card/set/language) from Scryfall + pHash")
    p.add_argument("--card-id-cards-json", type=Path, default=cfg.scryfall_default_cards)
    p.add_argument("--all-cards-json", type=Path, default=cfg.scryfall_all_cards)
    p.add_argument("--images-root", type=Path, default=cfg.scryfall_images_dir)
    p.add_argument("--phash-json", type=Path, default=cfg.vectors_file("phash", 8))
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-hard-negs-json", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard-k", type=int, default=24)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--prefix-chars", type=int, default=2)
    p.add_argument("--hard-candidate-cap", type=int, default=2500)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    stats = build_triplets(
        card_id_cards_json=args.card_id_cards_json,
        all_cards_json=args.all_cards_json,
        images_root=args.images_root,
        phash_json=args.phash_json,
        out_csv=args.out_csv,
        out_hard_negs_json=args.out_hard_negs_json,
        seed=args.seed,
        hard_k=args.hard_k,
        checkpoint_every=args.checkpoint_every,
        prefix_chars=args.prefix_chars,
        hard_candidate_cap=args.hard_candidate_cap,
        resume=not args.no_resume,
    )
    print(json.dumps({"triplets_csv": str(args.out_csv), "hard_negs_json": str(args.out_hard_negs_json), **stats}, indent=2))


if __name__ == "__main__":
    main()
