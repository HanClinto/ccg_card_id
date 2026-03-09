#!/usr/bin/env python3
"""Resolve each munchie scan to a specific Scryfall card (card_id, illustration_id, oracle_id).

Reads master.json (ocrName + set) and cross-references against all_cards.json.

Resolution statuses written to resolved.jsonl:
  resolved            — unique English (name, set) match
  ambiguous_same_art  — multiple matches but all share the same illustration_id (still usable)
  ambiguous           — multiple matches with different illustration_ids (skip for training)
  not_found           — no English card found with this name+set

Input:
  <data_dir>/datasets/munchie/data/master.json
  <data_dir>/all_cards.json

Output:
  <data_dir>/datasets/munchie/resolved.jsonl  (one JSON object per line)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
from ccg_card_id.catalog import catalog

# ------------------------------------------------------------------
# Images that should be excluded based on known quality issues.
# Matched against the 'name' field (filename) in master.json.
# ------------------------------------------------------------------
_KNOWN_BAD_SUBSTRINGS: list[str] = [
    # Upside-down scans
    "CardId_MGE__320212",
    # Confirmed non-English
    "LL_B7_2538",
    "CardId_LL_B33_4994",
]

_OFF_CENTER_NUMBERS = {
    # Numbers referenced in munchie_notes.txt as "way off center"
    "25927", "25928", "25929", "25930", "25931",
    "26300", "26301", "26332", "27319",
    "30305", "32923", "32924",
    "VP_12057",
}


def _is_known_bad(name: str) -> str | None:
    """Return a reason string if this filename should be excluded, else None."""
    for bad in _KNOWN_BAD_SUBSTRINGS:
        if bad in name:
            return f"known_bad:{bad}"
    # Check off-center numeric IDs (e.g. DIG1_27319.jpg → "27319")
    stem = Path(name).stem  # e.g. "DIG1_27319"
    suffix = stem.rsplit("_", 1)[-1]
    if suffix in _OFF_CENTER_NUMBERS:
        return f"off_center:{suffix}"
    return None


def _resolve_record(record: dict) -> dict:
    name = str(record.get("ocrName") or "").strip()
    set_code = str(record.get("set") or "").strip()
    front_file = str(record.get("name") or "").strip()

    base: dict = {
        "munchie_id": record.get("cardId", ""),
        "front_filename": front_file,
        "ocr_name": name,
        "set_code": set_code.upper(),
        "foil": record.get("foil", ""),
        "condition": record.get("condi", ""),
    }

    bad_reason = _is_known_bad(front_file)
    if bad_reason:
        return {**base, "status": "excluded", "exclude_reason": bad_reason}

    matches = catalog.cards_by_name_set(name, set_code, lang="en")

    if len(matches) == 0:
        return {**base, "status": "not_found"}

    if len(matches) == 1:
        c = matches[0]
        return {
            **base,
            "status": "resolved",
            "card_id": c["id"],
            "card_name": c["name"],
            "illustration_id": c.get("illustration_id", ""),
            "oracle_id": c.get("oracle_id", ""),
            "lang": c.get("lang", "en"),
        }

    # Multiple matches — check if all share the same illustration_id
    illus_ids = {c.get("illustration_id", "") for c in matches}
    illus_ids.discard("")
    if len(illus_ids) == 1:
        # All variants have the same artwork — safe to use for Phase 1 training
        c = matches[0]
        return {
            **base,
            "status": "ambiguous_same_art",
            "card_id": c["id"],
            "card_name": c["name"],
            "illustration_id": illus_ids.pop(),
            "oracle_id": c.get("oracle_id", ""),
            "lang": c.get("lang", "en"),
            "ambiguous_count": len(matches),
        }

    return {
        **base,
        "status": "ambiguous",
        "ambiguous_count": len(matches),
        "illustration_ids": sorted(illus_ids),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve munchie records to Scryfall card IDs")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--rebuild", action="store_true", help="Overwrite existing resolved.jsonl")
    args = p.parse_args()

    master_path = args.data_dir / "datasets" / "munchie" / "data" / "master.json"
    out_path = args.data_dir / "datasets" / "munchie" / "resolved.jsonl"

    if not master_path.exists():
        raise FileNotFoundError(f"master.json not found: {master_path}")

    if out_path.exists() and not args.rebuild:
        print(f"resolved.jsonl already exists at {out_path}  (pass --rebuild to regenerate)")
        return

    master: list[dict] = json.loads(master_path.read_text(encoding="utf-8"))
    print(f"Resolving {len(master)} munchie records …")

    results: list[dict] = []
    counts: dict[str, int] = {}
    for record in master:
        r = _resolve_record(record)
        results.append(r)
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResolution summary ({len(results)} total):")
    for status in ["resolved", "ambiguous_same_art", "ambiguous", "not_found", "excluded"]:
        n = counts.get(status, 0)
        print(f"  {status:<22} {n:4d}")
    usable = counts.get("resolved", 0) + counts.get("ambiguous_same_art", 0)
    print(f"  {'usable (Phase 1)':<22} {usable:4d}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
