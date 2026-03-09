#!/usr/bin/env python3
"""Annotate videos in the registry with Scryfall set codes using an LLM.

Reads videos with empty set_codes from the SQLite DB, sends their titles to
an LLM in batches, and writes back:
  - set_codes  (comma-separated Scryfall codes, e.g. "otj,otp,big")
  - status     → 'pending' if MTG + set_codes inferred
               → 'needs_review' if MTG but set unknown / low confidence
               → 'skip' if confidently not MTG

Re-running is safe: only videos with empty set_codes are processed by default.

LLM backend (checked in order):
  1. Azure OpenAI  — AZURE_API_KEY + AZURE_API_ENDPOINT in .env
                     optional: AZURE_DEPLOYMENT (default: gpt-4.1-mini)
                     optional: AZURE_API_VERSION (default: 2025-01-01-preview)
  2. Anthropic     — ANTHROPIC_API_KEY in .env

Usage (run from project root):
    python 02_data_sets/packopening/code/02_annotate.py
    python 02_data_sets/packopening/code/02_annotate.py --limit 20 --dry-run
    python 02_data_sets/packopening/code/02_annotate.py --re-annotate

Requires:
    pip install openai          (for Azure)
    pip install anthropic       (for Anthropic fallback)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db


def load_valid_set_codes(data_dir: Path) -> set[str]:
    """Return all Scryfall set codes present in default_cards.json."""
    path = data_dir / "default_cards.json"
    if not path.exists():
        return set()
    cards = json.loads(path.read_text(encoding="utf-8"))
    return {c["set"].lower() for c in cards if "set" in c}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are a Magic: The Gathering expert classifying YouTube video titles for a pack-opening dataset.

For each title return these fields:

  is_mtg      — true if the video is about Magic: The Gathering at all
  is_opening  — true ONLY if the video physically opens packs/boosters/boxes/cases
                (e.g. "opening", "cracking", "unboxing", "let's open", "booster pack opened")
                false for: price discussions, set reviews, spoilers, gameplay, strategy,
                           "what's in a box" explanations, commentary, vlogs
  set_codes   — Scryfall set codes (lowercase) for sets being opened, e.g. ["lea"] or ["otj","otp","big"]
                Leave [] if not an opening or if you can't determine the set.
  confidence  — "high" / "medium" / "low" (your confidence in set_codes)
  notes       — short freeform note (optional)

Rules for set_codes:
- Pre-release kits: include main set + promo set (e.g. OTJ pre-release → ["otj","otp","big"])
- Commander precons: include the commander set code (e.g. "commander legends" → ["cmr"])
- Collector boosters, set boosters, draft boosters all count
- If multiple sets in one video, list all

Respond ONLY with a JSON array (same length as input), each element:
  {"is_mtg": bool, "is_opening": bool, "set_codes": [str], "confidence": str, "notes": str}
"""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text)


def classify_batch_azure(titles: list[str], client, deployment: str) -> list[dict]:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": json.dumps(titles)},
        ],
        max_tokens=4096,
        temperature=0,
    )
    text = _strip_fences(response.choices[0].message.content)
    results = json.loads(text)
    if len(results) != len(titles):
        raise ValueError(f"Expected {len(titles)} results, got {len(results)}")
    return results


def classify_batch_anthropic(titles: list[str], client) -> list[dict]:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(titles)}],
    )
    text = _strip_fences(response.content[0].text)
    results = json.loads(text)
    if len(results) != len(titles):
        raise ValueError(f"Expected {len(titles)} results, got {len(results)}")
    return results


def _is_content_filter_error(e: Exception) -> bool:
    """True if the exception is an Azure content-filter rejection."""
    msg = str(e)
    return "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg or "content management policy" in msg


_FALLBACK = {"is_mtg": None, "set_codes": [], "confidence": "error", "notes": "skipped"}


def classify_with_recovery(titles: list[str], classify_fn) -> list[dict]:
    """Classify a list of titles, recovering from failures by splitting recursively.

    On any error (count mismatch, content filter, API error):
      - If batch size > 1: split in half and retry each half independently.
      - If batch size == 1: mark that title as 'error' and move on.

    Always returns a list of exactly len(titles) dicts.
    """
    if not titles:
        return []

    try:
        results = classify_fn(titles)
        # Validate count — treat mismatch as an error requiring recovery
        if len(results) != len(titles):
            raise ValueError(f"count mismatch: expected {len(titles)}, got {len(results)}")
        return results
    except Exception as e:
        if len(titles) == 1:
            # Can't split further — mark this one item as unclassifiable
            reason = "content_filter" if _is_content_filter_error(e) else f"error: {e}"
            print(f"\n    [skip 1 title — {reason}]: {titles[0][:60]}")
            return [{**_FALLBACK, "notes": reason}]

        # Split and retry each half
        mid = len(titles) // 2
        left = classify_with_recovery(titles[:mid], classify_fn)
        right = classify_with_recovery(titles[mid:], classify_fn)
        return left + right


def build_client():
    """Return (classify_fn, backend_name). Prefers Azure, falls back to Anthropic."""
    # Load .env
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    azure_key = os.environ.get("AZURE_API_KEY")
    azure_endpoint = os.environ.get("AZURE_API_ENDPOINT")

    if azure_key and azure_endpoint:
        try:
            from openai import AzureOpenAI
        except ImportError:
            print("ERROR: openai not installed — run: pip install openai", file=sys.stderr)
            sys.exit(1)
        deployment = os.environ.get("AZURE_DEPLOYMENT", "gpt-4.1-mini")
        api_version = os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        classify_fn = lambda titles: classify_batch_azure(titles, client, deployment)
        return classify_fn, f"Azure OpenAI ({deployment})"

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
        except ImportError:
            print("ERROR: anthropic not installed — run: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=anthropic_key)
        classify_fn = lambda titles: classify_batch_anthropic(titles, client)
        return classify_fn, "Anthropic (claude-haiku)"

    print(
        "ERROR: no LLM credentials found.\n"
        "  Set AZURE_API_KEY + AZURE_API_ENDPOINT in .env  (Azure OpenAI)\n"
        "  or ANTHROPIC_API_KEY in .env                    (Anthropic)",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate packopening video titles with Scryfall set codes"
    )
    p.add_argument("--re-annotate", action="store_true",
                   help="Re-classify videos that already have set_codes")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process N videos (0 = all)")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Titles per LLM call (default: 50)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing to DB")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    classify_fn, backend_name = build_client()
    print(f"LLM backend: {backend_name}")

    print("Loading valid Scryfall set codes...", end=" ", flush=True)
    valid_set_codes = load_valid_set_codes(args.data_dir)
    print(f"{len(valid_set_codes)} sets")

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    if args.re_annotate:
        rows = con.execute(
            "SELECT * FROM videos WHERE status NOT IN ('skip','done','processing','downloading','downloaded','frames_extracted')"
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM videos WHERE status = 'new'"
        ).fetchall()

    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("No unannotated videos found.")
        return

    print(f"Annotating {len(rows)} video(s) in batches of {args.batch_size}...")

    updated = skipped = errors = 0
    for i in range(0, len(rows), args.batch_size):
        batch_rows = rows[i : i + args.batch_size]
        titles = [r["title"] for r in batch_rows]
        batch_num = i // args.batch_size + 1
        total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(titles)} titles)...", end=" ", flush=True)

        results = classify_with_recovery(titles, classify_fn)
        print("OK")

        for row, cls in zip(batch_rows, results):
            is_mtg = cls.get("is_mtg")
            is_opening = cls.get("is_opening", False)
            set_codes_list = cls.get("set_codes", [])
            confidence = cls.get("confidence", "low")
            notes = cls.get("notes", "")

            # Validate set codes against Scryfall — flag hallucinated codes
            if valid_set_codes and set_codes_list:
                bad = [c for c in set_codes_list if c not in valid_set_codes]
                if bad:
                    notes = (notes + f" | bad_codes:{','.join(bad)}").strip(" |")
                    set_codes_list = [c for c in set_codes_list if c in valid_set_codes]
                    if not set_codes_list:
                        confidence = "low"  # all codes were invalid

            if confidence == "error":
                new_status = "needs_review"
                set_codes_str = ""
                errors += 1
            elif not is_mtg and confidence in ("high", "medium"):
                # Confidently not MTG at all
                new_status = "skip"
                set_codes_str = ""
                skipped += 1
            elif is_mtg and not is_opening:
                # MTG content but not a pack opening (discussion, review, gameplay, etc.)
                new_status = "skip"
                set_codes_str = ""
                skipped += 1
            elif is_opening and set_codes_list:
                new_status = "pending"
                set_codes_str = ",".join(set_codes_list)
                updated += 1
            else:
                # Opening but set unknown, or low confidence
                new_status = "needs_review"
                set_codes_str = ""
                updated += 1

            annotation_note = f"[llm] {confidence}"
            if notes:
                annotation_note += f": {notes}"

            if args.dry_run:
                marker = "SKIP" if new_status == "skip" else set_codes_str or "(no set)"
                print(f"    {row['video_id']}  {marker:<25}  {row['title'][:55]}")
            else:
                con.execute(
                    "UPDATE videos SET set_codes=?, status=?, notes=? WHERE video_id=?",
                    (set_codes_str, new_status, annotation_note, row["video_id"]),
                )
        if not args.dry_run:
            con.commit()

    print(f"\nDone. Updated: {updated}  Marked skip: {skipped}  Errors: {errors}")
    if not args.dry_run:
        pending = con.execute("SELECT COUNT(*) FROM videos WHERE status='pending'").fetchone()[0]
        needs_review = con.execute("SELECT COUNT(*) FROM videos WHERE status='needs_review'").fetchone()[0]
        print(f"DB status — pending: {pending}  needs_review: {needs_review}")
        if pending:
            print(f"\nNext: python 02_data_sets/packopening/code/pipeline/01_download.py --all")
        if needs_review:
            print(f"      Review {needs_review} videos manually (open DB, set status='pending')")


if __name__ == "__main__":
    main()
