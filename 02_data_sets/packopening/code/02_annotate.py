#!/usr/bin/env python3
"""Annotate videos in the registry with Scryfall set codes using Claude.

Reads videos with empty set_codes from the SQLite DB, sends their titles to
Claude Haiku in batches, and writes back:
  - set_codes  (comma-separated Scryfall codes, e.g. "otj,otp,big")
  - status     → 'ready' if MTG + set_codes inferred; 'skip' if confidently not MTG
  - slug       (rebuilt to include set_codes in the filename)

Non-MTG videos that Claude is confident about are marked 'skip' and left alone.
Everything else (low confidence, ambiguous, errors) stays 'pending' for manual review.

Re-running is safe: only videos with empty set_codes are processed by default.

Usage (run from project root):
    python 02_data_sets/packopening/code/02_annotate.py

    # Re-annotate videos whose set_codes are already set (e.g. after model update)
    python 02_data_sets/packopening/code/02_annotate.py --re-annotate

    # Process only the first N unannotated videos (useful for testing)
    python 02_data_sets/packopening/code/02_annotate.py --limit 20

    # Preview classifications without writing to DB
    python 02_data_sets/packopening/code/02_annotate.py --dry-run --limit 20

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY in .env or environment
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

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_SYSTEM = """You are a Magic: The Gathering expert classifying pack-opening video titles.

For each title decide:
  is_mtg      — true if this is a Magic: The Gathering pack opening / booster opening / unboxing
  set_codes   — Scryfall set codes (lowercase), e.g. ["lea"] or ["otj","otp","big"]
                Leave [] if not MTG or you can't determine the set.
  confidence  — "high" / "medium" / "low"
  notes       — short freeform note (optional)

Rules:
- Pre-release kits: include main set + promo set (e.g. OTJ → ["otj","otp","big"])
- Commander precons: include the commander set code (e.g. "commander legends" → ["cmr"])
- Collector boosters, set boosters, draft boosters all count as pack openings
- If multiple sets in one video, list all
- Respond ONLY with a JSON array (same length as input), each element:
  {"is_mtg": bool, "set_codes": [str], "confidence": str, "notes": str}
"""


def classify_batch(titles: list[str], client) -> list[dict]:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(titles)}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    results = json.loads(text)
    if len(results) != len(titles):
        raise ValueError(f"Expected {len(titles)} results, got {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate packopening video titles with Scryfall set codes via Claude"
    )
    p.add_argument("--re-annotate", action="store_true",
                   help="Re-classify videos that already have set_codes")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process N videos (0 = all)")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Titles per Claude call (default: 50)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing to DB")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    # Load API key from .env
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic not installed — run: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env", file=sys.stderr)
        sys.exit(1)

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    if args.re_annotate:
        rows = con.execute("SELECT * FROM videos WHERE status NOT IN ('skip','done','processing')").fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM videos WHERE (set_codes IS NULL OR set_codes = '') AND status NOT IN ('skip','done','processing')"
        ).fetchall()

    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("No unannotated videos found.")
        return

    print(f"Annotating {len(rows)} video(s) in batches of {args.batch_size}...")
    client = anthropic.Anthropic()

    updated = skipped = errors = 0
    for i in range(0, len(rows), args.batch_size):
        batch_rows = rows[i : i + args.batch_size]
        titles = [r["title"] for r in batch_rows]
        print(f"  Batch {i // args.batch_size + 1} ({len(titles)} titles)...", end=" ", flush=True)

        try:
            results = classify_batch(titles, client)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += len(batch_rows)
            continue

        for row, cls in zip(batch_rows, results):
            is_mtg = cls.get("is_mtg")
            set_codes_list = cls.get("set_codes", [])
            confidence = cls.get("confidence", "low")
            notes = cls.get("notes", "")

            if is_mtg is False and confidence in ("high", "medium"):
                new_status = "skip"
                set_codes_str = ""
                skipped += 1
            elif set_codes_list:
                new_status = "pending"
                set_codes_str = ",".join(set_codes_list)
                updated += 1
            else:
                # MTG but set unknown, or low confidence — leave for manual review
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
            print(f"      Review {needs_review} videos manually (open DB, check set_codes, set status='pending')")


if __name__ == "__main__":
    main()
