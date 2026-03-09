#!/usr/bin/env python3
"""Scan a YouTube channel and populate the video registry with new videos.

Uses yt-dlp to enumerate channel videos, then Claude to infer from each title:
  - Whether it is a Magic: The Gathering pack opening
  - Which Scryfall set code(s) it likely covers

New videos are inserted into the local SQLite DB with status 'needs_review'.
A human confirms or corrects set codes before changing status to 'pending'
so the download+match pipeline can proceed.

Usage (run from project root):
    # Scan OpenBoosters channel (default)
    python 02_data_sets/packopening/code/00_fetch_channel.py

    # Scan a different channel
    python 02_data_sets/packopening/code/00_fetch_channel.py \\
        --channel-url "https://www.youtube.com/@SomeOtherChannel"

    # Dry-run: print what would be added without writing to DB
    python 02_data_sets/packopening/code/00_fetch_channel.py --dry-run

    # Skip LLM inference (just add with empty set_codes / status 'pending')
    python 02_data_sets/packopening/code/00_fetch_channel.py --no-llm

Requires:
    pip install yt-dlp anthropic
    ANTHROPIC_API_KEY env var (or in .env)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
from db import open_db, upsert_video

DB_PATH = cfg.data_dir / "datasets" / "packopening" / "packopening.db"

DEFAULT_CHANNEL = "https://www.youtube.com/@OpenBoosters"

# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

_SLUG_BAD_CHARS = re.compile(r"[^a-z0-9]+")


def make_slug(video_id: str, set_codes: str, title: str) -> str:
    """Build a filesystem-safe slug: {video_id}_{set_codes}_{short_title}.

    Examples:
        dQw4w9WgXcQ_lea_alpha-booster-pack
        xYz123_otj-otp_outlaws-of-thunder
    """
    # Normalise set_codes: comma-sep → hyphen-sep, lowercase
    sets_part = re.sub(r"[, ]+", "-", set_codes.strip().lower()) if set_codes else "unknown"
    # Short title: first 4 meaningful words, lowercase, hyphen-joined
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", title).split()
    short = "-".join(w.lower() for w in words[:5] if len(w) > 1)
    short = _SLUG_BAD_CHARS.sub("-", short).strip("-")[:40]
    return f"{video_id}_{sets_part}_{short}"


# ---------------------------------------------------------------------------
# yt-dlp: enumerate channel videos (flat, no download)
# ---------------------------------------------------------------------------

def fetch_channel_videos(channel_url: str) -> list[dict]:
    """Return list of dicts with {video_id, url, title, channel} for all channel videos."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(uploader)s",
        "--no-warnings",
        channel_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        vid_id = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else ""
        channel = parts[2].strip() if len(parts) > 2 else ""
        if vid_id:
            videos.append({
                "video_id": vid_id,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": title,
                "channel": channel,
            })
    return videos


# ---------------------------------------------------------------------------
# LLM: infer set codes from video title
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Magic: The Gathering expert helping to label pack-opening videos for a card identification ML project.

For each video title, decide:
1. is_mtg: Is this a Magic: The Gathering pack opening (or unboxing)? (true/false)
2. set_codes: If MTG, list the Scryfall set code(s) most likely shown in the video.
   - Use lowercase Scryfall codes (e.g. "lea", "leg", "4ed", "otj", "mh3")
   - For pre-release kits include the promo set too (e.g. ["otj","otp","big"])
   - For Commander precons or special sets include all relevant codes
   - Leave empty [] if you can't tell or it's not MTG
3. confidence: "high" / "medium" / "low"
4. notes: short freeform note (set name inference, caveats, etc.)

Respond ONLY with valid JSON matching this schema:
{"is_mtg": bool, "set_codes": [str], "confidence": str, "notes": str}
"""

_BATCH_SYSTEM_PROMPT = """You are a Magic: The Gathering expert helping to label pack-opening videos.

You will receive a JSON array of video titles. For each, decide:
1. is_mtg: Is this a Magic: The Gathering pack opening / booster opening / set review?
2. set_codes: Scryfall set codes (lowercase, e.g. "lea", "otj"). Empty if unknown/not MTG.
3. confidence: "high" / "medium" / "low"

Return a JSON array (same length as input) where each element is:
{"is_mtg": bool, "set_codes": [str], "confidence": str, "notes": str}
"""


def infer_set_codes_batch(titles: list[str], client) -> list[dict]:
    """Call Claude once for a batch of titles. Returns list of classification dicts."""
    import anthropic

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=_BATCH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(titles)}],
    )
    text = response.content[0].text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    results = json.loads(text)
    if len(results) != len(titles):
        raise ValueError(f"LLM returned {len(results)} results for {len(titles)} titles")
    return results


def load_scryfall_sets(data_dir: Path) -> dict[str, str]:
    """Build a mapping of set_code → set_name from default_cards.json.
    Used to validate/display inferred codes. Returns {} if file not found."""
    sets_file = data_dir / "default_cards.json"
    if not sets_file.exists():
        return {}
    seen: dict[str, str] = {}
    try:
        cards = json.loads(sets_file.read_text(encoding="utf-8"))
        for card in cards:
            sc = card.get("set", "").lower()
            sn = card.get("set_name", "")
            if sc and sc not in seen:
                seen[sc] = sn
    except Exception:
        pass
    return seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Scan a YouTube channel and populate the packopening video registry"
    )
    p.add_argument(
        "--channel-url",
        default=DEFAULT_CHANNEL,
        help=f"YouTube channel URL (default: {DEFAULT_CHANNEL})",
    )
    p.add_argument(
        "--channel-name",
        default=None,
        help="Override channel name stored in DB (default: from yt-dlp metadata)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM inference; insert as 'pending'")
    p.add_argument("--batch-size", type=int, default=50, help="Titles per LLM call (default: 50)")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument(
        "--all-videos",
        action="store_true",
        help="Re-evaluate all videos, not just new ones (re-runs LLM on existing needs_review rows)",
    )
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = None if args.dry_run else open_db(db_path)

    # Load existing video IDs to skip already-known videos
    known_ids: set[str] = set()
    if con:
        rows = con.execute("SELECT video_id, status FROM videos").fetchall()
        for r in rows:
            if args.all_videos or r["status"] not in ("needs_review", "pending"):
                pass  # re-evaluate these
            else:
                known_ids.add(r["video_id"])

    print(f"Fetching video list from: {args.channel_url}")
    try:
        videos = fetch_channel_videos(args.channel_url)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(videos)} videos total")
    new_videos = [v for v in videos if v["video_id"] not in known_ids]
    print(f"  {len(new_videos)} new (not yet in DB)")

    if not new_videos:
        print("Nothing to add.")
        return

    # LLM inference
    classifications: list[dict] = []
    if args.no_llm:
        classifications = [{"is_mtg": None, "set_codes": [], "confidence": "none", "notes": ""} for _ in new_videos]
    else:
        try:
            import anthropic
        except ImportError:
            print("anthropic package not installed — run: pip install anthropic")
            sys.exit(1)

        # Load .env for ANTHROPIC_API_KEY
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

        client = anthropic.Anthropic()
        titles = [v["title"] for v in new_videos]
        print(f"  Running LLM classification in batches of {args.batch_size}...")
        for i in range(0, len(titles), args.batch_size):
            batch = titles[i : i + args.batch_size]
            print(f"    Batch {i//args.batch_size + 1}: {len(batch)} titles")
            try:
                batch_results = infer_set_codes_batch(batch, client)
            except Exception as e:
                print(f"    WARNING: LLM failed for this batch: {e}. Marking as needs_review with empty set_codes.")
                batch_results = [{"is_mtg": None, "set_codes": [], "confidence": "error", "notes": str(e)}] * len(batch)
            classifications.extend(batch_results)

    # Insert into DB
    import datetime
    today = datetime.date.today().isoformat()
    added = skipped_non_mtg = 0
    for video, cls in zip(new_videos, classifications):
        is_mtg = cls.get("is_mtg")
        set_codes_list = cls.get("set_codes", [])
        confidence = cls.get("confidence", "")
        notes = cls.get("notes", "")

        if is_mtg is False and confidence in ("high", "medium"):
            # Confidently not MTG — skip
            print(f"  SKIP (not MTG, {confidence}): {video['title'][:70]}")
            skipped_non_mtg += 1
            continue

        set_codes_str = ",".join(set_codes_list) if set_codes_list else ""
        status = "needs_review" if not args.no_llm else "pending"
        slug = make_slug(video["video_id"], set_codes_str, video["title"])
        channel = args.channel_name or video.get("channel", "")

        # Build notes string
        full_notes = f"[auto] confidence={confidence}"
        if notes:
            full_notes += f" | {notes}"
        if is_mtg is None:
            full_notes += " | is_mtg=unknown"

        if args.dry_run:
            print(f"  WOULD ADD: {video['video_id']} | {set_codes_str or '(no set)'} | {status} | {video['title'][:60]}")
        else:
            upsert_video(
                con,
                video_id=video["video_id"],
                slug=slug,
                url=video["url"],
                channel=channel,
                title=video["title"],
                set_codes=set_codes_str,
                status=status,
                added_date=today,
                notes=full_notes,
            )
            added += 1
            print(f"  + {video['video_id']} | {set_codes_str or '(no set)':<20} | {video['title'][:55]}")

    print(f"\nDone. Added {added} new videos. Skipped {skipped_non_mtg} non-MTG.")
    if not args.dry_run and added:
        print(f"DB: {db_path}")
        print(f"\nNext: review 'needs_review' rows and set status to 'pending' for videos to process.")
        print(f"      Then run: python 02_data_sets/packopening/code/01_download.py --all")


if __name__ == "__main__":
    main()
