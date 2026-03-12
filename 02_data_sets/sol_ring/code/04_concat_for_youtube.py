#!/usr/bin/env python3
"""
04_concat_for_youtube.py

Concatenates all Sol Ring raw MP4s into a single video with:
  - audio stripped (video track only)
  - 1-second black title cards between each clip ("Sol Ring / Set Name")
  - embedded chapter markers pointing to the card video starts (not the title cards)
  - a YouTube description block printed to stdout with chapter timestamps

Because chapter end times are set explicitly, yt-dlp splits cleanly — each
recovered file contains only the card video, with the title card excluded.

Recipients can recover individual files with:
  yt-dlp --split-chapters -o "%(section_title)s.%(ext)s" <YouTube URL>

Usage:
  # Concatenate only — produces solring_combined.mp4
  python 04_concat_for_youtube.py

  # Concatenate and upload to YouTube (requires one-time OAuth setup, see below)
  python 04_concat_for_youtube.py --upload

  # Skip re-concatenating if the combined file already exists
  python 04_concat_for_youtube.py --upload --skip-concat

  # Skip title-card transitions (plain concat, no reencoding)
  python 04_concat_for_youtube.py --no-transitions

YouTube OAuth setup (one time):
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable "YouTube Data API v3"
  3. Create OAuth 2.0 credentials (Desktop app type)
  4. Download as client_secrets.json and place it next to this script
  5. pip install google-api-python-client google-auth-oauthlib
  6. Run with --upload; a browser window will open for sign-in

Recovery for end users (no setup required):
  pip install yt-dlp
  yt-dlp --split-chapters -o "%(section_title)s.%(ext)s" <YouTube URL>
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]

sys.path.insert(0, str(_project_root))
from ccg_card_id.project_settings import get_data_root

_solring_dir = get_data_root() / "datasets" / "solring"
RAW_DIR = _solring_dir / "01_raw"
OUTPUT_PATH = _solring_dir / "solring_combined.mp4"
CLIENT_SECRETS = _script_dir / "client_secrets.json"
TOKEN_FILE = _script_dir / "youtube_token.json"

YOUTUBE_TITLE = "Sol Ring dataset — 21 printings (MTG card identification)"
YOUTUBE_TAGS = ["magic the gathering", "mtg", "sol ring", "card identification", "dataset"]

TRANSITION_DURATION = 1.0   # seconds of black title card between clips
CARD_NAME = "Sol Ring"      # same for every clip in this dataset

# Human-readable names for each Scryfall set code found in the filenames
SET_NAMES: dict[str, str] = {
    "afc":  "Adventures in the Forgotten Realms Commander",
    "c13":  "Commander 2013",
    "c14":  "Commander 2014",
    "c15":  "Commander 2015",
    "c16":  "Commander 2016",
    "c17":  "Commander 2017",
    "c18":  "Commander 2018",
    "c19":  "Commander 2019",
    "c20":  "Commander 2020",
    "c21":  "Commander 2021",
    "clb":  "Commander Legends: Battle for Baldur's Gate",
    "cma":  "Commander Anthology",
    "cm2":  "Commander Anthology Volume II",
    "cmd":  "Commander (2011)",
    "cmr":  "Commander Legends",
    "dmc":  "Dominaria United Commander",
    "khc":  "Kaldheim Commander",
    "mb1":  "Mystery Booster",
    "nec":  "Kamigawa: Neon Dynasty Commander",
    "phed": "Heads I Win, Tails You Lose",
    "znc":  "Zendikar Rising Commander",
}

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
FONT_SIZE_LARGE = 72
FONT_SIZE_SMALL = 40


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# ---------------------------------------------------------------------------

def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def get_video_specs(path: Path) -> tuple[int, int, str]:
    """Return (width, height, r_frame_rate) from the first video stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(result.stdout)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"]


def parse_set_code(stem: str) -> str:
    """Extract set code from filename stem, e.g. '..._solring_khc_...' → 'khc'."""
    parts = stem.split("_solring_")
    if len(parts) < 2:
        return stem
    return parts[1].split("_")[0]


def make_chapter_title(stem: str) -> str:
    """Build a chapter title (= yt-dlp output filename stem) from a video filename stem.

    Format: {scryfall_uuid}_{CARD_NAME}_{set_name}
    e.g. '0afa0e33-4804-4b00-b625-c2d6b61090fc_Sol Ring_Kaldheim Commander'
    """
    scryfall_id = stem.split("_solring_")[0]
    set_code = parse_set_code(stem)
    set_name = SET_NAMES.get(set_code, set_code.upper())
    return f"{scryfall_id}_{CARD_NAME}_{set_name}"


def format_yt_timestamp(total_seconds: float) -> str:
    s = int(total_seconds)
    h, remainder = divmod(s, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# Transition clip generation
# ---------------------------------------------------------------------------

def generate_transition(
    card_name: str,
    set_name: str,
    w: int,
    h: int,
    fps: str,
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Render a TRANSITION_DURATION-second black clip with card/set text.

    Uses Pillow to draw text onto a PNG (no dependency on ffmpeg's libfreetype),
    then ffmpeg to loop that image into a video clip.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype(FONT_PATH, FONT_SIZE_LARGE, index=0)
        font_small = ImageFont.truetype(FONT_PATH, FONT_SIZE_SMALL, index=0)
    except OSError:
        # Fallback: Pillow's built-in bitmap font (no external file needed)
        font_large = ImageFont.load_default()
        font_small = font_large

    # Center card name slightly above midpoint
    bbox = draw.textbbox((0, 0), card_name, font=font_large)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, h / 2 - th - 16), card_name, font=font_large, fill=(255, 255, 255))

    # Center set name slightly below midpoint
    bbox = draw.textbbox((0, 0), set_name, font=font_small)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, h / 2 + 16), set_name, font=font_small, fill=(187, 187, 187))

    img_path = tmp_dir / "title_card.png"
    img.save(str(img_path))

    # Convert the still image to a video clip at the source frame rate
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(img_path),
            "-t", str(TRANSITION_DURATION),
            "-r", fps,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stderr.decode())
        result.check_returncode()


# ---------------------------------------------------------------------------
# Chapter metadata
# ---------------------------------------------------------------------------

def build_chapters(
    videos: list[tuple[Path, float]],
    with_transitions: bool = True,
) -> list[dict]:
    """
    Chapters point to the card video start/end only — title cards are gaps
    between chapters, so yt-dlp --split-chapters recovers clean card files.
    """
    chapters = []
    t = 0.0
    for path, dur in videos:
        if with_transitions and chapters:  # no leading transition before first clip
            t += TRANSITION_DURATION
        chapters.append({"start": t, "end": t + dur, "title": make_chapter_title(path.stem)})
        t += dur
    return chapters


def write_ffmetadata(chapters: list[dict], dest: Path) -> None:
    lines = [";FFMETADATA1\n"]
    for ch in chapters:
        start_ms = int(ch["start"] * 1000)
        end_ms = int(ch["end"] * 1000)
        lines += [
            "[CHAPTER]\n",
            "TIMEBASE=1/1000\n",
            f"START={start_ms}\n",
            f"END={end_ms}\n",
            f'title={ch["title"]}\n',
            "\n",
        ]
    dest.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def concatenate(
    videos: list[tuple[Path, float]],
    output: Path,
    with_transitions: bool = True,
) -> None:
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        chapters = build_chapters(videos, with_transitions)

        if with_transitions:
            w, h, fps = get_video_specs(videos[0][0])
            print(f"Source video specs: {w}x{h} @ {fps} fps")

            # Generate one title card per clip (re-used between clips)
            print(f"Generating {len(videos)} title cards...")
            trans_clips: list[Path] = []
            for i, (path, _) in enumerate(videos):
                set_code = parse_set_code(path.stem)
                set_name = SET_NAMES.get(set_code, set_code.upper())
                clip_tmp = tmp / f"transition_{i:02d}"
                clip_tmp.mkdir()
                out = clip_tmp / "trans.mp4"
                generate_transition(CARD_NAME, set_name, w, h, fps, out, clip_tmp)
                trans_clips.append(out)

            # Build interleaved list: [vid_0, trans_1, vid_1, trans_2, vid_2, ...]
            # Title card i shows the info for clip i, appears BEFORE clip i
            # (except no title card before the very first clip so 0:00 is card video)
            concat_entries: list[Path] = []
            for i, (path, _) in enumerate(videos):
                if i > 0:
                    concat_entries.append(trans_clips[i])
                concat_entries.append(path)

            encode_args = ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
                           "-pix_fmt", "yuv420p"]
        else:
            concat_entries = [p for p, _ in videos]
            encode_args = ["-c:v", "copy"]

        # Step 1: Concat (reencoding if transitions present, stream-copy if not)
        concat_list = tmp / "concat_list.txt"
        concat_list.write_text(
            "".join(f"file '{p}'\n" for p in concat_entries)
        )
        no_meta_tmp = tmp / "no_meta.mp4"

        print("Concatenating clips...")
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-an", *encode_args,
             str(no_meta_tmp)],
            check=True,
        )

        # Step 2: Inject chapter metadata
        metadata_file = tmp / "chapters.ffmetadata"
        write_ffmetadata(chapters, metadata_file)

        print("Injecting chapter metadata...")
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", str(no_meta_tmp),
             "-i", str(metadata_file),
             "-map_metadata", "1",
             "-c", "copy",
             str(output)],
            check=True,
        )

    size_mb = output.stat().st_size / 1e6
    total_s = sum(d for _, d in videos)
    print(f"\nOutput: {output}  ({size_mb:.0f} MB, {format_yt_timestamp(total_s)} of card footage)")


# ---------------------------------------------------------------------------
# YouTube description
# ---------------------------------------------------------------------------

def build_description(chapters: list[dict], videos: list[tuple[Path, float]]) -> str:
    lines = [
        "Sol Ring dataset — 21 printings of Magic: The Gathering's Sol Ring filmed",
        "under various lighting and orientations. Each chapter is one printing.",
        "",
        "Recover individual video files with yt-dlp:",
        "  pip install yt-dlp",
        '  yt-dlp --split-chapters -o "%(section_title)s.%(ext)s" <this URL>',
        "",
    ]
    for ch, (path, _) in zip(chapters, videos):
        set_code = parse_set_code(path.stem)
        set_name = SET_NAMES.get(set_code, set_code.upper())
        lines.append(f'{format_yt_timestamp(ch["start"])} {CARD_NAME} — {set_name}')
    return "\n".join(lines)


def print_description(description: str) -> None:
    print("\n" + "=" * 70)
    print("PASTE THIS INTO THE YOUTUBE DESCRIPTION FIELD:")
    print("=" * 70)
    print(description)
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# YouTube upload
# ---------------------------------------------------------------------------

def upload_to_youtube(video_path: Path, title: str, description: str) -> str:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        print("ERROR: YouTube upload libraries not installed. Run:")
        print("  pip install google-api-python-client google-auth-oauthlib")
        sys.exit(1)

    if not CLIENT_SECRETS.exists():
        print(f"ERROR: {CLIENT_SECRETS} not found.")
        print("Download OAuth credentials from Google Cloud Console (see script docstring).")
        sys.exit(1)

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
        print(f"Token saved to {TOKEN_FILE}")

    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": YOUTUBE_TAGS,
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": "unlisted",
            "selfDeclaredMadeForKids": False,
        },
    }

    print(f"Uploading {video_path.name} ({video_path.stat().st_size / 1e6:.0f} MB)...")
    media = MediaFileUpload(
        str(video_path), mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024
    )
    request = youtube.videos().insert(
        part=",".join(body.keys()), body=body, media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  Upload progress: {int(status.progress() * 100)}%", end="\r")

    video_id = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"\nDone! Unlisted video: {url}")
    return video_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upload", action="store_true",
                        help="Upload the combined video to YouTube after concatenating")
    parser.add_argument("--skip-concat", action="store_true",
                        help="Skip concatenation if the output file already exists")
    parser.add_argument("--no-transitions", action="store_true",
                        help="Skip title-card transitions (plain concat, no reencoding)")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"ERROR: RAW_DIR not found: {RAW_DIR}")
        sys.exit(1)

    video_paths = sorted(RAW_DIR.glob("*.mp4"))
    if not video_paths:
        print(f"ERROR: No .mp4 files found in {RAW_DIR}")
        sys.exit(1)

    with_transitions = not args.no_transitions

    print(f"Found {len(video_paths)} videos in {RAW_DIR}")
    print("Getting durations...")
    videos = [(p, get_duration(p)) for p in video_paths]
    total = sum(d for _, d in videos)
    print(f"Total card footage: {format_yt_timestamp(total)} ({total:.1f}s)")

    chapters = build_chapters(videos, with_transitions)
    description = build_description(chapters, videos)
    print_description(description)

    if args.skip_concat and OUTPUT_PATH.exists():
        print(f"Skipping concat — {OUTPUT_PATH} already exists.")
    else:
        concatenate(videos, OUTPUT_PATH, with_transitions)

    if args.upload:
        upload_to_youtube(OUTPUT_PATH, YOUTUBE_TITLE, description)
    else:
        print(f"To upload, re-run with:  python {Path(__file__).name} --upload")


if __name__ == "__main__":
    main()
