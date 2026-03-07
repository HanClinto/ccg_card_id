"""Project-wide settings loader.

Loads optional key=value pairs from `.env` in the project root, then resolves
portable data paths from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_data_root() -> Path:
    _load_dotenv()
    return Path(os.environ.get("CCG_DATA_ROOT", "~/claw/data")).expanduser().resolve()


def get_data_dir() -> Path:
    _load_dotenv()
    explicit = os.environ.get("CCG_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (get_data_root()).resolve()
