"""
Central configuration for ccg_card_id.

Data directory resolution order:
  1. CCG_DATA_DIR environment variable (if set)
  2. <project_root>/data_cache/  (legacy location — used if it already exists)
  3. ~/claw/projects/data/       (clean default for new setups)

Usage:
    from ccg_card_id.config import cfg

    print(cfg.scryfall_dir)          # Path to scryfall data
    print(cfg.vectors_dir)           # Path to computed vectors
"""

import os
from pathlib import Path

# Project root = two levels up from this file (ccg_card_id/config.py → ccg_card_id/ → project/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_data_dir() -> Path:
    # 1. Explicit env override
    env = os.environ.get("CCG_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # 2. Legacy data_cache/ in project root (backwards-compatible)
    legacy = _PROJECT_ROOT / "data_cache"
    if legacy.exists():
        return legacy
    # 3. Clean default outside the repo
    return Path.home() / "claw" / "projects" / "data"


class Config:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir: Path = Path(data_dir).resolve() if data_dir else _get_data_dir()

        # Source data
        self.scryfall_dir: Path = self.data_dir / "scryfall"
        self.pokemon_dir: Path = self.data_dir / "pokemon"

        # Computed vectors (one subdir per vectorizer)
        self.vectors_dir: Path = self.data_dir / "vectors"
        self.phash_vectors_file: Path = self.vectors_dir / "scryfall_phash.json"
        self.dinov2_vectors_file: Path = self.vectors_dir / "scryfall_dinov2.npz"

        # Scryfall bulk data files
        self.scryfall_default_cards: Path = self.scryfall_dir / "default_cards.json"
        self.scryfall_all_cards: Path = self.scryfall_dir / "all_cards.json"

        # Scryfall image cache
        self.scryfall_images_dir: Path = self.scryfall_dir / "images" / "png"

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [
            self.scryfall_dir,
            self.pokemon_dir,
            self.vectors_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(data_dir={self.data_dir})"


# Shared default instance — import and use this in scripts
cfg = Config()
