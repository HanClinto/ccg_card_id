"""
Central configuration for ccg_card_id.

Data directory resolution order:
  1. CCG_DATA_DIR environment variable (if set)
  2. ~/claw/data/ccg_card_id/  (default)

To use a different location (e.g. an external drive):
    export CCG_DATA_DIR=/Volumes/MyDrive/ccg_card_id

Usage:
    from ccg_card_id.config import cfg

    print(cfg.data_dir)                  # ~/claw/data/ccg_card_id
    print(cfg.scryfall_images_dir)       # .../catalog/scryfall/images/png
    print(cfg.vectors_file("phash",64)) # .../default_cards_phash_64.json
"""

from pathlib import Path

from .project_settings import get_data_dir

# Official Meta DINOv2 model variants (smallest → largest)
DINOV2_MODELS: dict[str, str] = {
    "small": "facebook/dinov2-small",   # ViT-S/14 — 21M params, 384-dim  (default)
    "base":  "facebook/dinov2-base",    # ViT-B/14 — 86M params, 768-dim
    "large": "facebook/dinov2-large",   # ViT-L/14 — 307M params, 1024-dim
    "giant": "facebook/dinov2-giant",   # ViT-G/14 — 1.1B params, 1536-dim
}


def _get_data_dir() -> Path:
    return get_data_dir()


class Config:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir: Path = Path(data_dir).resolve() if data_dir else _get_data_dir()

        # Scryfall bulk data files (flat layout — all in data_dir)
        self.scryfall_default_cards: Path = self.data_dir / "default_cards.json"
        self.scryfall_all_cards: Path = self.data_dir / "all_cards.json"

        # Scryfall catalog reference images
        self.scryfall_images_dir: Path = self.data_dir / "catalog" / "scryfall" / "images" / "png"

    # ------------------------------------------------------------------
    # Vector file paths — named by method and hash_size grid dimension
    # e.g. phash_vectors_file("phash", 64) → default_cards_phash_64.json
    #      phash_vectors_file("whash_db4", 32) → default_cards_whash_db4_32.json
    # ------------------------------------------------------------------

    def dinov2_vectors_file(self, variant: str) -> Path:
        """
        Path to a pre-computed DINOv2 embedding file.

        Parameters
        ----------
        variant : str
            Model size name, e.g. "small", "base", "large", "giant"

        Returns
        -------
        Path to .npz file (may or may not exist yet)
        e.g. ~/claw/data/ccg_card_id/default_cards_dinov2_small.npz
        """
        return self.data_dir / f"default_cards_dinov2_{variant}.npz"

    def vectors_file(self, method: str, hash_size: int) -> Path:
        """
        Path to a pre-computed vector file.

        Parameters
        ----------
        method : str
            Hash method name, e.g. "phash", "whash_db4"
        hash_size : int
            Grid dimension used when building vectors, e.g. 32, 64, 128, 256

        Returns
        -------
        Path to JSON file (may or may not exist yet)
        """
        return self.data_dir / f"default_cards_{method}_{hash_size}.json"

    def ensure_dirs(self) -> None:
        """Create the data directory and image subdirs if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scryfall_images_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(data_dir={self.data_dir})"


# Shared default instance — import and use this in scripts
cfg = Config()
