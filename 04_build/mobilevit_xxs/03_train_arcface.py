from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train_arcface import build_parser, run


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
