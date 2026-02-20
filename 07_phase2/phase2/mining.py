from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .data import ManifestRow, random_choice


@dataclass
class TripletSample:
    anchor: ManifestRow
    positive: ManifestRow | None
    negative: ManifestRow
    positive_is_synthetic: bool


class TripletMiner:
    """Triplet mining with limited positives + optional hard negatives."""

    def __init__(
        self,
        rows: list[ManifestRow],
        hard_negatives: dict[str, list[str]] | None = None,
        seed: int = 42,
    ):
        self.rows = rows
        self.rng = random.Random(seed)
        self.hard_negatives = hard_negatives or {}
        self.by_card: dict[str, list[ManifestRow]] = defaultdict(list)
        for r in rows:
            self.by_card[r.card_id].append(r)
        self.card_ids = sorted(self.by_card.keys())

    def sample_for_anchor(self, anchor: ManifestRow) -> TripletSample:
        same = [r for r in self.by_card[anchor.card_id] if r.image_path != anchor.image_path]
        positive = same[0] if same else None
        positive_is_synthetic = positive is None

        hard_pool_ids = self.hard_negatives.get(anchor.card_id, [])
        hard_pool = [r for cid in hard_pool_ids for r in self.by_card.get(cid, [])]
        if hard_pool:
            negative = hard_pool[self.rng.randrange(len(hard_pool))]
        else:
            other_ids = [cid for cid in self.card_ids if cid != anchor.card_id]
            neg_card = random_choice(self.rng, other_ids)
            if neg_card is None:
                raise RuntimeError("Cannot sample negative from singleton dataset")
            negative = self.by_card[neg_card][self.rng.randrange(len(self.by_card[neg_card]))]

        return TripletSample(
            anchor=anchor,
            positive=positive,
            negative=negative,
            positive_is_synthetic=positive_is_synthetic,
        )


class PairMiner:
    """Simple pair miner for optional contrastive/pair losses."""

    def __init__(self, rows: list[ManifestRow], seed: int = 42):
        self.rows = rows
        self.rng = random.Random(seed)
        self.by_card: dict[str, list[ManifestRow]] = defaultdict(list)
        for r in rows:
            self.by_card[r.card_id].append(r)
        self.card_ids = sorted(self.by_card.keys())

    def sample_positive_pair(self) -> tuple[ManifestRow, ManifestRow] | None:
        valid = [cid for cid, rows in self.by_card.items() if len(rows) >= 2]
        cid = random_choice(self.rng, valid)
        if cid is None:
            return None
        rows = self.by_card[cid]
        a = rows[self.rng.randrange(len(rows))]
        b = rows[self.rng.randrange(len(rows))]
        return a, b

    def sample_negative_pair(self) -> tuple[ManifestRow, ManifestRow]:
        if len(self.card_ids) < 2:
            raise RuntimeError("Need at least 2 card_ids for negative pair")
        ca = self.card_ids[self.rng.randrange(len(self.card_ids))]
        cb = ca
        while cb == ca:
            cb = self.card_ids[self.rng.randrange(len(self.card_ids))]
        return (
            self.by_card[ca][self.rng.randrange(len(self.by_card[ca]))],
            self.by_card[cb][self.rng.randrange(len(self.by_card[cb]))],
        )
