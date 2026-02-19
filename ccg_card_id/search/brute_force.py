"""
Brute-force flat nearest-neighbor search.

Deliberately simple: no ANN indices, no fancy data structures.
Just numpy. New cards can be appended to the database at any time
without rebuilding anything.

Two distance modes are supported:
  - "hamming"  — for binary/integer hash vectors (e.g. pHash)
  - "cosine"   — for float embedding vectors (e.g. DINOv2)

Typical usage:

    from ccg_card_id.search.brute_force import CardSearchDB

    db = CardSearchDB.from_phash_json("vectors/scryfall_phash.json")
    results = db.search(query_hash_array, k=5)

    db = CardSearchDB.from_dinov2_npz("vectors/scryfall_dinov2.npz")
    results = db.search(query_embedding, k=5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imagehash
import numpy as np


@dataclass
class SearchResult:
    card_id: str
    score: float  # similarity (higher = better) for cosine; distance (lower = better) for hamming


class CardSearchDB:
    """
    In-memory flat card vector database with brute-force search.

    Parameters
    ----------
    card_ids : list[str]
        Ordered list of card identifiers corresponding to rows in `vectors`.
    vectors : np.ndarray
        2D array of shape (n_cards, vector_dim).
    mode : "hamming" | "cosine"
        Distance metric to use during search.
    """

    def __init__(
        self,
        card_ids: list[str],
        vectors: np.ndarray,
        mode: Literal["hamming", "cosine"],
    ):
        if len(card_ids) != vectors.shape[0]:
            raise ValueError(
                f"card_ids length ({len(card_ids)}) must match vectors rows ({vectors.shape[0]})"
            )
        self.card_ids = card_ids
        self.vectors = vectors
        self.mode = mode

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_phash_json(cls, path: str | Path) -> "CardSearchDB":
        """
        Load from a JSON file mapping card_id → pHash hex string.
        Converts hex hashes to flat int arrays for fast Hamming search.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)

        card_ids = list(raw.keys())
        arrays = [imagehash.hex_to_hash(h).hash.flatten().astype(np.int8) for h in raw.values()]
        vectors = np.stack(arrays)  # shape: (n, 64) for 8x8 pHash
        return cls(card_ids, vectors, mode="hamming")

    @classmethod
    def from_dinov2_npz(cls, path: str | Path) -> "CardSearchDB":
        """
        Load from a .npz file with keys 'embeddings' (float32) and 'card_ids' (str array).
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        card_ids = data["card_ids"].tolist()
        vectors = data["embeddings"].astype(np.float32)
        return cls(card_ids, vectors, mode="cosine")

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, card_id: str, vector: np.ndarray) -> None:
        """Add a single card to the database. No rebuild required."""
        self.card_ids.append(card_id)
        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        """
        Find the k nearest cards to `query`.

        Parameters
        ----------
        query : np.ndarray
            1D vector. Must match the shape of stored vectors.
        k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult], sorted best-first.
        """
        if self.mode == "cosine":
            return self._search_cosine(query, k)
        else:
            return self._search_hamming(query, k)

    def search_batch(self, queries: np.ndarray, k: int = 5) -> list[list[SearchResult]]:
        """
        Search for multiple queries at once (more efficient than looping).

        Parameters
        ----------
        queries : np.ndarray
            2D array of shape (n_queries, vector_dim).
        k : int
            Number of results per query.

        Returns
        -------
        list of list[SearchResult], one per query.
        """
        if self.mode == "cosine":
            scores = self._cosine_similarity_matrix(queries)
            top_k_idx = np.argsort(-scores, axis=1)[:, :k]
            return [
                [SearchResult(self.card_ids[j], float(scores[i, j])) for j in top_k_idx[i]]
                for i in range(len(queries))
            ]
        else:
            distances = self._hamming_distance_matrix(queries)
            top_k_idx = np.argsort(distances, axis=1)[:, :k]
            return [
                [SearchResult(self.card_ids[j], float(distances[i, j])) for j in top_k_idx[i]]
                for i in range(len(queries))
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity_matrix(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_gallery) cosine similarity, higher is better."""
        q = queries / (np.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        g = self.vectors / (np.linalg.norm(self.vectors, axis=-1, keepdims=True) + 1e-8)
        return q @ g.T

    def _hamming_distance_matrix(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_gallery) Hamming distance, lower is better."""
        # XOR each query against all gallery rows, then sum differing bits
        # Works for int8 arrays: nonzero count == Hamming distance
        return np.array(
            [np.sum(queries[i] != self.vectors, axis=1) for i in range(len(queries))],
            dtype=np.int32,
        )

    def _search_cosine(self, query: np.ndarray, k: int) -> list[SearchResult]:
        scores = self._cosine_similarity_matrix(query.reshape(1, -1))[0]
        top_k = np.argsort(-scores)[:k]
        return [SearchResult(self.card_ids[i], float(scores[i])) for i in top_k]

    def _search_hamming(self, query: np.ndarray, k: int) -> list[SearchResult]:
        distances = self._hamming_distance_matrix(query.reshape(1, -1))[0]
        top_k = np.argsort(distances)[:k]
        return [SearchResult(self.card_ids[i], float(distances[i])) for i in top_k]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.card_ids)

    def __repr__(self) -> str:
        return (
            f"CardSearchDB(n={len(self)}, dim={self.vectors.shape[1]}, mode={self.mode!r})"
        )
