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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imagehash
import numpy as np

_POPCOUNT_U8 = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


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
        Load from a JSON file mapping card_id → perceptual hash hex string.

        Memory-optimized: store hashes as packed bytes (uint8) instead of
        unpacked bits (int8). This keeps large hash sizes (e.g. 256x256 bits)
        practical in RAM.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)

        card_ids = list(raw.keys())
        hex_values = list(raw.values())
        if not hex_values:
            return cls([], np.zeros((0, 0), dtype=np.uint8), mode="hamming")

        n = len(hex_values)
        # Hash strings may occasionally miss leading zeros (or contain outliers).
        # Use the most common hex length as canonical size.
        length_counts = Counter(len(h) for h in hex_values)
        canon_hex_len = length_counts.most_common(1)[0][0]
        if canon_hex_len % 2 == 1:
            canon_hex_len += 1
        bytes_per_hash = canon_hex_len // 2

        vectors = np.empty((n, bytes_per_hash), dtype=np.uint8)
        for i, h in enumerate(hex_values):
            h_norm = h.strip()
            if len(h_norm) < canon_hex_len:
                h_norm = h_norm.zfill(canon_hex_len)
            elif len(h_norm) > canon_hex_len:
                h_norm = h_norm[-canon_hex_len:]
            vectors[i] = np.frombuffer(bytes.fromhex(h_norm), dtype=np.uint8)

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
            q = self._prepare_hamming_queries(queries)
            distances = self._hamming_distance_matrix(q)
            top_k_idx = np.argsort(distances, axis=1)[:, :k]
            return [
                [SearchResult(self.card_ids[j], float(distances[i, j])) for j in top_k_idx[i]]
                for i in range(len(q))
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
        # Packed mode (uint8): XOR bytes then popcount bits.
        if self.vectors.dtype == np.uint8:
            out = np.empty((len(queries), len(self.card_ids)), dtype=np.int32)
            for i in range(len(queries)):
                q = queries[i]
                xor = np.bitwise_xor(self.vectors, q)
                out[i] = _POPCOUNT_U8[xor].sum(axis=1, dtype=np.int32)
            return out

        # Legacy unpacked mode (int8 bits).
        return np.array(
            [np.sum(queries[i] != self.vectors, axis=1) for i in range(len(queries))],
            dtype=np.int32,
        )

    def _search_cosine(self, query: np.ndarray, k: int) -> list[SearchResult]:
        scores = self._cosine_similarity_matrix(query.reshape(1, -1))[0]
        top_k = np.argsort(-scores)[:k]
        return [SearchResult(self.card_ids[i], float(scores[i])) for i in top_k]

    def _prepare_hamming_queries(self, queries: np.ndarray) -> np.ndarray:
        """Convert unpacked bit queries to packed-byte queries when needed."""
        q = np.asarray(queries)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if self.vectors.dtype != np.uint8:
            return q

        # Already packed.
        if q.shape[1] == self.vectors.shape[1] and q.dtype == np.uint8:
            return q

        # Unpacked bits -> pack to bytes (allow non-multiple-of-8 bit lengths).
        target_bits = self.vectors.shape[1] * 8
        if q.shape[1] <= target_bits:
            q_u8 = (q > 0).astype(np.uint8)
            if q_u8.shape[1] < target_bits:
                pad = np.zeros((q_u8.shape[0], target_bits - q_u8.shape[1]), dtype=np.uint8)
                q_u8 = np.concatenate([q_u8, pad], axis=1)
            return np.packbits(q_u8, axis=1)

        raise ValueError(
            f"Hamming query dim mismatch: got {q.shape[1]}, expected <= {self.vectors.shape[1] * 8} bits "
            f"or {self.vectors.shape[1]} packed bytes"
        )

    def _search_hamming(self, query: np.ndarray, k: int) -> list[SearchResult]:
        q = self._prepare_hamming_queries(query)
        distances = self._hamming_distance_matrix(q)[0]
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
