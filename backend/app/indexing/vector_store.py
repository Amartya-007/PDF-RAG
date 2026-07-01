"""High-performance local vector store backed by numpy binary format.

Storage layout (all files live next to `path`):
  <name>.ids.npy   — UTF-8 encoded chunk IDs as a numpy object array
  <name>.mat.npy   — float32 matrix of shape (N, D), pre-L2-normalised
  <name>.json      — legacy JSON fallback (read-only migration path)

Why numpy binary instead of JSON:
- 10–30× faster load/save (no text parsing, direct memcpy)
- 4× smaller on disk (float32 vs decimal text)
- The pre-normalised matrix is persisted, so cosine search on startup
  requires zero additional computation — just load and multiply.
"""
from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """Persistent numpy-backed vector store with O(1) cosine search via BLAS."""

    def __init__(self, path: Path) -> None:
        # `path` is the legacy JSON path; we derive .npy siblings from it.
        self.path = path
        self._ids_path = path.with_suffix(".ids.npy")
        self._mat_path = path.with_suffix(".mat.npy")
        path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._ids: list[str] = []          # chunk_id list (index → id)
        self._id_index: dict[str, int] = {}  # id → row index (O(1) lookup)
        self._matrix: Optional[np.ndarray] = None  # (N, D) normalised float32
        self._dirty = False

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load from .npy files; fall back to legacy JSON on first run."""
        if self._ids_path.exists() and self._mat_path.exists():
            self._load_npy()
        elif self.path.exists():
            self._migrate_from_json()
        else:
            self._ids = []
            self._id_index = {}
            self._matrix = None
        self._dirty = False

    def _load_npy(self) -> None:
        try:
            ids_arr = np.load(str(self._ids_path), allow_pickle=True)
            mat = np.load(str(self._mat_path))
            self._ids = ids_arr.tolist()
            self._id_index = {cid: i for i, cid in enumerate(self._ids)}
            self._matrix = mat.astype(np.float32) if mat.dtype != np.float32 else mat
            logger.debug("vector store loaded: %d vectors from .npy", len(self._ids))
        except Exception as exc:
            logger.warning("Failed to load .npy vector store, resetting: %s", exc)
            self._ids = []
            self._id_index = {}
            self._matrix = None

    def _migrate_from_json(self) -> None:
        """One-time migration from legacy JSON format."""
        try:
            raw: dict[str, list[float]] = json.loads(
                self.path.read_text(encoding="utf-8")
            )
            if not raw:
                self._ids = []
                self._id_index = {}
                self._matrix = None
                return

            # Filter out any entries with inconsistent vector dimensions
            lengths = [len(v) for v in raw.values()]
            if not lengths:
                return
            # Use the most common length as canonical dimension
            from collections import Counter as _Counter
            canonical_dim = _Counter(lengths).most_common(1)[0][0]
            filtered = {k: v for k, v in raw.items() if len(v) == canonical_dim}
            skipped = len(raw) - len(filtered)
            if skipped:
                logger.warning(
                    "JSON migration: skipped %d vectors with wrong dimension "
                    "(expected %d)", skipped, canonical_dim
                )

            if not filtered:
                self._ids = []
                self._id_index = {}
                self._matrix = None
                return

            self._ids = list(filtered.keys())
            self._id_index = {cid: i for i, cid in enumerate(self._ids)}
            mat = np.array(list(filtered.values()), dtype=np.float32)
            self._matrix = _normalise_rows(mat)
            self._dirty = True
            self._save_npy()
            logger.info(
                "migrated %d vectors from JSON to .npy format", len(self._ids)
            )
        except Exception as exc:
            logger.warning("JSON migration failed, starting fresh: %s", exc)
            self._ids = []
            self._id_index = {}
            self._matrix = None

    def _save_npy(self) -> None:
        if not self._dirty:
            return
        ids_arr = np.array(self._ids, dtype=object)
        np.save(str(self._ids_path), ids_arr, allow_pickle=True)
        if self._matrix is not None:
            np.save(str(self._mat_path), self._matrix)
        else:
            # Empty matrix — save a placeholder
            np.save(str(self._mat_path), np.empty((0, 0), dtype=np.float32))
        self._dirty = False

    def save(self) -> None:
        """Flush to disk only when changed."""
        self._save_npy()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert_many(self, vectors: dict[str, list[float]]) -> None:
        """Insert or overwrite vectors. Existing IDs are updated in-place."""
        if not vectors:
            return

        new_ids = [cid for cid in vectors if cid not in self._id_index]
        update_ids = [cid for cid in vectors if cid in self._id_index]

        # Determine embedding dimension
        sample = next(iter(vectors.values()))
        D = len(sample)

        # Grow matrix for new entries
        if new_ids:
            new_mat = np.array(
                [vectors[cid] for cid in new_ids], dtype=np.float32
            )
            new_mat = _normalise_rows(new_mat)
            if self._matrix is None or self._matrix.shape[0] == 0:
                self._matrix = new_mat
            else:
                self._matrix = np.vstack([self._matrix, new_mat])
            for cid in new_ids:
                self._id_index[cid] = len(self._ids)
                self._ids.append(cid)

        # Update existing rows
        for cid in update_ids:
            row = np.array(vectors[cid], dtype=np.float32)
            norm = np.linalg.norm(row)
            if norm > 0:
                row /= norm
            self._matrix[self._id_index[cid]] = row  # type: ignore[index]

        self._dirty = True
        self._save_npy()

    def delete_missing(self, keep_ids: set[str]) -> None:
        """Remove all vectors whose IDs are not in keep_ids."""
        if not self._ids:
            return
        mask = np.array([cid in keep_ids for cid in self._ids], dtype=bool)
        if mask.all():
            return  # nothing to remove
        self._ids = [cid for cid, keep in zip(self._ids, mask) if keep]
        self._id_index = {cid: i for i, cid in enumerate(self._ids)}
        if self._matrix is not None and self._matrix.shape[0] > 0:
            self._matrix = self._matrix[mask]
        self._dirty = True
        self._save_npy()

    def existing_ids(self) -> set[str]:
        return set(self._id_index.keys())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, cosine_similarity) pairs, highest first."""
        if self._matrix is None or len(self._ids) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q /= q_norm

        # Single BLAS matrix-vector multiply → all similarities at once
        sims: np.ndarray = self._matrix @ q  # shape (N,)

        k = min(top_k, len(self._ids))
        if k == len(self._ids):
            top_idx = np.argsort(sims)[::-1]
        else:
            # argpartition is O(N); argsort only over top-k is O(k log k)
            part = np.argpartition(sims, -k)[-k:]
            top_idx = part[np.argsort(sims[part])[::-1]]

        return [(self._ids[i], float(sims[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._ids)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalise_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalise each row in-place and return the matrix."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


# Backward-compat shim used by a few other modules
def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.array(left, dtype=np.float32)
    b = np.array(right, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
