"""Utilities for stable identifier generation and file hashing.

Requirements: 21.7
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hex digest of a byte string.

    Attributes:
        data: The byte string to hash.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file in memory-efficient chunks.

    Attributes:
        path:       Path to the file to hash.
        chunk_size: Size of read buffer in bytes (default 64KB).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Use a 64KB buffer (65536 bytes) to maximize I/O performance
        # while keeping memory usage extremely low for massive files.
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    """Generate a stable, deterministic ID using a SHA-256 hash.

    Attributes:
        prefix: A string prefix to prepend to the ID (e.g., "doc").
        *parts: Variable arguments used as inputs to the hash.
        length: Number of hex characters to include in the ID.
    """
    # Use \x1f (Unit Separator) to prevent collision issues where
    # different combinations of inputs could result in the same hash.
    raw = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"