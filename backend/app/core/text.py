"""Text processing utilities for RAG ingestion.

Requirements: 21.7
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import batched

# Pre-compiled regex for performance
WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?")


def normalize_space(text: str) -> str:
    """Normalize whitespace by replacing newlines/tabs with spaces and stripping.

    Attributes:
        text: Raw input string.
    """
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Break text into a list of lowercase word tokens.

    Attributes:
        text: Raw input string.
    """
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def split_paragraphs(text: str) -> list[str]:
    """Split text into distinct paragraph blocks.

    Attributes:
        text: Raw input string.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n+", normalized)
    return [normalize_space(block) for block in blocks if normalize_space(block)]


def get_batches(items: list[str], batch_size: int) -> Iterable[tuple[str, ...]]:
    """Group items into fixed-size batches using optimized itertools.

    Attributes:
        items:      List of strings to process.
        batch_size: Number of items per batch.
    """
    return batched(items, batch_size)


def truncate_words(text: str, max_words: int) -> str:
    """Truncate text to a maximum word count, appending an ellipsis.

    Attributes:
        text:      Raw input string.
        max_words: Maximum number of words allowed.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + "..."