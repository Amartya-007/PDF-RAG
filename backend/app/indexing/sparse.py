from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from backend.app.core.text import tokenize
from backend.app.models import Chunk


class BM25Index:
    def __init__(self, path: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.path = path
        self.k1 = k1
        self.b = b
        self.doc_lengths: dict[str, int] = {}
        self.term_freqs: dict[str, dict[str, int]] = {}
        self.doc_freqs: dict[str, int] = {}
        self.avg_doc_length = 0.0
        self.load()

    def build(self, chunks: list[Chunk]) -> None:
        self.doc_lengths = {}
        self.term_freqs = {}
        doc_freqs: defaultdict[str, int] = defaultdict(int)

        for chunk in chunks:
            counts = Counter(tokenize(chunk.text))
            self.term_freqs[chunk.chunk_id] = dict(counts)
            self.doc_lengths[chunk.chunk_id] = sum(counts.values())
            for term in counts:
                doc_freqs[term] += 1

        self.doc_freqs = dict(doc_freqs)
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.save()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        total_docs = len(self.doc_lengths)
        if not query_terms or total_docs == 0:
            return []

        scores: defaultdict[str, float] = defaultdict(float)
        for term in query_terms:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for chunk_id, freqs in self.term_freqs.items():
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                doc_length = self.doc_lengths[chunk_id]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / (self.avg_doc_length or 1.0)
                )
                scores[chunk_id] += idf * (tf * (self.k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "doc_lengths": self.doc_lengths,
                    "term_freqs": self.term_freqs,
                    "doc_freqs": self.doc_freqs,
                    "avg_doc_length": self.avg_doc_length,
                }
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.doc_lengths = data.get("doc_lengths", {})
        self.term_freqs = data.get("term_freqs", {})
        self.doc_freqs = data.get("doc_freqs", {})
        self.avg_doc_length = float(data.get("avg_doc_length", 0.0))
