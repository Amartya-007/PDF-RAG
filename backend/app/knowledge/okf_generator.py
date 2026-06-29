from __future__ import annotations

import re
from pathlib import Path

from backend.app.core.hashing import stable_id
from backend.app.core.text import tokenize, truncate_words
from backend.app.database.store import MetadataStore
from backend.app.models import Chunk


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "was",
    "were",
    "have",
    "has",
    "into",
    "shall",
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "concept"


class OkfGenerator:
    def __init__(self, okf_dir: Path, store: MetadataStore) -> None:
        self.okf_dir = okf_dir
        self.store = store
        self.okf_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_document(self, chunks: list[Chunk], max_concepts: int = 12) -> list[Path]:
        terms = self._top_terms(chunks, max_concepts)
        paths: list[Path] = []
        for term in terms:
            related = [chunk for chunk in chunks if term in set(tokenize(chunk.text))][:5]
            if not related:
                continue
            title = term.replace("-", " ").title()
            slug = slugify(title)
            concept_id = stable_id("concept", slug, [chunk.chunk_id for chunk in related])
            markdown = self._render_concept(concept_id, title, slug, related)
            path = self.okf_dir / f"{slug}.md"
            path.write_text(markdown, encoding="utf-8")
            self.store.save_concept(
                concept_id=concept_id,
                title=title,
                slug=slug,
                text=markdown,
                source_chunk_ids=[chunk.chunk_id for chunk in related],
                verification_status="source-linked",
            )
            paths.append(path)
        return paths

    def _top_terms(self, chunks: list[Chunk], max_concepts: int) -> list[str]:
        counts: dict[str, int] = {}
        for chunk in chunks:
            for token in tokenize(chunk.text):
                if len(token) < 4 or token in STOPWORDS:
                    continue
                counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [term for term, _count in ranked[:max_concepts]]

    def _render_concept(self, concept_id: str, title: str, slug: str, chunks: list[Chunk]) -> str:
        sources = "\n".join(
            f"  - document_id: {chunk.document_id}\n    filename: {chunk.filename}\n"
            f"    pages: [{chunk.page_start}]"
            for chunk in chunks
        )
        source_list = "\n".join(
            f"- `{chunk.filename}`, page {chunk.page_start}, chunk `{chunk.chunk_id}`"
            for chunk in chunks
        )
        excerpts = "\n\n".join(
            f"### Source excerpt {index}\n\n{truncate_words(chunk.text, 90)}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return f"""---
id: {concept_id}
type: concept
title: {title}
slug: {slug}
verification_status: source-linked
source_documents:
{sources}
---

# {title}

This OKF concept is a source-linked retrieval aid. Verify final answers against the original PDF chunks.

## Source Excerpts

{excerpts}

## Sources

{source_list}
"""
