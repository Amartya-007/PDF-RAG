from __future__ import annotations

import re
from pathlib import Path

from backend.app.core.hashing import stable_id
from backend.app.core.text import tokenize, truncate_words
from backend.app.database.store import MetadataStore
from backend.app.knowledge.okf import render_simple_yaml
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
        slugs = {term: slugify(term.replace("-", " ").title()) for term in terms}
        paths: list[Path] = []
        for term in terms:
            related = [chunk for chunk in chunks if term in set(tokenize(chunk.text))][:5]
            if not related:
                continue
            title = term.replace("-", " ").title()
            slug = slugs[term]
            concept_id = stable_id("concept", slug, [chunk.chunk_id for chunk in related])
            related_terms = self._related_terms(term, terms, related)
            markdown = self._render_concept(
                concept_id=concept_id,
                title=title,
                slug=slug,
                chunks=related,
                aliases=[],
                tags=[term],
                related=[slugs[item] for item in related_terms],
                depends_on=[],
            )
            path = self.okf_dir / "concepts" / f"{slug}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown, encoding="utf-8")
            self.store.save_concept(
                concept_id=concept_id,
                title=title,
                slug=slug,
                text=markdown,
                source_chunk_ids=[chunk.chunk_id for chunk in related],
                verification_status="source-linked",
                aliases=[],
                tags=[term],
                related=[slugs[item] for item in related_terms],
                depends_on=[],
                path=str(path),
            )
            paths.append(path)
        self._write_indexes(paths)
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

    def _related_terms(self, term: str, terms: list[str], chunks: list[Chunk]) -> list[str]:
        chunk_tokens = [set(tokenize(chunk.text)) for chunk in chunks]
        related: list[tuple[str, int]] = []
        for candidate in terms:
            if candidate == term:
                continue
            score = sum(1 for tokens in chunk_tokens if candidate in tokens)
            if score:
                related.append((candidate, score))
        related.sort(key=lambda item: item[1], reverse=True)
        return [candidate for candidate, _score in related[:5]]

    def _render_concept(
        self,
        concept_id: str,
        title: str,
        slug: str,
        chunks: list[Chunk],
        aliases: list[str],
        tags: list[str],
        related: list[str],
        depends_on: list[str],
    ) -> str:
        source_documents = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "pages": [chunk.page_start],
            }
            for chunk in chunks
        ]
        source_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
            for chunk in chunks
        ]
        metadata = {
            "id": concept_id,
            "type": "concept",
            "title": title,
            "slug": slug,
            "aliases": aliases,
            "tags": tags,
            "related": related,
            "depends_on": depends_on,
            "verification_status": "source-linked",
            "source_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "source_documents": source_documents,
            "source_chunks": source_chunks,
        }
        source_list = "\n".join(
            f"- `{chunk.filename}`, page {chunk.page_start}, chunk `{chunk.chunk_id}`"
            for chunk in chunks
        )
        relationship_links = "\n".join(
            f"- [{item.replace('-', ' ').title()}]({item}.md)" for item in related
        )
        if not relationship_links:
            relationship_links = "- No related concepts generated yet."
        excerpts = "\n\n".join(
            f"### Source excerpt {index}\n\n{truncate_words(chunk.text, 90)}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return f"""---
{render_simple_yaml(metadata)}
---

# {title}

This OKF concept is a source-linked retrieval aid. Verify final answers against the original PDF chunks.

## Relationships

{relationship_links}

## Source Excerpts

{excerpts}

## Sources

{source_list}
"""

    def _write_indexes(self, paths: list[Path]) -> None:
        concept_entries = sorted(
            (path.stem, path.relative_to(self.okf_dir).as_posix()) for path in paths
        )
        root_lines = [
            "---",
            "type: index",
            "title: Knowledge Index",
            "---",
            "",
            "# Knowledge Index",
            "",
            "- [Concepts](concepts/index.md)",
            "",
        ]
        self.okf_dir.mkdir(parents=True, exist_ok=True)
        (self.okf_dir / "index.md").write_text("\n".join(root_lines), encoding="utf-8")

        concept_lines = [
            "---",
            "type: index",
            "title: Concepts",
            "---",
            "",
            "# Concepts",
            "",
        ]
        for slug, relative_path in concept_entries:
            concept_lines.append(f"- [{slug.replace('-', ' ').title()}]({Path(relative_path).name})")
        concepts_dir = self.okf_dir / "concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)
        (concepts_dir / "index.md").write_text("\n".join(concept_lines), encoding="utf-8")
