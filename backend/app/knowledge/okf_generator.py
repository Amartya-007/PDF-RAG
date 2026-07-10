"""Builds a per-document knowledge map (OKF bundle).

Each PDF is turned into a small graph of concepts: multi-word phrases
that actually carry meaning, each backed by the source chunks it came from, 
and linked to other concepts whose source chunks genuinely overlap with it.
"""
from __future__ import annotations

import re
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

from backend.app.core.hashing import stable_id
from backend.app.core.text import tokenize, truncate_words
from backend.app.database.store import MetadataStore
from backend.app.knowledge.okf import render_simple_yaml
from backend.app.models import Chunk

# Pre-compiled regex for slug generation
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")

STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
    "were", "have", "has", "into", "shall", "will", "should", "would",
    "could", "been", "being", "their", "there", "where", "when", "what",
    "which", "while", "about", "above", "after", "again", "against",
    "also", "among", "before", "below", "between", "both", "during",
    "each", "either", "further", "here", "however", "more",
    "most", "neither", "none", "other", "over", "same", "some", "such",
    "than", "then", "these", "those", "through", "under", "until",
    "upon", "within", "without", "your", "you", "they", "them", "its",
    "our", "his", "her", "all", "any", "but", "not", "may", "can",
})


def slugify(text: str) -> str:
    """Safely converts text to a URL-friendly slug."""
    slug = SLUG_PATTERN.sub("-", text.lower()).strip("-")
    return slug or "concept"


class _PhraseCandidate:
    """Lightweight state container for tracking concept occurrences."""
    __slots__ = ("phrase", "phrase_tokens", "chunk_hits", "sentence_keys", "score")

    def __init__(self, phrase: str) -> None:
        self.phrase = phrase
        self.phrase_tokens = set(phrase.split())
        self.chunk_hits: dict[str, int] = {}  # chunk_id -> occurrences
        self.sentence_keys: set[tuple[str, int]] = set()  # (chunk_id, sentence_index)
        self.score = 0.0


class OkfGenerator:
    """Generates an interconnected Markdown Knowledge Graph from document chunks."""

    def __init__(self, okf_dir: Path, store: MetadataStore) -> None:
        self.okf_dir = okf_dir
        self.store = store
        self.okf_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_document(self, chunks: list[Chunk], max_concepts: int = 12) -> list[Path]:
        """Runs the full extraction, ranking, and rendering pipeline."""
        if not chunks:
            return []

        candidates = self._extract_phrase_candidates(chunks)
        top_candidates = self._rank_candidates(candidates, max_concepts)
        
        if not top_candidates:
            return []

        document_slug = self._document_slug(chunks)
        slugs = {
            c.phrase: f"{document_slug}-{slugify(c.phrase)}" for c in top_candidates
        }
        source_chunk_sets = {
            c.phrase: c.sentence_keys for c in top_candidates
        }

        paths: list[Path] = []
        for candidate in top_candidates:
            related_phrases = self._related_by_overlap(
                candidate.phrase, top_candidates, source_chunk_sets
            )
            related_chunks = self._top_chunks_for_candidate(candidate, chunks)
            
            if not related_chunks:
                continue

            path = self._write_concept(candidate, related_chunks, related_phrases, slugs)
            paths.append(path)

        self._write_indexes(paths)
        return paths

    def _write_concept(
        self,
        candidate: _PhraseCandidate,
        related_chunks: list[Chunk],
        related_phrases: list[str],
        slugs: dict[str, str],
    ) -> Path:
        """Handles IO and DB persistence for a single concept."""
        title = candidate.phrase.title()
        slug = slugs[candidate.phrase]
        concept_id = stable_id("concept", slug, [c.chunk_id for c in related_chunks])
        
        markdown = self._render_concept(
            concept_id=concept_id,
            title=title,
            slug=slug,
            chunks=related_chunks,
            aliases=[],
            tags=list(candidate.phrase_tokens),
            related=[slugs[item] for item in related_phrases],
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
            source_chunk_ids=[chunk.chunk_id for chunk in related_chunks],
            verification_status="source-linked",
            aliases=[],
            tags=list(candidate.phrase_tokens),
            related=[slugs[item] for item in related_phrases],
            depends_on=[],
            path=str(path),
        )
        return path

    # ------------------------------------------------------------------
    # Extraction & Ranking
    # ------------------------------------------------------------------

    def _extract_phrase_candidates(self, chunks: list[Chunk]) -> dict[str, _PhraseCandidate]:
        candidates: dict[str, _PhraseCandidate] = {}
        
        for chunk in chunks:
            # Replaced brittle regex with NLTK for robust sentence boundary detection
            sentences = sent_tokenize(chunk.text or "")
            
            for sentence_index, sentence in enumerate(sentences):
                tokens = tokenize(sentence)
                for n in (3, 2, 1):
                    for phrase in self._ngrams(tokens, n):
                        candidate = candidates.setdefault(phrase, _PhraseCandidate(phrase))
                        candidate.chunk_hits[chunk.chunk_id] = candidate.chunk_hits.get(chunk.chunk_id, 0) + 1
                        candidate.sentence_keys.add((chunk.chunk_id, sentence_index))
                        
        return candidates

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[str]:
        if len(tokens) < n:
            return []
            
        phrases: list[str] = []
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            
            if len(window[0]) < 3 or window[0] in STOPWORDS:
                continue
            if len(window[-1]) < 3 or window[-1] in STOPWORDS:
                continue
            if n >= 2 and any(tok.isdigit() for tok in window):
                continue
                
            phrases.append(" ".join(window))
        return phrases

    def _rank_candidates(
        self, candidates: dict[str, _PhraseCandidate], max_concepts: int
    ) -> list[_PhraseCandidate]:
        # Score based on distinct sentence appearances, weighted heavily towards 
        # longer, more specific multi-word phrases.
        for candidate in candidates.values():
            distinct_sentences = len(candidate.sentence_keys)
            length_bonus = 1.0 + 0.35 * (len(candidate.phrase_tokens) - 1)
            candidate.score = distinct_sentences * length_bonus

        ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
        selected: list[_PhraseCandidate] = []
        
        for candidate in ranked:
            if candidate.score <= 0 or not candidate.chunk_hits:
                continue
                
            # FIXED SUBSTRING BUG: Uses token subsetting instead of pure string matching.
            # Ensures "tax" doesn't falsely overlap with "syntax".
            is_redundant = any(
                candidate.phrase_tokens.issubset(kept.phrase_tokens) or 
                kept.phrase_tokens.issubset(candidate.phrase_tokens)
                for kept in selected
            )
            
            if is_redundant:
                continue
                
            selected.append(candidate)
            if len(selected) >= max_concepts:
                break
                
        return selected

    # ------------------------------------------------------------------
    # Relations & Rendering
    # ------------------------------------------------------------------

    def _related_by_overlap(
        self,
        phrase: str,
        candidates: list[_PhraseCandidate],
        source_chunk_sets: dict[str, set[tuple[str, int]]],
    ) -> list[str]:
        own_chunks = source_chunk_sets.get(phrase)
        if not own_chunks:
            return []
            
        scored: list[tuple[str, float]] = []
        for other in candidates:
            if other.phrase == phrase:
                continue
                
            other_chunks = source_chunk_sets.get(other.phrase)
            if not other_chunks:
                continue
                
            intersection = len(own_chunks & other_chunks)
            if not intersection:
                continue
                
            union = len(own_chunks | other_chunks)
            jaccard = intersection / union
            scored.append((other.phrase, jaccard))
            
        scored.sort(key=lambda item: item[1], reverse=True)
        return [p for p, _ in scored[:5]]

    def _top_chunks_for_candidate(self, candidate: _PhraseCandidate, chunks: list[Chunk]) -> list[Chunk]:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ranked_chunk_ids = sorted(
            candidate.chunk_hits.items(), key=lambda item: item[1], reverse=True
        )
        return [by_id[chunk_id] for chunk_id, _hits in ranked_chunk_ids[:5] if chunk_id in by_id]

    def _document_slug(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return "document"
        filename_slug = slugify(Path(chunks[0].filename or "doc").stem)
        document_suffix = slugify(chunks[0].document_id.replace("doc_", ""))[:8]
        return f"{filename_slug}-{document_suffix}" if document_suffix else filename_slug

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
            "source_chunk_ids": [c.chunk_id for c in chunks],
            "source_documents": [
                {"document_id": c.document_id, "filename": c.filename, "pages": [c.page_start]}
                for c in chunks
            ],
        }
        
        source_list = "\n".join(
            f"- `{c.filename}`, page {c.page_start}, chunk `{c.chunk_id}`" for c in chunks
        )
        relationship_links = "\n".join(
            f"- [{item.replace('-', ' ').title()}]({item}.md)" for item in related
        ) or "- No related concepts found in this document."
        
        excerpts = "\n\n".join(
            f"### Source excerpt {i}\n\n{truncate_words(c.text or '', 90)}"
            for i, c in enumerate(chunks, start=1)
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
        
        self.okf_dir.mkdir(parents=True, exist_ok=True)
        
        root_content = (
            "---\ntype: index\ntitle: Knowledge Index\n---\n\n"
            "# Knowledge Index\n\n- [Concepts](concepts/index.md)\n"
        )
        (self.okf_dir / "index.md").write_text(root_content, encoding="utf-8")

        concept_lines = ["---", "type: index", "title: Concepts", "---", "", "# Concepts", ""]
        concept_lines.extend(
            f"- [{slug.replace('-', ' ').title()}]({Path(rel_path).name})"
            for slug, rel_path in concept_entries
        )
        
        concepts_dir = self.okf_dir / "concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)
        (concepts_dir / "index.md").write_text("\n".join(concept_lines), encoding="utf-8")