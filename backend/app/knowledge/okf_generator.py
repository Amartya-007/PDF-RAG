from __future__ import annotations

import re
from pathlib import Path

from backend.app.core.hashing import stable_id
from backend.app.core.text import tokenize, truncate_words
from backend.app.database.store import MetadataStore
from backend.app.knowledge.okf import render_simple_yaml
from backend.app.models import Chunk

STOPWORDS = {
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
}


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "concept"


class _PhraseCandidate:
    __slots__ = ("phrase", "chunk_hits", "sentence_keys", "score")

    def __init__(self, phrase: str) -> None:
        self.phrase = phrase
        self.chunk_hits: dict[str, int] = {}  # chunk_id -> occurrences in that chunk
        self.sentence_keys: set[tuple[str, int]] = set()  # (chunk_id, sentence_index)
        self.score = 0.0


class OkfGenerator:
    """Builds a per-document knowledge map (OKF bundle).

    Each PDF is turned into a small graph of concepts: multi-word phrases
    that actually carry meaning (not just the most frequent single words),
    each backed by the source chunks it came from, and linked to other
    concepts whose source chunks genuinely overlap with it in the document.
    That overlap is what gives the model real "this relates to that"
    context instead of a list of disconnected keywords.
    """

    def __init__(self, okf_dir: Path, store: MetadataStore) -> None:
        self.okf_dir = okf_dir
        self.store = store
        self.okf_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_document(self, chunks: list[Chunk], max_concepts: int = 12) -> list[Path]:
        if not chunks:
            return []

        candidates = self._extract_phrase_candidates(chunks)
        top_candidates = self._rank_candidates(candidates, max_concepts)
        if not top_candidates:
            return []

        document_slug = self._document_slug(chunks)
        slugs = {
            candidate.phrase: f"{document_slug}-{slugify(candidate.phrase)}"
            for candidate in top_candidates
        }
        source_chunk_sets = {
            candidate.phrase: candidate.sentence_keys
            for candidate in top_candidates
        }

        paths: list[Path] = []
        for candidate in top_candidates:
            related_phrases = self._related_by_overlap(candidate.phrase, top_candidates, source_chunk_sets)
            related_chunks = self._top_chunks_for_candidate(candidate, chunks)
            if not related_chunks:
                continue

            title = candidate.phrase.title()
            slug = slugs[candidate.phrase]
            concept_id = stable_id("concept", slug, [chunk.chunk_id for chunk in related_chunks])
            markdown = self._render_concept(
                concept_id=concept_id,
                title=title,
                slug=slug,
                chunks=related_chunks,
                aliases=[],
                tags=self._tags_for_candidate(candidate),
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
                tags=self._tags_for_candidate(candidate),
                related=[slugs[item] for item in related_phrases],
                depends_on=[],
                path=str(path),
            )
            paths.append(path)
        self._write_indexes(paths)
        return paths

    # ------------------------------------------------------------------
    # Phrase extraction: 1-3 word n-grams instead of single tokens.
    # ------------------------------------------------------------------

    def _extract_phrase_candidates(self, chunks: list[Chunk]) -> dict[str, _PhraseCandidate]:
        candidates: dict[str, _PhraseCandidate] = {}
        for chunk in chunks:
            sentences = self._split_sentences(chunk.text)
            for sentence_index, sentence in enumerate(sentences):
                tokens = tokenize(sentence)
                for n in (3, 2, 1):
                    for phrase in self._ngrams(tokens, n):
                        candidate = candidates.setdefault(phrase, _PhraseCandidate(phrase))
                        candidate.chunk_hits[chunk.chunk_id] = candidate.chunk_hits.get(chunk.chunk_id, 0) + 1
                        candidate.sentence_keys.add((chunk.chunk_id, sentence_index))
        return candidates

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Bounding n-grams to a sentence (rather than a whole 300+ word
        # chunk) keeps phrases coherent ("annual leave requests" instead of
        # garbage spanning two unrelated clauses) and, more importantly,
        # gives us a much finer unit than chunk_id to measure real overlap
        # between concepts - without this, a short document with only one
        # or two chunks makes every phrase appear to "co-occur" with every
        # other phrase simply because they share the same chunk.
        return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[str]:
        if len(tokens) < n:
            return []
        phrases: list[str] = []
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            # Don't let a phrase start or end on a stopword/short token -
            # avoids junk like "of the system" or "system and the".
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
        # Score = (number of distinct chunks it appears in) weighted up for
        # longer phrases, since a 2-3 word phrase repeating across multiple
        # chunks is a much stronger signal of being an actual concept than a
        # single common word repeating a lot in one place.
        for candidate in candidates.values():
            distinct_sentences = len(candidate.sentence_keys)
            length_bonus = 1.0 + 0.35 * (len(candidate.phrase.split()) - 1)
            candidate.score = distinct_sentences * length_bonus

        ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)

        # De-duplicate near-identical/overlapping candidates so we don't end
        # up with both "leave policy" and "leave policy applies" as separate
        # concepts. Greedily keep a candidate only if it isn't a substring of
        # (or doesn't fully contain) an already-selected, higher-scored one.
        selected: list[_PhraseCandidate] = []
        for candidate in ranked:
            if candidate.score <= 0:
                continue
            if len(candidate.chunk_hits) < 1:
                continue
            if any(
                candidate.phrase in kept.phrase or kept.phrase in candidate.phrase
                for kept in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= max_concepts:
                break
        return selected

    def _tags_for_candidate(self, candidate: _PhraseCandidate) -> list[str]:
        return candidate.phrase.split()

    # ------------------------------------------------------------------
    # Relations: real overlap between concepts' source chunks, not
    # incidental co-occurrence inside an already-filtered subset.
    # ------------------------------------------------------------------

    def _related_by_overlap(
        self,
        phrase: str,
        candidates: list[_PhraseCandidate],
        source_chunk_sets: dict[str, set[tuple[str, int]]],
    ) -> list[str]:
        own_chunks = source_chunk_sets[phrase]
        if not own_chunks:
            return []
        scored: list[tuple[str, float]] = []
        for other in candidates:
            if other.phrase == phrase:
                continue
            other_chunks = source_chunk_sets[other.phrase]
            if not other_chunks:
                continue
            intersection = len(own_chunks & other_chunks)
            if intersection == 0:
                continue
            union = len(own_chunks | other_chunks)
            jaccard = intersection / union if union else 0.0
            scored.append((other.phrase, jaccard))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [phrase for phrase, _score in scored[:5]]

    def _top_chunks_for_candidate(self, candidate: _PhraseCandidate, chunks: list[Chunk]) -> list[Chunk]:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ranked_chunk_ids = sorted(
            candidate.chunk_hits.items(), key=lambda item: item[1], reverse=True
        )
        return [by_id[chunk_id] for chunk_id, _hits in ranked_chunk_ids[:5] if chunk_id in by_id]

    # ------------------------------------------------------------------

    def _document_slug(self, chunks: list[Chunk]) -> str:
        if not chunks:
            return "document"
        filename_slug = slugify(Path(chunks[0].filename).stem)
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
            relationship_links = "- No related concepts found in this document."
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
