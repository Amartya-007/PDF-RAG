from __future__ import annotations

from backend.app.core.hashing import stable_id
from backend.app.core.text import split_paragraphs, tokenize
from backend.app.models import Chunk, Document, PageText


class Chunker:
    def __init__(self, target_words: int = 360, overlap_words: int = 50) -> None:
        self.target_words = target_words
        self.overlap_words = overlap_words

    def chunk_pages(self, document: Document, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            paragraphs = split_paragraphs(page.text)
            buffer: list[str] = []
            word_count = 0
            sequence = 0

            for paragraph in paragraphs:
                paragraph_words = len(tokenize(paragraph))
                if buffer and word_count + paragraph_words > self.target_words:
                    chunks.append(self._make_chunk(document, page, buffer, sequence))
                    sequence += 1
                    buffer = self._overlap(buffer)
                    word_count = len(tokenize(" ".join(buffer)))

                buffer.append(paragraph)
                word_count += paragraph_words

            if buffer:
                chunks.append(self._make_chunk(document, page, buffer, sequence))

        return chunks

    def _make_chunk(
        self,
        document: Document,
        page: PageText,
        paragraphs: list[str],
        sequence: int,
    ) -> Chunk:
        text = "\n\n".join(paragraphs).strip()
        section_path = page.section_path or (document.filename,)
        chunk_id = stable_id("chunk", document.document_id, page.page_number, sequence, text)
        parent_id = stable_id("parent", document.document_id, page.page_number, tuple(section_path))
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            filename=document.filename,
            page_start=page.page_number,
            page_end=page.page_number,
            section_path=tuple(section_path),
            text=text,
            chunk_type="paragraph",
            parent_chunk_id=parent_id,
            metadata={"word_count": len(tokenize(text))},
        )

    def _overlap(self, paragraphs: list[str]) -> list[str]:
        if self.overlap_words <= 0:
            return []
        kept: list[str] = []
        total = 0
        for paragraph in reversed(paragraphs):
            words = len(tokenize(paragraph))
            if kept and total + words > self.overlap_words:
                break
            kept.insert(0, paragraph)
            total += words
        return kept
