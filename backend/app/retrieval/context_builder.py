from __future__ import annotations

from backend.app.core.text import truncate_words
from backend.app.models import Citation, Chunk


def build_evidence_block(chunks: list[Chunk], max_words_per_source: int = 90) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        excerpt = truncate_words(chunk.text, max_words_per_source)
        citations.append(
            Citation(
                source_id=source_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_id=chunk.chunk_id,
                excerpt=excerpt,
            )
        )
        page_label = (
            str(chunk.page_start)
            if chunk.page_start == chunk.page_end
            else f"{chunk.page_start}-{chunk.page_end}"
        )
        lines.extend(
            [
                f"SOURCE {source_id}",
                f"Document: {chunk.filename}",
                f"Pages: {page_label}",
                f"Chunk: {chunk.chunk_id}",
                f"Section: {' > '.join(chunk.section_path)}",
                "Text:",
                excerpt,
                "",
            ]
        )
    return "\n".join(lines), citations
