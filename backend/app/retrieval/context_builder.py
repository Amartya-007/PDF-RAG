"""Evidence construction for LLM generation.

Transforms raw retrieved chunks into a standardized, token-efficient 
string format that the LLM can use for grounded question answering, 
while simultaneously building the citation mapping for the UI.
"""
from __future__ import annotations

from backend.app.core.text import truncate_words
from backend.app.models import Citation, Chunk


def build_evidence_block(
    chunks: list[Chunk], max_words_per_source: int = 90
) -> tuple[str, list[Citation]]:
    """Formats retrieved chunks into an LLM-readable string and generates citations.

    Args:
        chunks: The deduplicated, reranked list of document chunks.
        max_words_per_source: Hard limit on chunk length to prevent context window bloat.

    Returns:
        A tuple containing:
        1. A formatted string of all evidence (for the LLM prompt).
        2. A list of Citation objects (to return to the frontend UI).
    """
    if not chunks:
        return "", []

    lines: list[str] = []
    citations: list[Citation] = []
    
    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        
        # Safely handle potentially missing text
        text = chunk.text or ""
        excerpt = truncate_words(text, max_words_per_source)
        
        citations.append(
            Citation(
                source_id=source_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_id=chunk.chunk_id,
                excerpt=excerpt,
                heading_path=list(chunk.section_path),
            )
        )
        
        # Safely handle missing or identical page numbers
        if chunk.page_start and chunk.page_end and chunk.page_start != chunk.page_end:
            page_label = f"{chunk.page_start}-{chunk.page_end}"
        else:
            page_label = str(chunk.page_start or "Unknown")

        # Safely handle missing section paths
        section_label = " > ".join(chunk.section_path) if chunk.section_path else "Unknown"

        # Build the block (using extend for performance)
        lines.extend(
            [
                f"SOURCE {source_id}",
                f"Document: {chunk.filename or 'Unknown'}",
                f"Pages: {page_label}",
                f"Chunk: {chunk.chunk_id}",
                f"Section: {section_label}",
                "Text:",
                excerpt,
                "",  # Blank line separator
            ]
        )
        
    return "\n".join(lines), citations