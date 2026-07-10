from __future__ import annotations

from backend.app.models import Citation
from backend.app.verification.citation_validator import CitationValidator


def make_citation(source_id: str = "S1", excerpt: str | None = None) -> Citation:
    return Citation(
        source_id=source_id,
        document_id="doc1",
        filename="source.pdf",
        page_start=1,
        page_end=1,
        chunk_id="node1",
        excerpt=excerpt or "Revenue increased by 12% in 2024 after refunds settled.",
    )


def test_validate_returns_false_when_citations_are_empty() -> None:
    assert CitationValidator().validate("Revenue increased by 12% [S1].", []) is False


def test_validate_requires_answer_to_reference_known_source() -> None:
    citation = make_citation("S1")

    assert CitationValidator().validate("Revenue increased by 12%.", [citation]) is False
    assert CitationValidator().validate("Revenue increased by 12% [S9].", [citation]) is False


def test_validate_accepts_supported_cited_claim() -> None:
    citation = make_citation()

    assert (
        CitationValidator().validate("Revenue increased by 12% in 2024 [S1].", [citation])
        is True
    )


def test_validate_rejects_unsupported_numbers_and_names() -> None:
    citation = make_citation(
        excerpt="Asha Patel earned a CGPA of 8.7 at Northbridge University."
    )
    validator = CitationValidator()

    assert validator.validate("Asha Patel earned a CGPA of 9.1 [S1].", [citation]) is False
    assert validator.validate("Rohan Mehta earned a CGPA of 8.7 [S1].", [citation]) is False


def test_validate_rejects_directional_claim_not_supported_by_excerpt() -> None:
    citation = make_citation(excerpt="Revenue decreased by 12% in 2024.")

    assert CitationValidator().validate("Revenue increased by 12% in 2024 [S1].", [citation]) is False
