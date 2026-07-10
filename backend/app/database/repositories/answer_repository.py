"""AnswerRepository — optional persistence for Answer and Citation records."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import Final

from backend.app.domain.models.answer import Answer, Citation

# --- Pre-allocated SQL Constants ---
# Prevents Python from allocating memory for these large strings on every function call
_UPSERT_ANSWER_SQL: Final = """
    insert into answers(answer_id, session_id, question, answer, answerable, debug)
    values(?, ?, ?, ?, ?, ?)
    on conflict(answer_id) do update set
        session_id=excluded.session_id,
        question=excluded.question,
        answer=excluded.answer,
        answerable=excluded.answerable,
        debug=excluded.debug
"""
_DELETE_CITATIONS_SQL: Final = "delete from answer_citations where answer_id = ?"
_INSERT_CITATIONS_SQL: Final = """
    insert into answer_citations(
        answer_id, source_id, document_id, filename,
        page_start, page_end, chunk_id, excerpt, position
    )
    values(?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
_SELECT_ANSWER_SQL: Final = "select * from answers where answer_id = ?"
_SELECT_CITATIONS_SQL: Final = """
    select * from answer_citations
    where answer_id = ?
    order by position asc
"""


class AnswerRepository:
    """Stores completed answers when callers opt into persistence."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def save_answer(self, answer_id: str, session_id: str, answer: Answer) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_ANSWER_SQL,
                (
                    answer_id,
                    session_id,
                    answer.question,
                    answer.answer,
                    int(answer.answerable),
                    json.dumps(answer.debug),
                ),
            )
            conn.execute(_DELETE_CITATIONS_SQL, (answer_id,))
            
            # Optimization: Generator expression inside executemany saves memory
            # compared to building a massive list in memory first.
            conn.executemany(
                _INSERT_CITATIONS_SQL,
                (
                    (
                        answer_id,
                        citation.source_id,
                        citation.document_id,
                        citation.filename,
                        citation.page_start,
                        citation.page_end,
                        citation.chunk_id,
                        citation.excerpt,
                        position,
                    )
                    for position, citation in enumerate(answer.citations)
                ),
            )

    def get_answer(self, answer_id: str) -> Answer | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT_ANSWER_SQL, (answer_id,)).fetchone()
            if row is None:
                return None
                
            citation_rows = conn.execute(_SELECT_CITATIONS_SQL, (answer_id,)).fetchall()
            
        return Answer(
            question=row["question"],
            answer=row["answer"],
            # Optimization: Map citations directly using list comprehension
            citations=[self._citation_from_row(c_row) for c_row in citation_rows],
            answerable=bool(row["answerable"]),
            debug=json.loads(row["debug"] or "{}"),
        )

    @staticmethod
    def _citation_from_row(row: sqlite3.Row) -> Citation:
        return Citation(
            source_id=row["source_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            chunk_id=row["chunk_id"],
            excerpt=row["excerpt"],
        )