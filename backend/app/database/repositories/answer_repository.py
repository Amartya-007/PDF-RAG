"""AnswerRepository — optional persistence for Answer and Citation records."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

from backend.app.domain.models.answer import Answer, Citation


class AnswerRepository:
    """Stores completed answers when callers opt into persistence."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def save_answer(self, answer_id: str, session_id: str, answer: Answer) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into answers(answer_id, session_id, question, answer, answerable, debug)
                values(?, ?, ?, ?, ?, ?)
                on conflict(answer_id) do update set
                    session_id=excluded.session_id,
                    question=excluded.question,
                    answer=excluded.answer,
                    answerable=excluded.answerable,
                    debug=excluded.debug
                """,
                (
                    answer_id,
                    session_id,
                    answer.question,
                    answer.answer,
                    int(answer.answerable),
                    json.dumps(answer.debug),
                ),
            )
            conn.execute("delete from answer_citations where answer_id = ?", (answer_id,))
            conn.executemany(
                """
                insert into answer_citations(
                    answer_id, source_id, document_id, filename,
                    page_start, page_end, chunk_id, excerpt, position
                )
                values(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
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
                ],
            )

    def get_answer(self, answer_id: str) -> Answer | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from answers where answer_id = ?",
                (answer_id,),
            ).fetchone()
            if row is None:
                return None
            citation_rows = conn.execute(
                """
                select * from answer_citations
                where answer_id = ?
                order by position asc
                """,
                (answer_id,),
            ).fetchall()
        return Answer(
            question=row["question"],
            answer=row["answer"],
            citations=[self._citation_from_row(citation) for citation in citation_rows],
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
