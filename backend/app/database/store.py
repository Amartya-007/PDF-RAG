from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.app.models import Chunk, Document


class MetadataStore:
    def __init__(self, path: Path | str) -> None:
        self.path = path
        self._memory_connection: sqlite3.Connection | None = None
        if path != ":memory:":
            assert isinstance(path, Path)
            path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        if self.path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:")
                self._memory_connection.row_factory = sqlite3.Row
            return self._memory_connection
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=OFF")
        conn.execute("pragma synchronous=NORMAL")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists documents (
                    document_id text primary key,
                    filename text not null,
                    sha256 text not null unique,
                    path text not null,
                    status text not null,
                    created_at text not null default current_timestamp
                );

                create table if not exists chunks (
                    chunk_id text primary key,
                    document_id text not null,
                    filename text not null,
                    page_start integer not null,
                    page_end integer not null,
                    section_path text not null,
                    text text not null,
                    chunk_type text not null,
                    parent_chunk_id text,
                    metadata text not null,
                    foreign key(document_id) references documents(document_id)
                );

                create table if not exists concepts (
                    concept_id text primary key,
                    title text not null,
                    slug text not null unique,
                    text text not null,
                    source_chunk_ids text not null,
                    verification_status text not null,
                    created_at text not null default current_timestamp
                );
                """
            )

    def upsert_document(self, document: Document) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into documents(document_id, filename, sha256, path, status)
                values(?, ?, ?, ?, ?)
                on conflict(document_id) do update set
                    filename=excluded.filename,
                    sha256=excluded.sha256,
                    path=excluded.path,
                    status=excluded.status
                """,
                (
                    document.document_id,
                    document.filename,
                    document.sha256,
                    document.path,
                    document.status,
                ),
            )

    def find_document_by_hash(self, sha256: str) -> Document | None:
        with self.connect() as conn:
            row = conn.execute("select * from documents where sha256 = ?", (sha256,)).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self) -> list[Document]:
        with self.connect() as conn:
            rows = conn.execute("select * from documents order by created_at desc").fetchall()
        return [self._document_from_row(row) for row in rows]

    def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        with self.connect() as conn:
            conn.execute("delete from chunks where document_id = ?", (document_id,))
            conn.executemany(
                """
                insert into chunks(
                    chunk_id, document_id, filename, page_start, page_end, section_path,
                    text, chunk_type, parent_chunk_id, metadata
                )
                values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.filename,
                        chunk.page_start,
                        chunk.page_end,
                        json.dumps(list(chunk.section_path)),
                        chunk.text,
                        chunk.chunk_type,
                        chunk.parent_chunk_id,
                        json.dumps(chunk.metadata),
                    )
                    for chunk in chunks
                ],
            )

    def list_chunks(self) -> list[Chunk]:
        with self.connect() as conn:
            rows = conn.execute("select * from chunks").fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        marks = ",".join("?" for _ in chunk_ids)
        with self.connect() as conn:
            rows = conn.execute(f"select * from chunks where chunk_id in ({marks})", chunk_ids).fetchall()
        by_id = {row["chunk_id"]: self._chunk_from_row(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def save_concept(
        self,
        concept_id: str,
        title: str,
        slug: str,
        text: str,
        source_chunk_ids: list[str],
        verification_status: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into concepts(concept_id, title, slug, text, source_chunk_ids, verification_status)
                values(?, ?, ?, ?, ?, ?)
                on conflict(concept_id) do update set
                    title=excluded.title,
                    slug=excluded.slug,
                    text=excluded.text,
                    source_chunk_ids=excluded.source_chunk_ids,
                    verification_status=excluded.verification_status
                """,
                (
                    concept_id,
                    title,
                    slug,
                    text,
                    json.dumps(source_chunk_ids),
                    verification_status,
                ),
            )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> Document:
        return Document(
            document_id=row["document_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            path=row["path"],
            status=row["status"],
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_path=tuple(json.loads(row["section_path"])),
            text=row["text"],
            chunk_type=row["chunk_type"],
            parent_chunk_id=row["parent_chunk_id"],
            metadata=json.loads(row["metadata"]),
        )
