from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from backend.app.knowledge.okf import OkfConcept
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

    def close(self) -> None:
        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None

    def __del__(self) -> None:
        self.close()

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
                    aliases text not null default '[]',
                    tags text not null default '[]',
                    related text not null default '[]',
                    depends_on text not null default '[]',
                    path text,
                    created_at text not null default current_timestamp
                );
                """
            )
            self._ensure_concept_columns(conn)

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
        aliases: list[str] | None = None,
        tags: list[str] | None = None,
        related: list[str] | None = None,
        depends_on: list[str] | None = None,
        path: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                insert into concepts(
                    concept_id, title, slug, text, source_chunk_ids, verification_status,
                    aliases, tags, related, depends_on, path
                )
                values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(concept_id) do update set
                    title=excluded.title,
                    slug=excluded.slug,
                    text=excluded.text,
                    source_chunk_ids=excluded.source_chunk_ids,
                    verification_status=excluded.verification_status,
                    aliases=excluded.aliases,
                    tags=excluded.tags,
                    related=excluded.related,
                    depends_on=excluded.depends_on,
                    path=excluded.path
                """,
                (
                    concept_id,
                    title,
                    slug,
                    text,
                    json.dumps(source_chunk_ids),
                    verification_status,
                    json.dumps(aliases or []),
                    json.dumps(tags or []),
                    json.dumps(related or []),
                    json.dumps(depends_on or []),
                    path,
                ),
            )

    def list_concepts(self) -> list[OkfConcept]:
        with self.connect() as conn:
            rows = conn.execute("select * from concepts order by title").fetchall()
        return [self._concept_from_row(row) for row in rows]

    def concepts_by_ids(self, concept_ids: list[str]) -> list[OkfConcept]:
        if not concept_ids:
            return []
        marks = ",".join("?" for _ in concept_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"select * from concepts where concept_id in ({marks})",
                concept_ids,
            ).fetchall()
        by_id = {row["concept_id"]: self._concept_from_row(row) for row in rows}
        return [by_id[concept_id] for concept_id in concept_ids if concept_id in by_id]

    def _ensure_concept_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in conn.execute("pragma table_info(concepts)").fetchall()
        }
        migrations = {
            "aliases": "alter table concepts add column aliases text not null default '[]'",
            "tags": "alter table concepts add column tags text not null default '[]'",
            "related": "alter table concepts add column related text not null default '[]'",
            "depends_on": "alter table concepts add column depends_on text not null default '[]'",
            "path": "alter table concepts add column path text",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)

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

    @staticmethod
    def _concept_from_row(row: sqlite3.Row) -> OkfConcept:
        return OkfConcept(
            concept_id=row["concept_id"],
            title=row["title"],
            slug=row["slug"],
            text=row["text"],
            source_chunk_ids=json.loads(row["source_chunk_ids"]),
            verification_status=row["verification_status"],
            aliases=json.loads(row["aliases"] or "[]"),
            tags=json.loads(row["tags"] or "[]"),
            related=json.loads(row["related"] or "[]"),
            depends_on=json.loads(row["depends_on"] or "[]"),
            path=row["path"],
        )
