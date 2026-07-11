"""FTS5Index — SQLite FTS5 full-text index over DocumentNode text and titles.

Primary lexical search mechanism for the vectorless RAG pipeline.
BM25 ranking is provided by SQLite's built-in FTS5 rank function.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable

from backend.app.domain.models.node import DocumentNode

logger = logging.getLogger(__name__)

# Pre-compiled regex for performance
_SANITIZE_RE = re.compile(r"[^\w\s\-]")

# Small stopword set for natural-language questions. FTS5's bare MATCH syntax
# ANDs every bareword term together, so passing a whole question through
# verbatim (e.g. "how many earned leave days can employees carry forward?")
# would require ALL of those words -- including "how"/"many"/"can" -- to be
# present in a node's text, which is almost never true. Stripping stopwords
# and OR-combining what's left lets BM25 ranking reward more matching terms
# instead of requiring all of them.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "into", "is", "it", "its", "of", "on", "or", "our", "that", "the",
    "their", "there", "these", "this", "those", "to", "was", "were", "what",
    "when", "where", "which", "who", "whom", "why", "will", "with", "you",
    "your", "many", "much", "tell", "me", "about", "please",
})


class FTS5Index:
    """Full-text search index backed by SQLite FTS5.

    Attributes:
        connection_factory: A callable that returns a sqlite3.Connection.
    """

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory
        self._consistent = True
        self._indexed_rowids: set[int] = set()
        self._fts5_available = self._probe_fts5()
        if not self._fts5_available:
            logger.warning("FTS5 missing; falling back to LIKE search.")

    def upsert(self, node: DocumentNode) -> None:
        """Insert or update a node in the FTS5 index."""
        try:
            with self._connect() as conn:
                rowid = self._node_rowid(conn, node.id)
                if rowid is None:
                    raise sqlite3.IntegrityError(f"Cannot index missing node: {node.id}")
                
                # Bulk delete-then-insert is safer for virtual tables
                if rowid in self._indexed_rowids:
                    self._delete_fts_row(conn, rowid, node.id, node.text, node.title or "")
                
                conn.execute(
                    "INSERT INTO nodes_fts(rowid, node_id, text, title) VALUES (?, ?, ?, ?)",
                    (rowid, node.id, node.text, node.title or ""),
                )
                self._indexed_rowids.add(rowid)
        except sqlite3.Error as exc:
            self._consistent = False
            logger.warning("FTS5 upsert failed: %s", exc)

    def delete(self, node_id: str) -> None:
        """Remove a node from the FTS5 index."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT rowid, text, title FROM nodes WHERE node_id = ?",
                    (node_id,),
                ).fetchone()
                if row and int(row["rowid"]) in self._indexed_rowids:
                    self._delete_fts_row(conn, row["rowid"], node_id, row["text"], row["title"] or "")
                    self._indexed_rowids.discard(int(row["rowid"]))
        except sqlite3.Error as exc:
            self._consistent = False
            logger.warning("FTS5 delete failed: %s", exc)

    def is_consistent(self) -> bool:
        """Whether the index is currently believed to be in sync with SQLite."""
        return self._consistent

    def rebuild(self, nodes: list[DocumentNode]) -> None:
        """Wipe and rebuild the index using batch operations.

        Only rowids we've actually tracked as indexed are cleared first. A
        bare ``DELETE FROM nodes_fts`` assumes every row in the external
        content table (``nodes``) is currently indexed; when that's not
        true (e.g. a node exists in ``nodes`` but was never added to this
        index), FTS5 raises a spurious "database disk image is malformed"
        error while reconciling its shadow tables against a content row it
        never indexed.
        """
        try:
            with self._connect() as conn:
                for rowid in list(self._indexed_rowids):
                    row = conn.execute(
                        "SELECT node_id, text, title FROM nodes WHERE rowid = ?",
                        (rowid,),
                    ).fetchone()
                    if row is not None:
                        self._delete_fts_row(
                            conn, rowid, row["node_id"], row["text"], row["title"] or ""
                        )
                self._indexed_rowids.clear()

                # Optimized: Use executemany to push iteration to C-layer
                data = [(self._node_rowid(conn, n.id), n.id, n.text, n.title or "") for n in nodes]
                conn.executemany(
                    "INSERT INTO nodes_fts(rowid, node_id, text, title) VALUES (?, ?, ?, ?)", 
                    data
                )
                self._indexed_rowids = {d[0] for d in data if d[0] is not None}
            self._consistent = True
        except sqlite3.Error as exc:
            self._consistent = False
            logger.error("FTS5 rebuild failed: %s", exc)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (node_id, score) pairs ranked by BM25."""
        if not query.strip():
            return []
        if self._fts5_available and self._consistent:
            return self._fts5_search(query, top_k)
        return self._like_search(query, top_k)

    def _fts5_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        safe_query = self._sanitise_fts5_query(query)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT node_id, -rank AS score FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
                    (safe_query, top_k),
                ).fetchall()
            return [(r["node_id"], float(r["score"])) for r in rows]
        except sqlite3.OperationalError:
            return self._like_search(query, top_k)

    def _like_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Fallback LIKE-based search."""
        terms = [t.strip() for t in query.split() if len(t.strip()) > 1]
        if not terms:
            return []
        
        conditions = " OR ".join("n.text LIKE ? OR n.title LIKE ?" for _ in terms)
        params = [f"%{t}%" for t in terms for _ in range(2)] + [top_k]
        
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT node_id, 1.0 AS score FROM nodes n WHERE {conditions} LIMIT ?",
                params,
            ).fetchall()
        return [(r["node_id"], float(r["score"])) for r in rows]

    def _probe_fts5(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT * FROM nodes_fts LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False

    @staticmethod
    def _node_rowid(conn: sqlite3.Connection, node_id: str) -> int | None:
        row = conn.execute("SELECT rowid FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return int(row["rowid"]) if row else None

    @staticmethod
    def _delete_fts_row(conn: sqlite3.Connection, rowid: int, node_id: str, text: str, title: str) -> None:
        conn.execute(
            "INSERT INTO nodes_fts(nodes_fts, rowid, node_id, text, title) VALUES('delete', ?, ?, ?, ?)",
            (rowid, node_id, text, title),
        )

    @staticmethod
    def _sanitise_fts5_query(query: str) -> str:
        # Pre-compiled regex used here for performance
        sanitised = query.replace('"', ' ').replace("'", " ").strip()
        cleaned = _SANITIZE_RE.sub(" ", sanitised).strip()
        if not cleaned:
            return '""'

        all_terms = cleaned.split()
        significant = [t for t in all_terms if t.lower() not in _STOPWORDS]
        terms = significant or all_terms
        if not terms:
            return '""'

        # OR-combine so a node matching *some* terms still surfaces; BM25
        # ranking (via `rank`) naturally favors nodes matching more terms.
        return " OR ".join(f'"{t}"' for t in terms)