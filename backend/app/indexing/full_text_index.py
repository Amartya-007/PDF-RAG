"""FTS5Index — SQLite FTS5 full-text index over DocumentNode text and titles.

Primary lexical search mechanism for the vectorless RAG pipeline.
BM25 ranking is provided by SQLite's built-in FTS5 rank function,
which is present in all CPython distributions since 3.x.

Consistency contract
--------------------
If any upsert fails (e.g. because a transaction was rolled back externally),
the index marks itself *inconsistent*.  Subsequent ``search`` calls fall back
to LIKE-based text matching until ``rebuild()`` clears the flag.
``IndexManager.rebuild_all()`` is the canonical way to restore consistency.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable

from backend.app.domain.models.node import DocumentNode

logger = logging.getLogger(__name__)

_FTS5_PROBE = "SELECT fts5(?)"   # will fail if FTS5 is absent


class FTS5Index:
    """Full-text search index backed by SQLite FTS5.

    Args:
        connection_factory: Returns a :class:`sqlite3.Connection`.
            The factory is called on every operation so the index can
            share a connection pool with the rest of the store layer.
    """

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory
        self._consistent = True
        self._fts5_available = self._probe_fts5()
        if not self._fts5_available:
            logger.warning(
                "FTS5 is not available in this SQLite build; "
                "falling back to LIKE-based text search.  "
                "Upgrade to a standard CPython distribution for full BM25 support."
            )

    # ── Write operations ───────────────────────────────────────────────────

    def upsert(self, node: DocumentNode) -> None:
        """Insert or update a node in the FTS5 index."""
        try:
            with self._connect() as conn:
                # FTS5 content= tables require explicit delete + insert for updates
                conn.execute(
                    "DELETE FROM nodes_fts WHERE node_id = ?", (node.id,)
                )
                conn.execute(
                    "INSERT INTO nodes_fts(node_id, text, title) VALUES (?, ?, ?)",
                    (node.id, node.text, node.title or ""),
                )
        except sqlite3.Error as exc:
            self._consistent = False
            logger.warning(
                "FTS5 upsert failed for node %s — marking index inconsistent: %s",
                node.id, exc,
            )

    def delete(self, node_id: str) -> None:
        """Remove a node from the FTS5 index."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM nodes_fts WHERE node_id = ?", (node_id,)
                )
        except sqlite3.Error as exc:
            self._consistent = False
            logger.warning("FTS5 delete failed for node %s: %s", node_id, exc)

    def rebuild(self, nodes: list[DocumentNode]) -> None:
        """Wipe and rebuild the entire FTS5 index from *nodes*."""
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM nodes_fts")
                conn.executemany(
                    "INSERT INTO nodes_fts(node_id, text, title) VALUES (?, ?, ?)",
                    [(n.id, n.text, n.title or "") for n in nodes],
                )
            self._consistent = True
        except sqlite3.Error as exc:
            self._consistent = False
            logger.error("FTS5 rebuild failed: %s", exc)

    # ── Read operations ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(node_id, score)`` pairs ranked by BM25.

        Falls back to LIKE matching when FTS5 is unavailable or the index
        is inconsistent.  Results are sorted by descending score.
        """
        if not query.strip():
            return []
        if self._fts5_available and self._consistent:
            return self._fts5_search(query, top_k)
        return self._like_search(query, top_k)

    def is_consistent(self) -> bool:
        return self._consistent

    # ── Private helpers ────────────────────────────────────────────────────

    def _fts5_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        safe_query = self._sanitise_fts5_query(query)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT node_id, -rank AS score
                    FROM nodes_fts
                    WHERE nodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, top_k),
                ).fetchall()
            return [(r["node_id"], float(r["score"])) for r in rows]
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed (%s) — falling back to LIKE", exc)
            return self._like_search(query, top_k)

    def _like_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """LIKE-based fallback — much slower, no BM25, but always works."""
        terms = [t.strip() for t in query.split() if len(t.strip()) > 1]
        if not terms:
            return []
        conditions = " OR ".join(
            "n.text LIKE ? OR n.title LIKE ?" for _ in terms
        )
        params = [f"%{t}%", f"%{t}%"] + [f"%{t}%", f"%{t}%"] * (len(terms) - 1)
        # Flatten: each term needs two params (text + title)
        flat_params = []
        for t in terms:
            flat_params.extend([f"%{t}%", f"%{t}%"])
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT n.node_id, 1.0 AS score
                    FROM nodes n
                    WHERE {conditions}
                    LIMIT ?
                    """,
                    flat_params + [top_k],
                ).fetchall()
            return [(r["node_id"], float(r["score"])) for r in rows]
        except sqlite3.Error as exc:
            logger.error("LIKE fallback search failed: %s", exc)
            return []

    def _probe_fts5(self) -> bool:
        """Check whether the SQLite build includes FTS5 support."""
        try:
            with self._connect() as conn:
                conn.execute("SELECT * FROM nodes_fts LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False

    @staticmethod
    def _sanitise_fts5_query(query: str) -> str:
        """Escape characters that would produce an FTS5 syntax error."""
        # Remove FTS5 special chars; wrap in double-quotes for phrase safety
        sanitised = query.replace('"', ' ').replace("'", " ").strip()
        # Keep only alphanumeric, spaces, and hyphens
        import re
        sanitised = re.sub(r"[^\w\s\-]", " ", sanitised)
        return sanitised.strip() or '""'
