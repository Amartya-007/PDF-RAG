"""TreeNavigator — LLM-driven traversal of the DocumentNode hierarchy.

Given a user query and a set of document trees (loaded from NodeRepository),
the navigator asks llama3.2 to identify which top-level sections are
relevant, then drills into their children recursively — like a human
scanning a table of contents before reading the relevant chapters.

Traversal protocol
------------------
1. Inject the top-level nodes of each document as a compact JSON nav map
   (node_id, title, summary of first 60 words, page range).
2. The LLM returns JSON ``{"node_ids": ["N001", "N003"]}``.
3. Repeat with the children of selected nodes (up to MAX_DEPTH levels).
4. Return the raw text of all selected leaf nodes as ``DocumentNode`` objects.

Fail-safes
----------
- Malformed / empty LLM JSON → fall back to returning all candidate nodes.
- Network / timeout error → log warning and return all candidates.
- Cancelled flag checked between every LLM call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from backend.app.domain.models.node import DocumentNode

logger = logging.getLogger(__name__)

MAX_DEPTH = 3
NAV_SUMMARY_WORDS = 60


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Minimal interface expected from OllamaClient."""
    def generate(self, prompt: str) -> str: ...


def _first_n_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")


class TreeNavigator:
    """LLM-driven recursive tree traversal over ``DocumentNode`` hierarchies.

    Args:
        generator: Any object with a ``generate(prompt) -> str`` method.
                   When ``None`` the navigator returns all top-level nodes
                   (safe offline fallback, equivalent to flat retrieval).
    """

    def __init__(self, generator: GeneratorProtocol | None) -> None:
        self._generator = generator

    # ── Public API ─────────────────────────────────────────────────────────

    def navigate(
        self,
        query: str,
        trees: dict[str, list[DocumentNode]],
        cancelled: list[bool] | None = None,
    ) -> list[DocumentNode]:
        """Return the most relevant ``DocumentNode`` objects for *query*.

        Args:
            query:     User question string.
            trees:     Mapping of document_id → flat list of DocumentNodes
                       (as returned by NodeRepository).
            cancelled: Optional mutable list; when ``cancelled[0]`` is True
                       the traversal stops immediately.

        Returns:
            Flat list of selected leaf DocumentNodes, deduplicated.
        """
        if not query.strip() or not trees:
            return []

        selected: list[DocumentNode] = []
        seen_ids: set[str] = set()

        for doc_id, nodes in trees.items():
            if cancelled and cancelled[0]:
                break
            doc_selected = self._traverse_document(query, nodes, cancelled)
            for node in doc_selected:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    selected.append(node)

        return selected

    # ── Per-document traversal ─────────────────────────────────────────────

    def _traverse_document(
        self,
        query: str,
        nodes: list[DocumentNode],
        cancelled: list[bool] | None,
    ) -> list[DocumentNode]:
        """Recursively navigate one document's node tree."""
        # Build parent→children map
        by_parent: dict[str | None, list[DocumentNode]] = {}
        by_id: dict[str, DocumentNode] = {}
        for node in nodes:
            by_parent.setdefault(node.parent_id, []).append(node)
            by_id[node.id] = node

        # Start from root-level nodes (parent_id is None)
        candidates = by_parent.get(None, [])
        return self._drill_down(query, candidates, by_parent, by_id, depth=0, cancelled=cancelled)

    def _drill_down(
        self,
        query: str,
        candidates: list[DocumentNode],
        by_parent: dict[str | None, list[DocumentNode]],
        by_id: dict[str, DocumentNode],
        depth: int,
        cancelled: list[bool] | None,
    ) -> list[DocumentNode]:
        if not candidates or depth >= MAX_DEPTH:
            return candidates

        if cancelled and cancelled[0]:
            return candidates

        selected_ids = self._route(query, candidates)
        chosen = [n for n in candidates if n.id in selected_ids] or candidates

        result: list[DocumentNode] = []
        for node in chosen:
            if cancelled and cancelled[0]:
                break
            children = by_parent.get(node.id, [])
            if children:
                result.extend(self._drill_down(
                    query, children, by_parent, by_id, depth + 1, cancelled
                ))
            else:
                result.append(node)

        return result

    def _route(self, query: str, candidates: list[DocumentNode]) -> set[str]:
        """Ask the LLM which candidate node IDs match the query."""
        if self._generator is None or not candidates:
            return {n.id for n in candidates}

        nav = json.dumps(
            [
                {
                    "node_id": n.id,
                    "title": n.title or "(untitled)",
                    "pages": f"{n.page_start}–{n.page_end}",
                    "summary": _first_n_words(n.text, NAV_SUMMARY_WORDS),
                }
                for n in candidates
            ],
            indent=2,
        )

        prompt = (
            "You are a precise document analyst.\n"
            "Given the document sections below and the user query, "
            "select the sections most likely to contain the answer.\n\n"
            f"Sections:\n{nav}\n\n"
            f"Query: \"{query}\"\n\n"
            "Respond ONLY with valid JSON — no explanation:\n"
            '{"node_ids": ["<id1>", "<id2>"]}\n'
            "If nothing is relevant: {\"node_ids\": []}"
        )

        try:
            raw = self._generator.generate(prompt).strip()
            match = re.search(r'\{.*?"node_ids"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                ids = data.get("node_ids", [])
                if isinstance(ids, list):
                    return set(str(i) for i in ids)
        except Exception as exc:
            logger.warning("TreeNavigator LLM routing failed: %s — returning all candidates", exc)

        return {n.id for n in candidates}
