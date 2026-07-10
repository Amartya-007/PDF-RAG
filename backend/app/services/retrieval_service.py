"""Service-layer vectorless retrieval orchestration."""
from __future__ import annotations

import re
from typing import Any

from backend.app.domain.enums import QueryType
from backend.app.domain.models.node import DocumentNode
from backend.app.retrieval.query_classifier import QueryClassifier


class RetrievalService:
    """Routes classified queries through the appropriate vectorless retriever."""

    def __init__(
        self,
        node_repo: Any,
        lexical: Any,
        navigator: Any,
        ranker: Any,
        classifier: QueryClassifier | None = None,
        reranker: Any | None = None,
    ) -> None:
        self._repo = node_repo
        self._lexical = lexical
        self._navigator = navigator
        self._ranker = ranker
        self._classifier = classifier or QueryClassifier()
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        include_debug: bool = False,
    ) -> tuple[list[DocumentNode], dict[str, Any]]:
        query_type = self._classifier.classify(query)
        session_node_ids = self._session_node_ids(session_id)

        if query_type == QueryType.FAST_FACT:
            nodes, hits, reasons = self._retrieve_fast_fact(
                query, session_id, session_node_ids, top_k
            )
        elif query_type == QueryType.TOPIC:
            nodes, hits, reasons = self._retrieve_topic(
                query, session_id, session_node_ids, top_k
            )
        elif query_type == QueryType.COMPARISON:
            nodes, hits, reasons = self._retrieve_comparison(
                query, session_id, session_node_ids, top_k
            )
        elif query_type == QueryType.SUMMARY:
            nodes, hits, reasons = self._retrieve_summary(
                query, session_id, session_node_ids, top_k
            )
        else:
            nodes, hits, reasons = self._retrieve_generic(
                query, session_id, session_node_ids, top_k
            )

        debug = (
            self._debug(query_type, nodes, hits, reasons)
            if include_debug
            else {}
        )
        return nodes[:top_k], debug

    def _retrieve_fast_fact(
        self,
        query: str,
        session_id: str,
        session_node_ids: set[str],
        top_k: int,
    ) -> tuple[list[DocumentNode], list[tuple[str, float]], dict[str, str]]:
        hits = self._search(query, session_id, session_node_ids, top_k * 4)
        nodes = self._nodes_from_hits(hits)
        ranked = self._ranker.rank_fast_fact(query, nodes, {node_id for node_id, _ in hits})
        return ranked[:top_k], hits, self._reasons(hits, "fast_fact_lexical")

    def _retrieve_topic(
        self,
        query: str,
        session_id: str,
        session_node_ids: set[str],
        top_k: int,
    ) -> tuple[list[DocumentNode], list[tuple[str, float]], dict[str, str]]:
        hits = self._search(query, session_id, session_node_ids, top_k * 4)
        matched = self._nodes_from_hits(hits)
        expanded = self._navigator.expand(matched, self._repo)
        candidates = self._merge_nodes(expanded, matched)
        ranked = self._ranker.rank_topic(query, candidates, {node_id for node_id, _ in hits})
        reasons = self._reasons(hits, "topic_lexical")
        reasons.update({node.id: "tree_expansion" for node in expanded})
        return ranked[:top_k], hits, reasons

    def _retrieve_comparison(
        self,
        query: str,
        session_id: str,
        session_node_ids: set[str],
        top_k: int,
    ) -> tuple[list[DocumentNode], list[tuple[str, float]], dict[str, str]]:
        all_hits: list[tuple[str, float]] = []
        reasons: dict[str, str] = {}
        for subquery in self._comparison_subqueries(query):
            hits = self._search(subquery, session_id, session_node_ids, top_k * 2)
            all_hits.extend(hits)
            reasons.update(self._reasons(hits, f"comparison:{subquery}"))
        nodes = self._nodes_from_hits(self._unique_hits(all_hits))
        ranked = self._rank_generic(query, nodes, set())
        return ranked[:top_k], self._unique_hits(all_hits), reasons

    def _retrieve_summary(
        self,
        query: str,
        session_id: str,
        session_node_ids: set[str],
        top_k: int,
    ) -> tuple[list[DocumentNode], list[tuple[str, float]], dict[str, str]]:
        hits = self._search(query, session_id, session_node_ids, top_k * 2)
        matched = self._nodes_from_hits(hits)
        expanded = self._navigator.expand(
            matched,
            self._repo,
            expand_depth=3,
            include_siblings=False,
        )
        ordered = sorted(
            self._merge_nodes(expanded, matched),
            key=lambda node: (node.document_id, node.page_start, node.position),
        )
        reasons = self._reasons(hits, "summary_heading")
        reasons.update({node.id: "ordered_tree_children" for node in expanded})
        return ordered[:top_k], hits, reasons

    def _retrieve_generic(
        self,
        query: str,
        session_id: str,
        session_node_ids: set[str],
        top_k: int,
    ) -> tuple[list[DocumentNode], list[tuple[str, float]], dict[str, str]]:
        hits = self._search(query, session_id, session_node_ids, top_k * 4)
        matched = self._nodes_from_hits(hits)
        expanded = self._navigator.expand(matched, self._repo)
        tree_ids = {node.id for node in expanded}
        candidates = self._merge_nodes(expanded, matched)
        ranked = self._rank_generic(query, candidates, tree_ids)
        reasons = self._reasons(hits, "lexical")
        reasons.update({node.id: "tree_expansion" for node in expanded})
        return ranked[:top_k], hits, reasons

    def _search(
        self,
        query: str,
        session_id: str,
        session_node_ids: set[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        return self._lexical.search(
            query,
            session_id=session_id,
            top_k=top_k,
            session_node_ids=session_node_ids,
        )

    def _rank_generic(
        self,
        query: str,
        candidates: list[DocumentNode],
        tree_ids: set[str],
    ) -> list[DocumentNode]:
        if self._reranker is not None:
            reranked = self._reranker.rerank(query, candidates, tree_ids)
            return self._coerce_ranked_nodes(reranked)
        ranked = self._ranker.rank(query, candidates, tree_ids, top_k=len(candidates))
        return self._coerce_ranked_nodes(ranked)

    @staticmethod
    def _coerce_ranked_nodes(ranked: list[Any]) -> list[DocumentNode]:
        nodes: list[DocumentNode] = []
        for item in ranked:
            node = getattr(item, "node", item)
            if isinstance(node, DocumentNode):
                nodes.append(node)
        return nodes

    def _nodes_from_hits(self, hits: list[tuple[str, float]]) -> list[DocumentNode]:
        return self._repo.get_many([node_id for node_id, _score in hits])

    @staticmethod
    def _merge_nodes(
        primary: list[DocumentNode],
        fallback: list[DocumentNode],
    ) -> list[DocumentNode]:
        merged: list[DocumentNode] = []
        seen: set[str] = set()
        for node in [*primary, *fallback]:
            if node.id in seen:
                continue
            seen.add(node.id)
            merged.append(node)
        return merged

    def _session_node_ids(self, session_id: str) -> set[str]:
        return {node.id for node in self._repo.list_nodes_for_session(session_id)}

    @staticmethod
    def _unique_hits(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for node_id, score in hits:
            best[node_id] = max(score, best.get(node_id, score))
        return list(best.items())

    @staticmethod
    def _reasons(hits: list[tuple[str, float]], reason: str) -> dict[str, str]:
        return {node_id: reason for node_id, _score in hits}

    @staticmethod
    def _comparison_subqueries(query: str) -> list[str]:
        direct = re.search(
            r"^(?:\s*(?:compare|comparison|difference(?:\s+between)?))?"
            r"\s*(?P<left>.+?)\s+(?:vs\.?|versus|and|with|to)\s+"
            r"(?P<right>.+?)\s*\??$",
            query,
            flags=re.IGNORECASE,
        )
        if direct:
            return [
                RetrievalService._clean_comparison_entity(direct.group("left")),
                RetrievalService._clean_comparison_entity(direct.group("right")),
            ]

        cleaned = re.sub(
            r"\b(compare|comparison|difference|between)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        parts = [
            part.strip(" ?.,:;")
            for part in re.split(r"\b(?:and|with|to)\b", cleaned, flags=re.IGNORECASE)
            if part.strip(" ?.,:;")
        ]
        return parts[:2] if len(parts) >= 2 else [query]

    @staticmethod
    def _clean_comparison_entity(value: str) -> str:
        return re.sub(
            r"\b(compare|comparison|difference|between)\b",
            " ",
            value,
            flags=re.IGNORECASE,
        ).strip(" ?.,:;")

    def _debug(
        self,
        query_type: QueryType,
        nodes: list[DocumentNode],
        hits: list[tuple[str, float]],
        reasons: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "query_type": query_type.value,
            "selected_node_ids": [node.id for node in nodes],
            "lexical_hits": hits,
            "score_breakdown": {
                node.id: {
                    **self._lexical.score_breakdown(node.id),
                    **self._ranker.score_details(node.id),
                }
                for node in nodes
            },
            "retrieval_reasons": {
                node.id: reasons.get(node.id, "ranked_candidate")
                for node in nodes
            },
        }
