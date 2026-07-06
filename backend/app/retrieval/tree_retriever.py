"""Vectorless RAG — LLM-driven tree retriever.

At query time: injects the document tree's nav JSON (titles + summaries) into
llama3.2, the LLM picks node IDs, we drill into children recursively, then
fetch raw text of selected leaf nodes as Chunk objects.  No vector search.
"""
from __future__ import annotations

import json
import logging
import re

from backend.app.indexing.tree_indexer import DocumentTree, TreeNode
from backend.app.models import Chunk

logger = logging.getLogger(__name__)
MAX_DEPTH = 3


class TreeRetriever:
    def __init__(self, ollama_client) -> None:
        self.ollama = ollama_client

    def retrieve(self, query: str, trees: list[DocumentTree]) -> list[Chunk]:
        results: list[Chunk] = []
        for tree in trees:
            try:
                nodes = self._traverse(query, tree)
                results.extend(self._nodes_to_chunks(nodes, tree))
            except Exception as exc:
                logger.warning("Tree retrieval failed for %s: %s", tree.filename, exc)
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for chunk in results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduped.append(chunk)
        return deduped[:12]

    def _traverse(self, query: str, tree: DocumentTree) -> list[TreeNode]:
        selected: list[TreeNode] = []
        candidates = tree.nodes
        for depth in range(MAX_DEPTH):
            if not candidates:
                break
            node_ids = self._route(query, candidates)
            chosen = [n for n in candidates if n.node_id in set(node_ids)] if node_ids else candidates
            if not chosen:
                selected = candidates
                break
            children = [child for n in chosen for child in n.children]
            if not children or depth == MAX_DEPTH - 1:
                selected = chosen
                break
            candidates = children
        return selected

    def _route(self, query: str, nodes: list[TreeNode]) -> list[str]:
        if not nodes:
            return []
        if self.ollama is None:
            return [n.node_id for n in nodes]
        nav = json.dumps([n.nav_dict() for n in nodes], indent=2)
        prompt = (
            "You are a precise document analyst. Given the document sections and a user query, "
            "identify the section IDs most likely to contain the answer.\n\n"
            f"Sections:\n{nav}\n\n"
            f"Query: \"{query}\"\n\n"
            "Respond ONLY with valid JSON: {\"node_ids\": [\"N001\", \"N002\"]}\n"
            "If nothing is relevant: {\"node_ids\": []}"
        )
        try:
            raw = self.ollama.generate(prompt).strip()
            m = re.search(r'\{.*?"node_ids"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                ids = data.get("node_ids", [])
                return [str(i) for i in ids] if isinstance(ids, list) else []
        except Exception as exc:
            logger.debug("LLM routing failed: %s", exc)
        return [n.node_id for n in nodes]

    @staticmethod
    def _nodes_to_chunks(nodes: list[TreeNode], tree: DocumentTree) -> list[Chunk]:
        chunks: list[Chunk] = []
        for node in nodes:
            text = node.raw_text.strip() or node.summary or node.title
            chunks.append(Chunk(
                chunk_id=f"tree:{tree.document_id}:{node.node_id}",
                document_id=tree.document_id,
                filename=tree.filename,
                page_start=node.page_start,
                page_end=node.page_end,
                section_path=(node.title,),
                text=text,
                chunk_type="tree_node",
                metadata={"node_id": node.node_id, "level": str(node.level), "summary": node.summary},
            ))
        return chunks
