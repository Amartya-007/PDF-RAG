"""Vectorless RAG — LLM-driven tree retriever.

At query time: injects the document tree's nav JSON (titles + summaries) into
the LLM. The LLM picks node IDs, we drill into children recursively, and then
fetch raw text of selected leaf nodes as Chunk objects. No vector search.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend.app.indexing.tree_indexer import DocumentTree, TreeNode
from backend.app.models import Chunk

logger = logging.getLogger(__name__)
MAX_DEPTH = 3


class TreeRetriever:
    """Uses an LLM to intelligently navigate a document's hierarchical tree."""

    def __init__(self, ollama_client: Any) -> None:
        self.ollama = ollama_client

    def retrieve(self, query: str, trees: list[DocumentTree], top_k: int = 12) -> list[Chunk]:
        """Navigates multiple document trees and returns the most relevant chunks.

        Args:
            query: The user's question.
            trees: List of parsed DocumentTree structures.
            top_k: Maximum number of final chunks to return.
            
        Returns:
            A deduplicated list of Chunk objects.
        """
        results: list[Chunk] = []
        
        for tree in trees:
            try:
                nodes = self._traverse(query, tree)
                results.extend(self._nodes_to_chunks(nodes, tree))
            except Exception as exc:
                logger.warning("Tree retrieval failed for %s: %s", tree.filename, exc)
                
        # Deduplicate while preserving relevance order and enforcing top_k
        seen: set[str] = set()
        deduped: list[Chunk] = []
        
        for chunk in results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduped.append(chunk)
                if len(deduped) >= top_k:
                    break
                    
        return deduped

    def _traverse(self, query: str, tree: DocumentTree) -> list[TreeNode]:
        """Drills down into the document tree up to MAX_DEPTH based on LLM routing."""
        selected: list[TreeNode] = []
        candidates = tree.nodes
        
        for depth in range(MAX_DEPTH):
            if not candidates:
                break
                
            node_ids = self._route(query, candidates)
            
            # If LLM explicitly returns nothing, stop traversing this branch
            if not node_ids:
                break
                
            # O(1) lookup set (calculated once, not inside the loop)
            allowed_ids = set(node_ids)
            chosen = [n for n in candidates if n.node_id in allowed_ids]
            
            if not chosen:
                break
                
            children = [child for n in chosen for child in n.children]
            
            # If we hit the bottom of the tree, or our max depth, we are done
            if not children or depth == MAX_DEPTH - 1:
                selected = chosen
                break
                
            candidates = children
            
        return selected

    def _route(self, query: str, nodes: list[TreeNode]) -> list[str]:
        """Asks the LLM to select the most relevant node IDs from the current level."""
        if not nodes:
            return []
            
        if self.ollama is None:
            # Fallback if no LLM is configured: explore everything
            return [n.node_id for n in nodes]
            
        # Minified JSON saves context window tokens
        nav = json.dumps([n.nav_dict() for n in nodes])
        
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
            # Regex extracts JSON even if LLM adds conversational filler like "Here is the JSON:"
            match = re.search(r'\{.*?"node_ids"\s*:\s*\[.*?\]\s*\}', raw, re.DOTALL)
            
            if match:
                data = json.loads(match.group())
                ids = data.get("node_ids", [])
                return [str(i) for i in ids] if isinstance(ids, list) else []
                
        except Exception as exc:
            logger.debug("LLM routing parsing failed: %s", exc)
            
        # Fallback safety net: if the LLM hallucinates or crashes, return all nodes
        return [n.node_id for n in nodes]

    @staticmethod
    def _nodes_to_chunks(nodes: list[TreeNode], tree: DocumentTree) -> list[Chunk]:
        """Converts raw tree nodes into standardized application Chunks."""
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
                metadata={
                    "node_id": node.node_id, 
                    "level": str(node.level), 
                    "summary": node.summary
                },
            ))
            
        return chunks