"""Vectorless RAG — hierarchical tree indexer.

During ingestion each PDF is parsed into a JSON tree of sections (title, page
range, LLM-generated summary, children). At query time TreeRetriever injects
this tree into llama3.2's context window and the LLM reasons through it to
select the exact nodes that contain the answer — no vector search required.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.ingestion.parser.pdf_parser import PageText

_HEADING_RE = re.compile(r"^(?:\d+[\.\d]*\s+)?[A-Z][A-Za-z0-9 \-,:\'\"]{0,90}$")

def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if line.endswith((".", "?", "!", ",", ";")):
        return False
    if not 1 <= len(line.split()) <= 14:
        return False
    return bool(_HEADING_RE.match(line))


@dataclass
class TreeNode:
    node_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    raw_text: str
    summary: str = ""
    children: list["TreeNode"] = field(default_factory=list)

    def to_dict(self, include_raw: bool = False) -> dict:
        d = {
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "summary": self.summary,
            "children": [c.to_dict(include_raw=include_raw) for c in self.children],
        }
        if include_raw:
            d["raw_text"] = self.raw_text
        return d

    def nav_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "pages": f"{self.page_start}-{self.page_end}",
            "summary": self.summary or self.title,
            "children": [c.nav_dict() for c in self.children],
        }


@dataclass
class DocumentTree:
    document_id: str
    filename: str
    nodes: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"document_id": self.document_id, "filename": self.filename,
                "nodes": [n.to_dict() for n in self.nodes]}

    def to_dict_with_raw(self) -> dict:
        return {"document_id": self.document_id, "filename": self.filename,
                "nodes": [n.to_dict(include_raw=True) for n in self.nodes]}

    def nav_json(self, max_nodes: int = 30) -> str:
        return json.dumps([n.nav_dict() for n in self.nodes[:max_nodes]], indent=2)

    def find_node(self, node_id: str) -> "TreeNode | None":
        def _search(nodes: list[TreeNode]) -> "TreeNode | None":
            for n in nodes:
                if n.node_id == node_id:
                    return n
                found = _search(n.children)
                if found:
                    return found
            return None
        return _search(self.nodes)

    def all_nodes(self) -> list[TreeNode]:
        result: list[TreeNode] = []
        def _collect(nodes: list[TreeNode]) -> None:
            for n in nodes:
                result.append(n)
                _collect(n.children)
        _collect(self.nodes)
        return result


class TreeIndexer:
    def __init__(self, trees_dir: Path, ollama_client=None) -> None:
        self.trees_dir = trees_dir
        self.trees_dir.mkdir(parents=True, exist_ok=True)
        self.ollama = ollama_client

    def build(self, document_id: str, filename: str, pages: list[PageText]) -> DocumentTree:
        nodes = self._segment_pages(pages)
        tree = DocumentTree(document_id=document_id, filename=filename, nodes=nodes)
        if self.ollama is not None:
            self._summarise_tree(tree)
        else:
            self._heuristic_summaries(tree)
        return tree

    def save(self, tree: DocumentTree) -> Path:
        path = self.trees_dir / f"{tree.document_id}.json"
        path.write_text(json.dumps(tree.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_with_raw(self, tree: DocumentTree) -> Path:
        path = self.trees_dir / f"{tree.document_id}_raw.json"
        path.write_text(json.dumps(tree.to_dict_with_raw(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, document_id: str) -> "DocumentTree | None":
        path = self.trees_dir / f"{document_id}.json"
        if not path.exists():
            return None
        return self._dict_to_tree(json.loads(path.read_text(encoding="utf-8")))

    def load_with_raw(self, document_id: str) -> "DocumentTree | None":
        path = self.trees_dir / f"{document_id}_raw.json"
        if not path.exists():
            return self.load(document_id)
        return self._dict_to_tree(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, document_id: str) -> bool:
        return (self.trees_dir / f"{document_id}.json").exists()

    def _segment_pages(self, pages: list[PageText]) -> list[TreeNode]:
        sections: list[tuple[str, int, list[str]]] = []
        current_title = "Introduction"
        current_page = pages[0].page_number if pages else 1
        current_lines: list[str] = []

        for page in pages:
            for line in page.text.splitlines():
                if _looks_like_heading(line):
                    if current_lines:
                        sections.append((current_title, current_page, current_lines))
                    current_title = line.strip()
                    current_page = page.page_number
                    current_lines = []
                else:
                    current_lines.append(line)
        if current_lines:
            sections.append((current_title, current_page, current_lines))

        if not sections:
            all_text = "\n".join(p.text for p in pages)
            title = pages[0].text.splitlines()[0][:60] if pages else "Document"
            return [TreeNode(
                node_id="N001", title=title, level=1,
                page_start=pages[0].page_number if pages else 1,
                page_end=pages[-1].page_number if pages else 1,
                raw_text=all_text,
            )]

        total_pages = pages[-1].page_number if pages else 1
        flat: list[TreeNode] = []
        for i, (title, page_start, lines) in enumerate(sections):
            page_end = sections[i + 1][1] - 1 if i + 1 < len(sections) else total_pages
            flat.append(TreeNode(
                node_id=f"N{i+1:03d}", title=title,
                level=self._detect_level(title),
                page_start=page_start,
                page_end=max(page_start, page_end),
                raw_text="\n".join(lines),
            ))
        return self._nest_by_level(flat)

    @staticmethod
    def _detect_level(title: str) -> int:
        if re.match(r"^\d+\.\d+\.\d+", title): return 3
        if re.match(r"^\d+\.\d+", title): return 2
        return 1

    @staticmethod
    def _nest_by_level(flat: list[TreeNode]) -> list[TreeNode]:
        root: list[TreeNode] = []
        stack: list[TreeNode] = []
        for node in flat:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                root.append(node)
            stack.append(node)
        return root

    def _summarise_tree(self, tree: DocumentTree) -> None:
        for node in tree.all_nodes():
            if node.summary:
                continue
            prompt = (
                f"Summarise this document section in 40-80 words. Be factual and concise.\n\n"
                f"Title: {node.title}\nText:\n{node.raw_text[:1200]}\n\nSummary:"
            )
            try:
                node.summary = self.ollama.generate(prompt).strip()[:400]
            except Exception:
                node.summary = self._heuristic_summary(node)

    @staticmethod
    def _heuristic_summaries(tree: DocumentTree) -> None:
        for node in tree.all_nodes():
            if not node.summary:
                node.summary = TreeIndexer._heuristic_summary(node)

    @staticmethod
    def _heuristic_summary(node: TreeNode) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", node.raw_text.strip())
        return " ".join(sentences[:2])[:300]

    @classmethod
    def _dict_to_tree(cls, data: dict) -> DocumentTree:
        return DocumentTree(
            document_id=data["document_id"], filename=data["filename"],
            nodes=[cls._dict_to_node(n) for n in data.get("nodes", [])],
        )

    @classmethod
    def _dict_to_node(cls, data: dict) -> TreeNode:
        return TreeNode(
            node_id=data["node_id"], title=data["title"], level=data["level"],
            page_start=data["page_start"], page_end=data["page_end"],
            raw_text=data.get("raw_text", ""), summary=data.get("summary", ""),
            children=[cls._dict_to_node(c) for c in data.get("children", [])],
        )
