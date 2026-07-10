"""CLI entry point for local-pdf-rag service."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.rag_service import RagService


def _print_json(data: Any) -> None:
    """Standardized JSON printer."""
    print(json.dumps(data, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))


def main() -> None:
    """CLI orchestrator for ingestion, retrieval, and querying tasks."""
    parser = argparse.ArgumentParser(prog="local-pdf-rag")
    subs = parser.add_subparsers(dest="command", required=True)

    # Command definitions
    subs.add_parser("init")
    subs.add_parser("status")

    ingest = subs.add_parser("ingest")
    ingest.add_argument("path")
    ingest.add_argument("--no-okf", action="store_true")

    for cmd in ["import-okf", "validate-okf"]:
        subs.add_parser(cmd).add_argument("path")

    for cmd in ["retrieve", "ask"]:
        p = subs.add_parser(cmd)
        p.add_argument("question")
        p.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    service = RagService()

    try:
        if args.command == "init":
            service.init()
            print(f"Initialized data directory: {service.settings.data_dir}")

        elif args.command == "status":
            _print_json({
                "documents": len(service.store.list_documents()),
                "chunks": len(service.store.list_chunks())
            })

        elif args.command == "ingest":
            _print_json(service.ingest(Path(args.path), build_okf=not args.no_okf))

        elif args.command == "import-okf":
            concepts = service.import_okf_bundle(Path(args.path))
            _print_json({
                "imported_concepts": len(concepts),
                "concept_ids": [c.concept_id for c in concepts],
            })

        elif args.command == "validate-okf":
            _print_json({"issues": service.validate_okf_bundle(Path(args.path))})

        elif args.command == "retrieve":
            chunks, debug = service.retrieve(args.question, include_debug=args.debug)
            _print_json({
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "filename": c.filename,
                        "page_start": c.page_start,
                        "text": c.text[:500],
                    } for c in chunks
                ],
                "debug": debug,
            })

        elif args.command == "ask":
            answer = service.ask(args.question, include_debug=args.debug)
            _print_json({
                "answer": answer.answer,
                "answerable": answer.answerable,
                "citations": answer.citations,
                "debug": answer.debug,
            })

    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()