from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.rag_service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(prog="local-pdf-rag")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("init")
    subcommands.add_parser("status")

    ingest = subcommands.add_parser("ingest")
    ingest.add_argument("path")
    ingest.add_argument("--no-okf", action="store_true")

    retrieve = subcommands.add_parser("retrieve")
    retrieve.add_argument("question")
    retrieve.add_argument("--debug", action="store_true")

    ask = subcommands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    service = RagService()

    if args.command == "init":
        service.init()
        print(f"Initialized data directory: {service.settings.data_dir}")
    elif args.command == "status":
        documents = service.store.list_documents()
        chunks = service.store.list_chunks()
        print(json.dumps({"documents": len(documents), "chunks": len(chunks)}, indent=2))
    elif args.command == "ingest":
        document = service.ingest(Path(args.path), build_okf=not args.no_okf)
        print(json.dumps(document.__dict__, indent=2))
    elif args.command == "retrieve":
        chunks, debug = service.retrieve(args.question, include_debug=args.debug)
        payload = {
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "filename": chunk.filename,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text[:500],
                }
                for chunk in chunks
            ],
            "debug": debug,
        }
        print(json.dumps(payload, indent=2))
    elif args.command == "ask":
        answer = service.ask(args.question, include_debug=args.debug)
        payload = {
            "answer": answer.answer,
            "answerable": answer.answerable,
            "citations": [citation.__dict__ for citation in answer.citations],
            "debug": answer.debug,
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
