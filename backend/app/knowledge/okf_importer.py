from __future__ import annotations

from pathlib import Path

from backend.app.database.store import MetadataStore
from backend.app.knowledge.okf import OkfConcept, import_okf_bundle


class OkfImporter:
    def __init__(self, okf_dir: Path, store: MetadataStore) -> None:
        self.okf_dir = okf_dir
        self.store = store
        self.okf_dir.mkdir(parents=True, exist_ok=True)

    def import_bundle(self, source_root: Path) -> list[OkfConcept]:
        concepts = import_okf_bundle(source_root, self.okf_dir)
        for concept in concepts:
            self.store.save_concept(
                concept_id=concept.concept_id,
                title=concept.title,
                slug=concept.slug,
                text=concept.text,
                source_chunk_ids=concept.source_chunk_ids,
                verification_status=concept.verification_status,
                aliases=concept.aliases,
                tags=concept.tags,
                related=concept.related,
                depends_on=concept.depends_on,
                path=concept.path,
            )
        return concepts
