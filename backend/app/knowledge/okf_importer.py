"""Orchestrates the importation of OKF bundles into the system.

Handles the file-system operations and bulk database ingestion
required to register new knowledge graph concepts.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.app.database.store import MetadataStore
from backend.app.knowledge.okf import OkfConcept, import_okf_bundle

logger = logging.getLogger(__name__)


class OkfImporter:
    """Manages the ingestion of OKF concept bundles into the storage layer."""

    def __init__(self, okf_dir: Path, store: MetadataStore) -> None:
        self.okf_dir = okf_dir
        self.store = store
        # Ensure the destination directory exists immediately upon instantiation
        self.okf_dir.mkdir(parents=True, exist_ok=True)

    def import_bundle(self, source_root: Path) -> list[OkfConcept]:
        """Validates, copies, and persists an OKF bundle to the database.

        Args:
            source_root: The directory containing the OKF markdown files.

        Returns:
            A list of the successfully imported OkfConcept objects.
        """
        logger.info("Starting OKF bundle import from: %s", source_root)
        
        # 1. Parse and copy files to the internal OKF directory
        concepts = import_okf_bundle(source_root, self.okf_dir)
        
        if not concepts:
            logger.warning("No valid concepts found in bundle: %s", source_root)
            return []

        # 2. Bulk Database Insert
        # If your MetadataStore has a `save_concepts` or `insert_many` method, USE IT HERE.
        # Example: self.store.save_concepts_bulk(concepts)
        
        # Fallback (If you haven't implemented bulk inserts in MetadataStore yet, 
        # doing it in a loop is okay, but ideally wrap it in a single transaction).
        try:
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
            logger.info("Successfully imported %d concepts.", len(concepts))
            
        except Exception as e:
            logger.error("Database error during OKF import: %s", e)
            raise
            
        return concepts