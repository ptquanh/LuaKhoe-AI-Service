"""
CLI script to ingest .txt and .md files from a directory into pgvector.

Usage:
    python scripts/ingest_documents.py --source-dir data/knowledge_base/
    python scripts/ingest_documents.py --source-dir data/knowledge_base/ --dry-run
"""

import asyncio
import argparse
import os
import sys

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.database import init_db
from src.modules.advisory.repository import VectorStoreService
from src.modules.advisory.ingestion import IngestionPipeline
from src.shared.utils.logger import logger


SUPPORTED_EXTENSIONS = {".txt", ".md"}


async def run_ingestion(source_dir: str, dry_run: bool = False):
    if not os.path.isdir(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    # Collect files
    files: list[tuple[str, str]] = []
    for root, _, filenames in os.walk(source_dir):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append((os.path.join(root, fname), fname))

    if not files:
        logger.warning(f"No .txt or .md files found in {source_dir}")
        return

    logger.info(f"Found {len(files)} document(s) to ingest:")
    for fpath, fname in files:
        size_kb = os.path.getsize(fpath) / 1024
        logger.info(f"  → {fname} ({size_kb:.1f} KB)")

    if dry_run:
        logger.info("[DRY RUN] No documents were ingested.")
        return

    # Initialize DB and pipeline
    await init_db()
    vector_store = VectorStoreService()
    pipeline = IngestionPipeline(vector_store)

    total_chunks = 0
    for fpath, fname in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logger.warning(f"Skipping empty file: {fname}")
            continue

        count = await pipeline.ingest_text(
            text=content,
            source=fname,
            metadata={"file_path": fpath},
        )
        total_chunks += count

    logger.info(
        f"Ingestion complete: {total_chunks} chunks from {len(files)} file(s)"
    )

    # Report total count in DB
    db_count = await vector_store.get_chunk_count()
    logger.info(f"Total chunks in pgvector: {db_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest agronomic documents into pgvector"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing .txt and .md files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without ingesting",
    )
    args = parser.parse_args()

    asyncio.run(run_ingestion(args.source_dir, args.dry_run))


if __name__ == "__main__":
    main()
