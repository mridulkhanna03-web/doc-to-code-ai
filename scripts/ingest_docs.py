"""End-to-end ingestion runner — the Day 1 milestone.

Pipeline:

1. Walk the FastAPI docs clone at ``data/raw_docs/fastapi/docs/en/docs``.
2. Fetch → parse → clean → chunk every ``*.md`` file
   (via :mod:`src.ingestion.pipeline`).
3. Persist the chunked corpus as JSON (``data/processed/chunks.json``).
4. Generate L2-normalised embeddings via Ollama (``nomic-embed-text``).
5. Upsert into the persistent ChromaDB collection
   (``data/chroma_storage/`` — collection
   ``fastapi_docs_embeddings``).

CLI:

    python scripts/ingest_docs.py                # incremental upsert (default)
    python scripts/ingest_docs.py --reset        # drop the collection first
    python scripts/ingest_docs.py --skip-embed   # only run parse/chunk/save

Idempotent: re-running without ``--reset`` upserts the same ids (no
duplicates). Embedding generation is the expensive step (~minutes on CPU,
seconds on GPU); avoid re-running unless the chunks have changed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow ``python scripts/ingest_docs.py`` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.embeddings.embedding_generator import EmbeddingError, generate_embeddings  # noqa: E402
from src.ingestion.pipeline import run_ingestion, save_processed  # noqa: E402
from src.preprocessing.chunker import embedding_text  # noqa: E402
from src.retrieval.vector_store import (  # noqa: E402
    VectorStoreError,
    add_documents,
    init_collection,
    reset_collection,
)
from src.utils.logger import setup_logger  # noqa: E402

log = setup_logger("scripts.ingest_docs")

DEFAULT_SOURCE: Path = settings.RAW_DOCS_DIR / "fastapi" / "docs" / "en" / "docs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE,
        help=f"root directory to scan for *.md (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--pattern", default="*.md",
        help="glob pattern under --source (default: *.md)",
    )
    parser.add_argument(
        "--collection", default=settings.CHROMA_COLLECTION,
        help=f"ChromaDB collection name (default: {settings.CHROMA_COLLECTION})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="drop the collection before ingesting (rebuild from scratch)",
    )
    parser.add_argument(
        "--skip-embed", action="store_true",
        help="parse + chunk + save JSON only; skip embedding generation and ChromaDB write",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.source.exists():
        log.error(
            f"source not found: {args.source} — "
            f"run `python scripts/download_docs.py` first."
        )
        return 2

    # -------------------------------------------------------------- chunk stage
    t0 = time.time()
    log.info(f"--- ingestion start: source={args.source} ---")
    chunks = run_ingestion(args.source, pattern=args.pattern)
    if not chunks:
        log.error("no chunks produced — aborting")
        return 1

    try:
        save_processed(chunks)
    except OSError as exc:
        log.error(f"persist chunks failed: {exc}")
        return 1
    log.info(f"chunking stage done in {time.time() - t0:.1f}s ({len(chunks)} chunks)")

    if args.skip_embed:
        log.info("--skip-embed set; stopping before embedding/Chroma write")
        return 0

    # ---------------------------------------------------------- embedding stage
    # We embed a doc/section-prefixed version of each chunk (via embedding_text)
    # so metadata keywords are present in the vector. The chunk's stored text
    # stays original so the LLM sees clean documentation content.
    t1 = time.time()
    log.info(f"--- embedding {len(chunks)} chunks via Ollama '{settings.EMBEDDING_MODEL}' ---")
    try:
        embeddings = generate_embeddings([embedding_text(c) for c in chunks])
    except EmbeddingError as exc:
        log.error(f"embedding generation failed: {exc}")
        log.error(
            "  tip: confirm `ollama serve` is running and "
            f"`ollama list` shows '{settings.EMBEDDING_MODEL}'."
        )
        return 1
    log.info(f"embedding stage done in {time.time() - t1:.1f}s "
             f"({len(embeddings)} vectors, dim={len(embeddings[0])})")

    # -------------------------------------------------------------- store stage
    t2 = time.time()
    try:
        collection = reset_collection(args.collection) if args.reset \
            else init_collection(args.collection)
        before_count = collection.count()
        inserted = add_documents(collection, chunks, embeddings)
        after_count = collection.count()
    except VectorStoreError as exc:
        log.error(f"ChromaDB write failed: {exc}")
        return 1
    log.info(f"store stage done in {time.time() - t2:.1f}s")

    # -------------------------------------------------------------- summary
    elapsed = time.time() - t0
    log.info("--- ingestion complete ---")
    log.info(f"  chunks produced  : {len(chunks)}")
    log.info(f"  embeddings created: {len(embeddings)} (dim {len(embeddings[0])})")
    log.info(f"  collection       : {args.collection}")
    log.info(f"    before         : {before_count}")
    log.info(f"    upserted       : {inserted}")
    log.info(f"    after          : {after_count}")
    log.info(f"  total wall-clock : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
