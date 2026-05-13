"""ChromaDB persistent vector store with HNSW + cosine similarity.

Public API:

- :func:`get_client` — singleton ``PersistentClient`` pointing at ``data/chroma_storage``.
- :func:`init_collection` — get or create the ``fastapi_docs_embeddings`` collection.
- :func:`add_documents` — upsert chunks + embeddings + metadata in batches.
- :func:`search` — k-NN query with cosine-distance filter.
- :func:`reset_collection` — drop and recreate (useful for re-ingestion).

Every ChromaDB call is wrapped in try/except, logged with collection name and
operation, then re-raised as :class:`VectorStoreError` so callers see a single
stable failure type.
"""

from __future__ import annotations

from typing import Any, Sequence, TypedDict

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from config import settings
from src.preprocessing.chunker import Chunk
from src.utils.logger import setup_logger

log = setup_logger("retrieval.vector_store")

_ADD_BATCH_SIZE: int = 500  # ChromaDB recommends batched adds for large corpora.
_client: chromadb.PersistentClient | None = None


class VectorStoreError(RuntimeError):
    """Raised when a ChromaDB operation fails."""


class SearchHit(TypedDict):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


def get_client() -> chromadb.PersistentClient:
    """Return a singleton :class:`PersistentClient` for ``data/chroma_storage``.

    Wraps client construction so disk / permission errors at the storage path
    surface clearly instead of as a cryptic chroma traceback.
    """
    global _client
    if _client is not None:
        return _client

    try:
        _client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    except Exception as exc:  # chroma raises a variety of errors during init
        log.error(f"Chroma PersistentClient init failed at {settings.CHROMA_DIR}: {exc}")
        raise VectorStoreError(f"chroma init failed: {exc}") from exc

    log.debug(f"Chroma client ready at {settings.CHROMA_DIR}")
    return _client


def init_collection(name: str = settings.CHROMA_COLLECTION) -> Collection:
    """Get or create the named collection. HNSW index, cosine space."""
    client = get_client()
    try:
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        log.error(f"get_or_create_collection failed (name={name}): {exc}")
        raise VectorStoreError(f"collection init failed: {exc}") from exc

    log.debug(f"collection ready: name={name}, count={collection.count()}")
    return collection


def reset_collection(name: str = settings.CHROMA_COLLECTION) -> Collection:
    """Delete the collection if it exists, then recreate it empty."""
    client = get_client()
    try:
        client.delete_collection(name)
        log.info(f"deleted existing collection: {name}")
    except Exception as exc:
        # delete_collection raises if it doesn't exist — that's fine on first run.
        log.debug(f"delete_collection({name}) noop or non-fatal: {exc}")

    return init_collection(name)


def add_documents(
    collection: Collection,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    batch_size: int = _ADD_BATCH_SIZE,
) -> int:
    """Upsert ``chunks`` + ``embeddings`` into ``collection`` in batches.

    Returns the total number of documents inserted. Uses ``upsert`` so re-runs
    are idempotent — existing ids get overwritten rather than raising.
    """
    if len(chunks) != len(embeddings):
        raise VectorStoreError(
            f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )
    if not chunks:
        return 0

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    inserted = 0

    for batch_idx, start in enumerate(range(0, len(chunks), batch_size), start=1):
        end = start + batch_size
        batch_chunks = chunks[start:end]
        batch_embeds = embeddings[start:end]

        ids = [c["id"] for c in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [dict(c["metadata"]) for c in batch_chunks]  # chroma needs plain dicts

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=[list(v) for v in batch_embeds],
            )
        except Exception as exc:
            log.error(
                f"collection.upsert failed (batch {batch_idx}/{total_batches}, "
                f"size={len(batch_chunks)}, collection={collection.name}): {exc}"
            )
            raise VectorStoreError(f"upsert failed at batch {batch_idx}: {exc}") from exc

        inserted += len(batch_chunks)
        if batch_idx % 5 == 0 or batch_idx == total_batches:
            log.info(f"  upsert batch {batch_idx}/{total_batches} ({inserted}/{len(chunks)})")

    log.info(f"added {inserted} documents to '{collection.name}' (total now {collection.count()})")
    return inserted


def search(
    collection: Collection,
    query_embedding: Sequence[float],
    k: int = settings.K_RESULTS,
    similarity_threshold: float | None = settings.SIMILARITY_THRESHOLD,
) -> list[SearchHit]:
    """k-NN search with optional cosine-distance filter.

    ``similarity_threshold`` is an **upper bound on cosine distance** (lower is
    closer; 0.0 = identical, 2.0 = opposite). The default ``0.3`` mirrors the
    project spec — anything beyond that is considered off-topic.
    Pass ``None`` to disable the filter entirely.
    """
    try:
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.error(f"collection.query failed (collection={collection.name}, k={k}): {exc}")
        raise VectorStoreError(f"query failed: {exc}") from exc

    if not result.get("ids") or not result["ids"][0]:
        log.debug(f"search returned 0 results (k={k})")
        return []

    hits: list[SearchHit] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        if similarity_threshold is not None and dist > similarity_threshold:
            continue
        hits.append(SearchHit(id=id_, text=doc, metadata=dict(meta or {}), distance=float(dist)))

    log.debug(
        f"search: k={k}, threshold={similarity_threshold}, "
        f"returned {len(hits)}/{len(ids)} after filter"
    )
    return hits


def collection_count(name: str = settings.CHROMA_COLLECTION) -> int:
    """Return the number of documents currently in the collection (0 if missing)."""
    try:
        return init_collection(name).count()
    except VectorStoreError:
        return 0
