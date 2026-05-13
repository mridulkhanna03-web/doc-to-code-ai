"""RAG retrieval — query → context bundle the generator can ground on.

Public API:

- :func:`retrieve` — embed query, search ChromaDB, return ranked hits.
- :func:`construct_context` — assemble hits into a single prompt-ready
  context string with numbered source attributions.
- :func:`retrieve_context` — convenience one-shot: query → ``(context_str, hits)``.

The "reranking" step the spec asks for is satisfied by ChromaDB's own
distance-ordered output (k-NN over cosine space). We layer one optimisation
on top: a soft diversity bonus that demotes the *third* or later hit from the
same source document — so the LLM doesn't get three near-identical chunks at
the cost of breadth. Pass ``diversify=False`` to disable.
"""

from __future__ import annotations

from typing import Optional

from config import settings
from src.embeddings.embedding_generator import embed_query
from src.retrieval.vector_store import SearchHit, init_collection, search
from src.utils.logger import setup_logger

log = setup_logger("retrieval.retriever")


def _diversity_rerank(hits: list[SearchHit], soft_penalty: float = 0.05) -> list[SearchHit]:
    """Apply a small distance penalty to the 3rd+ chunk from the same source_doc.

    The aim is to favour breadth without throwing away a legitimately rich
    single-doc match. Top 2 hits from any doc keep their original distance;
    the 3rd hit picks up +``soft_penalty``, the 4th +``2*soft_penalty``, etc.
    Hits are then re-sorted by adjusted distance.
    """
    counts: dict[str, int] = {}
    adjusted: list[tuple[float, SearchHit]] = []
    for h in hits:
        doc = h["metadata"].get("source_doc", "")
        n = counts.get(doc, 0)
        bonus = max(0, n - 1) * soft_penalty
        adjusted.append((h["distance"] + bonus, h))
        counts[doc] = n + 1
    adjusted.sort(key=lambda pair: pair[0])
    return [h for _, h in adjusted]


def retrieve(
    query: str,
    k: int = settings.K_RESULTS,
    similarity_threshold: Optional[float] = settings.SIMILARITY_THRESHOLD,
    diversify: bool = True,
) -> list[SearchHit]:
    """Embed ``query`` and return up to ``k`` ranked, optionally diversified hits.

    Any failure in the embedding or search layer surfaces as the original
    ``EmbeddingError`` / ``VectorStoreError`` so the caller can decide how to
    react.
    """
    query_vec = embed_query(query)
    collection = init_collection()

    # We over-fetch a bit to give the diversity rerank room to reorder.
    raw_k = k * 2 if diversify else k
    hits = search(collection, query_vec, k=raw_k, similarity_threshold=similarity_threshold)

    if diversify:
        hits = _diversity_rerank(hits)

    hits = hits[:k]
    log.info(f"retrieve({query!r}, k={k}): returned {len(hits)} hits")
    return hits


def construct_context(hits: list[SearchHit]) -> str:
    """Render a numbered context block suitable for dropping into an LLM prompt.

    Each hit becomes a section like::

        [Source 1: tutorial-first-steps / Step 1: import FastAPI]
        <chunk text>

    The format is deliberately compact (no extra markdown headers) so the
    LLM's attention budget stays on the doc content, not framing.
    """
    if not hits:
        return "(no relevant documentation chunks found for this query)"

    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        doc = meta.get("source_doc", "?")
        section = meta.get("section", "?")
        blocks.append(f"[Source {i}: {doc} / {section}]\n{hit['text']}")

    return "\n\n---\n\n".join(blocks)


def retrieve_context(
    query: str,
    k: int = settings.K_RESULTS,
    similarity_threshold: Optional[float] = settings.SIMILARITY_THRESHOLD,
    diversify: bool = True,
) -> tuple[str, list[SearchHit]]:
    """Convenience: ``(context_string, raw_hits)`` for a single query."""
    hits = retrieve(query, k=k, similarity_threshold=similarity_threshold, diversify=diversify)
    return construct_context(hits), hits
