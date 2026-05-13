"""Generate L2-normalised embeddings via Ollama's ``/api/embed`` endpoint.

Default model is ``nomic-embed-text`` (768 dimensions). We talk to Ollama over
HTTP rather than via a Python wrapper to keep the dependency surface small.

Every external call (HTTP POST, JSON decode) is wrapped in its own try/except
so failures surface with context: which endpoint, which model, which batch.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import requests

from config import settings
from src.utils.logger import setup_logger

log = setup_logger("embeddings.generator")

_EMBED_PATH: str = "/api/embed"
_MAX_RETRIES: int = 3
_BACKOFF_BASE_SECONDS: float = 0.5


class EmbeddingError(RuntimeError):
    """Raised when Ollama embedding requests fail or return invalid data."""


def _post_embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """POST one batch of texts to Ollama. All HTTP/JSON errors are caught,
    logged with context, and re-raised as :class:`EmbeddingError`.
    """
    url = settings.OLLAMA_HOST.rstrip("/") + _EMBED_PATH
    payload = {"model": settings.EMBEDDING_MODEL, "input": list(texts)}

    try:
        response = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.error(
            f"Ollama embed request failed: {exc} "
            f"(url={url}, model={settings.EMBEDDING_MODEL}, batch_size={len(texts)})"
        )
        raise EmbeddingError(f"embed POST failed: {exc}") from exc

    try:
        body = response.json()
    except ValueError as exc:
        log.error(f"Ollama embed returned non-JSON body (url={url}): {exc}")
        raise EmbeddingError(f"embed response not JSON: {exc}") from exc

    embeddings = body.get("embeddings")
    if not embeddings:
        log.error(f"Ollama embed returned no embeddings (model={settings.EMBEDDING_MODEL}, body={body})")
        raise EmbeddingError(f"no embeddings in response: {body}")
    if len(embeddings) != len(texts):
        log.error(
            f"Ollama embed count mismatch: requested={len(texts)}, got={len(embeddings)}"
        )
        raise EmbeddingError(
            f"embedding count mismatch: got {len(embeddings)} for {len(texts)} inputs"
        )
    return embeddings


def _post_with_retry(texts: Sequence[str]) -> list[list[float]]:
    """Retry the embed POST with exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _post_embed_batch(texts)
        except EmbeddingError as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES - 1:
                break
            wait = _BACKOFF_BASE_SECONDS * (2 ** attempt)
            log.warning(
                f"embed batch attempt {attempt + 1}/{_MAX_RETRIES} failed; "
                f"retrying in {wait:.1f}s"
            )
            time.sleep(wait)
    assert last_exc is not None
    raise EmbeddingError(f"all {_MAX_RETRIES} embed attempts failed: {last_exc}") from last_exc


def _l2_normalise(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Row-wise L2-normalise a 2-D list of vectors. Zero rows are left unchanged."""
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).tolist()


def generate_embeddings(
    texts: Sequence[str],
    batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    log_every: int = 5,
) -> list[list[float]]:
    """Return one L2-normalised 768-d vector per input string, in input order.

    The per-batch call is wrapped in try/except so a single bad batch surfaces
    a clear error pointing at the failing batch index.
    """
    if not texts:
        return []

    out: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = list(texts[start:start + batch_size])
        try:
            raw = _post_with_retry(batch)
        except EmbeddingError as exc:
            log.error(
                f"embed failed at batch {batch_idx}/{total_batches} "
                f"(chunks {start}..{start + len(batch) - 1}): {exc}"
            )
            raise

        if batch_idx == 1:
            dim = len(raw[0])
            if dim != settings.EMBEDDING_DIMENSIONS:
                raise EmbeddingError(
                    f"unexpected embedding dim {dim} (expected {settings.EMBEDDING_DIMENSIONS}); "
                    f"is settings.EMBEDDING_MODEL={settings.EMBEDDING_MODEL!r} correct?"
                )

        out.extend(_l2_normalise(raw))

        if batch_idx % log_every == 0 or batch_idx == total_batches:
            log.info(
                f"  embedded batch {batch_idx}/{total_batches} "
                f"({len(out)}/{len(texts)} vectors)"
            )

    return out


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Wraps the call so retrieval-time failures
    surface a clear, query-specific error.
    """
    try:
        vectors = generate_embeddings([text], batch_size=1, log_every=1_000_000)
    except EmbeddingError as exc:
        log.error(f"query embedding failed for {text[:60]!r}: {exc}")
        raise
    return vectors[0]
