"""Ingestion orchestrator.

Takes a source (a directory, file, or list of paths/URLs), runs each item
through fetch → parse → clean → chunk, and returns a flat list of ready-to-
embed :class:`~src.preprocessing.chunker.Chunk` dicts. Optionally persists the
result as JSON for inspection and reuse.

Public API:

- :func:`process_source` — process one file/URL into chunks.
- :func:`run_ingestion` — process many sources, aggregate chunks.
- :func:`save_processed` — dump chunks to JSON.
- :func:`load_processed` — read them back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from config import settings
from src.ingestion.doc_fetcher import detect_format, fetch_local, fetch_url
from src.ingestion.doc_parser import parse
from src.preprocessing.chunker import Chunk, semantic_chunk
from src.preprocessing.text_cleaner import clean_text
from src.utils.logger import setup_logger

log = setup_logger("ingestion.pipeline")


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _expand_sources(source: str | Path | Iterable[str | Path], pattern: str) -> list[str | Path]:
    """Resolve ``source`` into a concrete list of files/URLs to process."""
    if isinstance(source, (list, tuple, set)):
        return list(source)

    if isinstance(source, str) and _is_url(source):
        return [source]

    p = Path(source)
    if p.is_dir():
        return sorted(p.rglob(pattern))
    if p.is_file():
        return [p]

    raise FileNotFoundError(f"source not found: {source}")


def process_source(source: str | Path) -> list[Chunk]:
    """Run the full ingestion pipeline for a single file or URL.

    Returns ``[]`` on read or parse failure — the error is logged but the
    pipeline continues on the next item so a single bad file can't sink the
    whole batch.
    """
    src_str = str(source)
    try:
        if _is_url(src_str):
            content = fetch_url(src_str)
        else:
            content = fetch_local(src_str)
    except Exception as exc:  # noqa: BLE001 — we deliberately swallow & log
        log.error(f"fetch failed for {src_str}: {exc}")
        return []

    if not content.strip():
        log.warning(f"empty content: {src_str}")
        return []

    fmt = detect_format(content, hint=src_str)

    try:
        parsed = parse(content, fmt, source=src_str)
    except Exception as exc:  # noqa: BLE001
        log.error(f"parse failed for {src_str}: {exc}")
        return []

    cleaned = clean_text(parsed["text"])
    chunks = semantic_chunk(cleaned, source=src_str)

    log.debug(f"processed {src_str}: {len(chunks)} chunks")
    return chunks


def run_ingestion(
    source: str | Path | Iterable[str | Path],
    pattern: str = "*.md",
    log_every: int = 25,
) -> list[Chunk]:
    """Process all items in ``source`` and return the aggregated chunks.

    ``source`` may be a directory (globbed by ``pattern``), a single file, a
    URL, or any iterable of those. ``log_every`` controls progress logging.
    """
    items = _expand_sources(source, pattern)
    log.info(f"ingestion start: {len(items)} item(s) from {source}")

    all_chunks: list[Chunk] = []
    failures = 0
    for i, item in enumerate(items, 1):
        chunks = process_source(item)
        if not chunks:
            failures += 1
        all_chunks.extend(chunks)

        if i % log_every == 0 or i == len(items):
            log.info(
                f"  progress {i}/{len(items)}: "
                f"{len(all_chunks)} chunks so far, {failures} failures"
            )

    log.info(
        f"ingestion done: {len(all_chunks)} chunks from {len(items) - failures}/{len(items)} sources"
    )
    return all_chunks


def save_processed(chunks: list[Chunk], output_path: str | Path | None = None) -> Path:
    """Persist ``chunks`` as JSON. Returns the path written to.

    Wraps file open and ``json.dump`` so disk-full / permission / serialisation
    errors surface with the target path and chunk count.
    """
    target = Path(output_path) if output_path else settings.PROCESSED_DIR / "chunks.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        log.error(f"save_processed failed (path={target}, chunks={len(chunks)}): {exc}")
        raise
    log.info(f"saved {len(chunks)} chunks to {target}")
    return target


def load_processed(path: str | Path | None = None) -> list[Chunk]:
    """Read chunks JSON previously written by :func:`save_processed`.

    Wraps file open and ``json.load`` so missing-file / decode errors surface
    with the source path.
    """
    source = Path(path) if path else settings.PROCESSED_DIR / "chunks.json"
    try:
        with source.open("r", encoding="utf-8") as f:
            chunks: list[Chunk] = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error(f"load_processed failed (path={source}): {exc}")
        raise
    log.info(f"loaded {len(chunks)} chunks from {source}")
    return chunks
