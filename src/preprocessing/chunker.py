"""Semantic chunking that keeps fenced code blocks atomic.

The pipeline goes: cleaned markdown text in → list of ``Chunk`` dicts out, each
chunk a self-contained slice of the document with a stable id and structured
metadata that the vector store and retriever both rely on.

Two design rules baked into this module:

1. **Code blocks are never split.** A ``` fenced block becomes one chunk even
   if it exceeds the size target. This is what makes generated code examples
   actually runnable.
2. **Recursive separator priority** mirrors the spec: paragraph → line →
   sentence → word. Implemented via LangChain's
   ``RecursiveCharacterTextSplitter`` with custom separators.

``chunk_size`` and ``chunk_overlap`` come from ``config.settings`` and are
treated as *approximate* token budgets — internally we operate in characters
(≈ 4 chars/token), which is robust without pinning a specific tokenizer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from src.utils.logger import setup_logger

log = setup_logger("preprocessing.chunker")

# Heuristic: ~4 characters per token for English + code mixed corpus.
_CHARS_PER_TOKEN: int = 4

# Minimum chars for a prose chunk to be worth keeping. Prose shorter than this
# is almost always connective text ("or:", "as in:", "...to:") that appears
# between two related code blocks. Code chunks bypass this filter — we never
# discard a code block however small.
_MIN_PROSE_CHARS: int = 30

# Same code-fence pattern used elsewhere — keep in sync.
_CODE_FENCE_RE = re.compile(
    r"^```[A-Za-z0-9_+\-]*\s*\n.*?\n```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# All-level markdown headings, used to locate the closest preceding section.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Separator priority from the project spec: paragraph > line > sentence > word.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

# Restricted slug character set for chunk ids — safe for filenames and URLs.
_SLUG_BAD_CHARS_RE = re.compile(r"[^a-z0-9]+")


class ChunkMetadata(TypedDict):
    source_doc: str
    section: str
    position: int
    has_code: bool


class Chunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata


def _slug(text: str, max_len: int = 60) -> str:
    """Lowercase, hyphenated, ASCII-only slug — safe for ids and filenames."""
    s = _SLUG_BAD_CHARS_RE.sub("-", text.lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "section"


def _doc_name(source: str | Path | None) -> str:
    """Build a unique-enough ``source_doc`` identifier from a file path.

    Just using ``Path.stem`` collides when two files share a filename across
    folders (FastAPI has ``tutorial/websockets.md`` and ``advanced/websockets.md``).
    We include the parent directory name so the id becomes ``tutorial-websockets``
    vs ``advanced-websockets``. The parent ``docs`` (for top-level files) is
    dropped to avoid the redundant ``docs-index``-style prefix.
    """
    if source is None:
        return "doc"
    p = Path(str(source))
    stem = p.stem or "doc"
    parent = p.parent.name
    if not parent or parent in {".", "docs"}:
        return stem
    return f"{parent}-{stem}"


def _section_finder(text: str):
    """Return a closure: ``section_at(offset) -> heading string``.

    The closure scans the precomputed heading list to find the closest
    heading at or before ``offset``. If no heading precedes ``offset``,
    it returns ``"intro"``.
    """
    headings: list[tuple[int, str]] = [
        (m.start(), m.group(2).strip()) for m in _HEADING_RE.finditer(text)
    ]

    def section_at(offset: int) -> str:
        current = "intro"
        for pos, title in headings:
            if pos > offset:
                break
            current = title
        return current

    return section_at


def _build_splitter(chunk_size_chars: int, overlap_chars: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=_SEPARATORS,
        keep_separator=False,
        length_function=len,
        is_separator_regex=False,
    )


def _segment(text: str) -> list[tuple[int, int, str, str]]:
    """Cut ``text`` into ordered prose/code segments.

    Each segment is ``(start, end, kind, content)`` where ``kind`` is either
    ``"prose"`` or ``"code"``. Prose segments are everything outside fenced
    code blocks; code segments are the full ``` … ``` blocks verbatim.
    """
    segments: list[tuple[int, int, str, str]] = []
    cursor = 0
    for m in _CODE_FENCE_RE.finditer(text):
        if m.start() > cursor:
            segments.append((cursor, m.start(), "prose", text[cursor:m.start()]))
        segments.append((m.start(), m.end(), "code", m.group(0)))
        cursor = m.end()
    if cursor < len(text):
        segments.append((cursor, len(text), "prose", text[cursor:]))
    return segments


def embedding_text(chunk: Chunk) -> str:
    """Return chunk text augmented with doc + section context for embedding.

    The stored chunk text (``chunk['text']``) is kept clean so the LLM sees
    documentation as written. But during embedding we prepend a compact
    ``[Doc: X] [Section: Y]`` header — this lifts doc/section keywords into
    the embedding vector so a query like *"how to use first-steps"* matches
    chunks whose body doesn't repeat that phrase. Standard RAG retrieval-
    augmentation pattern; trades a small dilution of body signal for a large
    gain in metadata-driven recall.
    """
    meta = chunk["metadata"]
    return f"[Doc: {meta['source_doc']}] [Section: {meta['section']}]\n\n{chunk['text']}"


def semantic_chunk(
    text: str,
    source: str | Path | None = None,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a cleaned markdown document.

    Parameters mirror the project spec: ``chunk_size`` ≈ tokens (default 512),
    ``overlap`` ≈ tokens (default 100). Internally converted to characters via
    a 4-char-per-token heuristic.

    Returns a list of :class:`Chunk` dicts, in document order, each with a
    deterministic ``id`` of the form ``{doc_name}_{section_slug}_{n}``.
    """
    if not text or not text.strip():
        return []

    chunk_size_chars = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN

    splitter = _build_splitter(chunk_size_chars, overlap_chars)
    section_at = _section_finder(text)
    doc_name = _doc_name(source)

    chunks: list[Chunk] = []
    chunk_n = 0

    for start, _end, kind, content in _segment(text):
        if not content.strip():
            continue

        if kind == "code":
            section = section_at(start)
            chunks.append(
                Chunk(
                    id=f"{doc_name}_{_slug(section)}_{chunk_n}",
                    text=content,
                    metadata=ChunkMetadata(
                        source_doc=doc_name,
                        section=section,
                        position=start,
                        has_code=True,
                    ),
                )
            )
            chunk_n += 1
            continue

        sub_chunks = splitter.split_text(content)
        for sub in sub_chunks:
            sub_clean = sub.strip()
            if not sub_clean:
                continue
            has_code = "```" in sub_clean
            if not has_code and len(sub_clean) < _MIN_PROSE_CHARS:
                continue
            # Position is the parent-segment offset; we don't try to track per-sub-chunk
            # offsets because LangChain doesn't return them and the precision isn't worth
            # the complexity here.
            section = section_at(start)
            chunks.append(
                Chunk(
                    id=f"{doc_name}_{_slug(section)}_{chunk_n}",
                    text=sub_clean,
                    metadata=ChunkMetadata(
                        source_doc=doc_name,
                        section=section,
                        position=start,
                        has_code=has_code,
                    ),
                )
            )
            chunk_n += 1

    log.debug(
        f"chunked {doc_name}: {len(chunks)} chunks "
        f"(target={chunk_size}t / {chunk_size_chars}c, overlap={overlap}t / {overlap_chars}c)"
    )
    return chunks
