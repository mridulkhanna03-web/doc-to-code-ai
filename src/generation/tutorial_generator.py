"""Tutorial generation: topic + retrieved context → markdown tutorial on disk.

Public API:

- :func:`generate_tutorial` — produce one tutorial for one topic and save it.
- :func:`generate_tutorials` — batch helper for producing the Day-3 set of 10.

Each tutorial is 800-1200 words, includes at least two working Python code
blocks, a "Common pitfalls" section, and a "Sources" footer — enforced by
the tutorial prompt in :mod:`prompts`. We don't try to mechanically validate
word count, but we log a warning when it falls outside the target band so
the operator notices a misbehaving generation.

The generated markdown is saved at ``tutorials/{slugged-topic}.md``. The
function returns a :class:`TutorialResult` dict so the caller can compose
without re-reading the file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, TypedDict

from config import settings
from src.generation.llm_client import LLMError, generate
from src.generation.prompts import build_tutorial_prompt
from src.retrieval.retriever import retrieve_context
from src.retrieval.vector_store import SearchHit
from src.utils.logger import setup_logger

log = setup_logger("generation.tutorial_generator")

_TUTORIAL_K: int = 6
_MIN_WORDS: int = 800
_MAX_WORDS: int = 1200
_SLUG_BAD_RE = re.compile(r"[^a-z0-9]+")


class TutorialResult(TypedDict):
    topic: str
    slug: str
    path: str
    markdown: str
    word_count: int
    backend: str
    hits: list[SearchHit]


def _slug(text: str, max_len: int = 60) -> str:
    s = _SLUG_BAD_RE.sub("-", text.lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "tutorial"


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def generate_tutorial(
    topic: str,
    k: int = _TUTORIAL_K,
    out_dir: Path | None = None,
) -> TutorialResult:
    """Generate one tutorial markdown for ``topic`` and write it to disk.

    Retrieves ``k`` documentation chunks, assembles the tutorial prompt,
    calls the LLM (Ollama primary, Gemini fallback), then writes the
    resulting markdown to ``{out_dir}/{slug}.md``.

    Errors from the LLM (:class:`LLMError`) propagate — the caller decides
    whether to retry or skip this topic. File-write errors are wrapped and
    re-raised with the target path in context.
    """
    target_dir = Path(out_dir) if out_dir is not None else settings.TUTORIALS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"generating tutorial: {topic!r}")
    context, hits = retrieve_context(topic, k=k)
    if not hits:
        log.warning(f"no documentation chunks retrieved for {topic!r} — tutorial will be weak")

    system, user = build_tutorial_prompt(context, topic)

    try:
        markdown, backend = generate(system, user)
    except LLMError as exc:
        log.error(f"tutorial LLM call failed for {topic!r}: {exc}")
        raise

    wc = _word_count(markdown)
    if wc < _MIN_WORDS:
        log.warning(f"tutorial below target word count: {wc} < {_MIN_WORDS} (topic={topic!r})")
    elif wc > _MAX_WORDS:
        log.warning(f"tutorial above target word count: {wc} > {_MAX_WORDS} (topic={topic!r})")

    slug = _slug(topic)
    path = target_dir / f"{slug}.md"
    try:
        path.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        log.error(f"failed to write tutorial to {path}: {exc}")
        raise

    log.info(
        f"saved tutorial {path} ({wc} words, backend={backend}, "
        f"{len(hits)} source chunks)"
    )
    return TutorialResult(
        topic=topic,
        slug=slug,
        path=str(path),
        markdown=markdown,
        word_count=wc,
        backend=backend,
        hits=hits,
    )


def generate_tutorials(
    topics: Iterable[str],
    out_dir: Path | None = None,
    stop_on_error: bool = False,
) -> list[TutorialResult]:
    """Generate tutorials for every topic in ``topics`` and return the list.

    By default, a single LLM failure logs and skips the topic so the batch
    completes; set ``stop_on_error=True`` to abort on first failure.
    """
    results: list[TutorialResult] = []
    topics_list = list(topics)
    for i, topic in enumerate(topics_list, start=1):
        log.info(f"--- tutorial {i}/{len(topics_list)}: {topic!r} ---")
        try:
            result = generate_tutorial(topic, out_dir=out_dir)
            results.append(result)
        except LLMError as exc:
            log.error(f"skipping {topic!r}: {exc}")
            if stop_on_error:
                raise
        except OSError as exc:
            log.error(f"skipping {topic!r} (write failed): {exc}")
            if stop_on_error:
                raise
    log.info(f"generated {len(results)}/{len(topics_list)} tutorials")
    return results


# Suggested topic set for the Day-3 "10 tutorials" milestone. Imported by
# scripts so the exact slate is reproducible.
DEFAULT_TUTORIAL_TOPICS: list[str] = [
    "Creating a basic FastAPI application",
    "Defining path parameters with type validation",
    "Handling query parameters and optional values",
    "Accepting a request body with a Pydantic model",
    "Adding a response model to filter output fields",
    "Implementing dependency injection with Depends",
    "Securing endpoints with OAuth2 password flow",
    "Handling file uploads with UploadFile",
    "Running background tasks after a response",
    "Custom exception handlers and error responses",
]
