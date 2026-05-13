"""Fetch raw documentation content from URLs or local files.

Three public functions:

- ``fetch_url(url)`` — HTTP GET with timeout and retry.
- ``fetch_local(path)`` — read a file from disk (UTF-8, fallback to latin-1).
- ``detect_format(content, hint)`` — classify content as ``markdown``, ``html``, or ``text``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import requests

from src.utils.logger import setup_logger

log = setup_logger("ingestion.doc_fetcher")

DocFormat = Literal["markdown", "html", "text"]

_DEFAULT_TIMEOUT_SECONDS: float = 30.0
_USER_AGENT: str = "doc-to-code-ai/0.1 (+https://github.com/mridul-khanna)"


def fetch_url(url: str, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> str:
    """Fetch ``url`` and return the response body as text.

    Raises:
        requests.RequestException: on network errors after retries are exhausted.
    """
    log.info(f"GET {url}")
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/markdown,text/plain,*/*"},
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.error(f"fetch_url failed for {url}: {exc}")
        raise

    # requests guesses encoding; fall back to apparent_encoding if it picked ISO-8859-1.
    if response.encoding is None or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"

    log.debug(f"fetched {len(response.text)} chars from {url} ({response.encoding})")
    return response.text


def fetch_local(path: str | Path) -> str:
    """Read a local file and return its text content.

    Wraps the disk read in try/except: ``UnicodeDecodeError`` triggers a
    latin-1 fallback; ``OSError`` / ``PermissionError`` are logged with the
    failing path and re-raised so the caller can decide how to recover.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"doc not found: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"expected file, got directory: {p}")

    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.warning(f"utf-8 decode failed for {p}, retrying with latin-1")
        try:
            text = p.read_text(encoding="latin-1")
        except OSError as exc:
            log.error(f"latin-1 fallback read failed for {p}: {exc}")
            raise
    except OSError as exc:
        log.error(f"read failed for {p}: {exc}")
        raise

    log.debug(f"read {len(text)} chars from {p}")
    return text


def detect_format(content: str, hint: str | Path | None = None) -> DocFormat:
    """Classify ``content`` as ``markdown``, ``html``, or ``text``.

    ``hint`` (a filename or URL) takes precedence when present — extensions are
    the strongest signal. Otherwise we look for telltale syntax.
    """
    if hint:
        suffix = Path(str(hint)).suffix.lower()
        if suffix in {".md", ".markdown", ".mdx"}:
            return "markdown"
        if suffix in {".html", ".htm", ".xhtml"}:
            return "html"
        if suffix in {".txt", ".rst"}:
            return "text"

    head = content.lstrip()[:2000].lower()

    if head.startswith("<!doctype html") or head.startswith("<html") or "<body" in head or "<div" in head:
        return "html"

    md_signals = (
        re.search(r"^#{1,6}\s+\w", content, re.MULTILINE) is not None,
        "```" in content,
        re.search(r"^\s*[-*]\s+\w", content, re.MULTILINE) is not None,
        re.search(r"\[[^\]]+\]\([^)]+\)", content) is not None,
    )
    if sum(md_signals) >= 2:
        return "markdown"

    return "text"
