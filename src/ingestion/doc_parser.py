"""Parse raw documentation into structured text + code blocks + metadata.

Public API:

- ``parse_markdown(content, source)`` — strip frontmatter, return ``ParsedDoc``.
- ``parse_html(content, source)`` — strip chrome, convert <pre><code> to fenced
  blocks, return ``ParsedDoc``.
- ``extract_code_blocks(text)`` — find all fenced ``` blocks in markdown.
- ``extract_metadata(text, source)`` — pull title + section headings.
- ``parse(content, fmt, source)`` — dispatcher keyed on ``doc_fetcher.detect_format``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from bs4 import BeautifulSoup

from src.ingestion.doc_fetcher import DocFormat
from src.utils.logger import setup_logger

log = setup_logger("ingestion.doc_parser")


class CodeBlock(TypedDict):
    id: int
    language: str
    code: str
    start: int  # char offset in text
    end: int


class Section(TypedDict):
    level: int
    title: str


class Metadata(TypedDict):
    title: str
    sections: list[Section]
    source_path: str
    doc_type: str


class ParsedDoc(TypedDict):
    text: str
    code_blocks: list[CodeBlock]
    metadata: Metadata


# Fenced code block: opening ``` (with optional language), body, closing ```.
_CODE_BLOCK_RE = re.compile(
    r"^```([A-Za-z0-9_+\-]*)\s*\n(.*?)\n```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

_YAML_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Plain-text include `{! foo !}` — stripped, never resolved.
_PLAIN_INCLUDE_RE = re.compile(r"^\{!.*?!\}\s*$", re.MULTILINE)

# MkDocs-Material code include: `{* path/to/file.py *}` with optional flags like
# `hl[1]`, `hl[1:5]`, `ln[1:10]`. We capture only the path; flags are ignored
# during resolution (they only matter when MkDocs renders highlights).
_CODE_INCLUDE_RE = re.compile(
    r"\{\*\s+(?P<path>\S+?)(?:\s+(?:hl|ln)\[[^\]]*\])*\s*\*\}"
)

# Map file extensions to fenced-block languages.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".sh": "bash",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` until we hit a directory whose direct child is ``docs_src``.

    FastAPI's repo layout has ``docs_src/`` at the repo root; markdown files
    reference it as ``../../docs_src/...`` even though the literal relative
    path doesn't resolve. This helper lets us anchor those references back to
    the real location.
    """
    p = start.resolve()
    if p.is_file():
        p = p.parent
    for ancestor in (p, *p.parents):
        if (ancestor / "docs_src").is_dir():
            return ancestor
    return None


def _resolve_include_path(
    rel_path: str,
    md_path: Path,
    repo_root: Path | None,
) -> Path | None:
    """Locate the file referenced by an include directive, or return ``None``.

    Resolution order:

    1. Literal: ``md_path.parent / rel_path``.
    2. Repo-rooted: if ``rel_path`` contains ``docs_src/``, take the suffix
       beginning at ``docs_src/`` and join with ``repo_root``.
    """
    literal = (md_path.parent / rel_path).resolve()
    if literal.is_file():
        return literal

    if repo_root is not None and "docs_src/" in rel_path:
        suffix = rel_path[rel_path.index("docs_src/"):]
        candidate = (repo_root / suffix).resolve()
        if candidate.is_file():
            return candidate

    return None


def _inline_code_includes(text: str, md_path: Path | None) -> tuple[str, int, int]:
    """Replace ``{* path *}`` directives with fenced code blocks read from disk.

    Returns ``(new_text, resolved_count, unresolved_count)``. Unresolved
    directives are left in place; ``text_cleaner`` will strip them later.
    """
    if md_path is None or not _CODE_INCLUDE_RE.search(text):
        return text, 0, 0

    repo_root = _find_repo_root(md_path)
    counts = {"resolved": 0, "unresolved": 0}

    def _sub(match: re.Match[str]) -> str:
        rel = match.group("path")
        resolved = _resolve_include_path(rel, md_path, repo_root)
        if resolved is None:
            counts["unresolved"] += 1
            log.warning(f"unresolved include in {md_path.name}: {rel}")
            return match.group(0)
        try:
            code = resolved.read_text(encoding="utf-8").rstrip("\n")
        except OSError as exc:
            counts["unresolved"] += 1
            log.warning(f"read failed for include {resolved}: {exc}")
            return match.group(0)
        counts["resolved"] += 1
        lang = _EXT_TO_LANG.get(resolved.suffix.lower(), "")
        return f"\n```{lang}\n{code}\n```\n"

    new_text = _CODE_INCLUDE_RE.sub(_sub, text)
    return new_text, counts["resolved"], counts["unresolved"]


def extract_code_blocks(text: str) -> list[CodeBlock]:
    """Find all fenced code blocks in ``text`` and return them in order."""
    blocks: list[CodeBlock] = []
    for i, m in enumerate(_CODE_BLOCK_RE.finditer(text)):
        blocks.append(
            CodeBlock(
                id=i,
                language=(m.group(1) or "text").strip(),
                code=m.group(2),
                start=m.start(),
                end=m.end(),
            )
        )
    return blocks


def extract_metadata(text: str, source: str | Path | None = None) -> Metadata:
    """Pull a title (first H1) and the full heading outline from markdown text."""
    title = ""
    sections: list[Section] = []
    for m in _HEADING_RE.finditer(text):
        level = len(m.group(1))
        heading = m.group(2).strip()
        sections.append(Section(level=level, title=heading))
        if level == 1 and not title:
            title = heading

    if not title and source:
        title = Path(str(source)).stem.replace("-", " ").replace("_", " ").title()

    return Metadata(
        title=title,
        sections=sections,
        source_path=str(source) if source is not None else "",
        doc_type="markdown",
    )


def parse_markdown(content: str, source: str | Path | None = None) -> ParsedDoc:
    """Parse a markdown document.

    Steps:

    1. Strip YAML frontmatter.
    2. Strip plain-text MkDocs includes (``{! foo !}``) — we don't follow those.
    3. Resolve code includes (``{* path *}``) by reading the referenced file
       and inlining its content as a fenced code block. Unresolved directives
       are left for ``text_cleaner`` to strip.
    4. Extract code blocks + metadata from the post-resolution text.
    """
    text = _YAML_FRONTMATTER_RE.sub("", content)
    text = _PLAIN_INCLUDE_RE.sub("", text)

    md_path = Path(source) if source is not None else None
    text, resolved, unresolved = _inline_code_includes(text, md_path)
    if resolved or unresolved:
        log.debug(
            f"includes in {md_path.name if md_path else '<inline>'}: "
            f"resolved={resolved}, unresolved={unresolved}"
        )

    code_blocks = extract_code_blocks(text)
    metadata = extract_metadata(text, source)
    metadata["doc_type"] = "markdown"

    log.debug(f"parsed markdown: title={metadata['title']!r}, "
              f"sections={len(metadata['sections'])}, code_blocks={len(code_blocks)}")
    return ParsedDoc(text=text, code_blocks=code_blocks, metadata=metadata)


def parse_html(content: str, source: str | Path | None = None) -> ParsedDoc:
    """Parse an HTML document via BeautifulSoup. Strips nav/script/style chrome,
    converts ``<pre><code>`` blocks to fenced markdown, returns a ParsedDoc.

    BeautifulSoup is permissive but ``lxml`` can fail on truly broken input —
    we catch any parser-layer exception and re-raise with the source path so
    the failing document is obvious in logs.
    """
    try:
        soup = BeautifulSoup(content, "lxml")
    except Exception as exc:  # bs4 raises a wide variety; catch broadly
        log.error(f"BeautifulSoup failed to parse {source or '<inline>'}: {exc}")
        raise

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    sections: list[Section] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(tag.name[1])
        sections.append(Section(level=level, title=tag.get_text(strip=True)))

    # Convert <pre><code class="language-xxx">...</code></pre> to fenced markdown
    # so downstream code-block extraction can find them.
    for pre in soup.find_all("pre"):
        code_tag = pre.find("code")
        if code_tag is None:
            continue
        language = ""
        classes = code_tag.get("class") or []
        for c in classes:
            if c.startswith("language-"):
                language = c[len("language-"):]
                break
        code_text = code_tag.get_text()
        replacement = soup.new_string(f"\n```{language}\n{code_text}\n```\n")
        pre.replace_with(replacement)

    body_text = soup.get_text(separator="\n")
    # Collapse 3+ newlines down to 2 so paragraph boundaries stay clean.
    body_text = re.sub(r"\n{3,}", "\n\n", body_text)

    code_blocks = extract_code_blocks(body_text)

    metadata = Metadata(
        title=title,
        sections=sections,
        source_path=str(source) if source is not None else "",
        doc_type="html",
    )
    log.debug(f"parsed html: title={title!r}, sections={len(sections)}, code_blocks={len(code_blocks)}")
    return ParsedDoc(text=body_text, code_blocks=code_blocks, metadata=metadata)


def parse(content: str, fmt: DocFormat, source: str | Path | None = None) -> ParsedDoc:
    """Dispatch to ``parse_markdown`` / ``parse_html`` / plain-text passthrough."""
    if fmt == "markdown":
        return parse_markdown(content, source)
    if fmt == "html":
        return parse_html(content, source)

    metadata = extract_metadata(content, source)
    metadata["doc_type"] = "text"
    return ParsedDoc(text=content, code_blocks=extract_code_blocks(content), metadata=metadata)
