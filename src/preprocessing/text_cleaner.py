"""Text normalisation for documentation prose.

We want to scrub doc-site syntax noise (MkDocs anchor tags, admonition markers,
tab containers, inline HTML chrome, repeated blank lines) WITHOUT touching the
content inside fenced code blocks — that needs to stay byte-exact so generated
code examples remain valid.

Public API:

- ``clean_text(raw)`` — full pipeline.
- ``normalize_whitespace(text)`` — collapse runs of blank lines + strip trailing space.
- ``remove_navigation_elements(text)`` — kill MkDocs / mkdocs-material chrome.
"""

from __future__ import annotations

import re

from src.utils.logger import setup_logger

log = setup_logger("preprocessing.text_cleaner")

# A complete fenced code block (matches what doc_parser uses).
_FENCED_CODE_RE = re.compile(
    r"^```[A-Za-z0-9_+\-]*\s*\n.*?\n```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)

# MkDocs heading-anchor syntax: `## Title { #my-anchor }`
# Anchored to a single line — `[ \t]*` instead of `\s*` so we don't eat the
# trailing newline (which would glue the heading to the next paragraph).
_HEADING_ANCHOR_RE = re.compile(r"[ \t]*\{\s*#[A-Za-z0-9_\-]+\s*\}[ \t]*$", re.MULTILINE)

# MkDocs admonition opener: `!!! note "Optional title"` → leave the title text.
_ADMONITION_RE = re.compile(r'^!!!\s+\w+(?:\s+"([^"]*)")?\s*$', re.MULTILINE)

# MkDocs Material tab marker: `=== "tab name"` → keep "tab name" as a subheading.
_TAB_RE = re.compile(r'^===\s+"([^"]+)"\s*$', re.MULTILINE)

# MkDocs-Material attribute lists: `text { .class #id }` (inline, not anchor-only).
_ATTR_LIST_RE = re.compile(r"\s*\{:[^}]*\}")

# MkDocs include directives in either flavour: `{! foo !}` or `{* foo *}`.
_INCLUDE_RE = re.compile(r"^\{[!*].*?[!*]\}\s*$", re.MULTILINE)

# Bare HTML tags we want to drop in prose (rare — most prose is plain markdown).
_HTML_TAG_RE = re.compile(r"<(?:/?)(?:font|span|div|small|sub|sup|kbd)\b[^>]*>", re.IGNORECASE)

# Trailing whitespace per line.
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# 3+ consecutive newlines collapsed to 2 (one blank line).
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Strip trailing whitespace on each line; cap consecutive blank lines at 1.

    Does NOT trim leading/trailing newlines of the whole string — preserving
    those is important when this is called on a slice between code blocks.
    """
    text = _TRAILING_WS_RE.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


def remove_navigation_elements(text: str) -> str:
    """Strip MkDocs syntax noise (anchors, admonitions, tabs, includes)."""
    text = _HEADING_ANCHOR_RE.sub("", text)
    text = _ATTR_LIST_RE.sub("", text)
    text = _INCLUDE_RE.sub("", text)
    text = _ADMONITION_RE.sub(lambda m: f"**{m.group(1)}**" if m.group(1) else "", text)
    text = _TAB_RE.sub(lambda m: f"**{m.group(1)}**", text)
    text = _HTML_TAG_RE.sub("", text)
    return text


def _clean_prose(prose: str) -> str:
    """Apply all prose-only transforms, in order."""
    prose = remove_navigation_elements(prose)
    prose = normalize_whitespace(prose)
    return prose


def clean_text(raw: str) -> str:
    """Clean ``raw`` markdown text while keeping fenced code blocks byte-exact.

    Strategy: find each code block, leave it alone, clean the prose segments
    between/around them, then reassemble in original order.
    """
    if not raw:
        return raw

    output: list[str] = []
    cursor = 0
    for m in _FENCED_CODE_RE.finditer(raw):
        # Clean the prose chunk before this code block.
        prose = raw[cursor:m.start()]
        output.append(_clean_prose(prose))
        # Append the code block verbatim.
        output.append(m.group(0))
        cursor = m.end()

    # Clean any remaining prose after the last code block (or all of raw if there were none).
    output.append(_clean_prose(raw[cursor:]))

    cleaned = "".join(output)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip() + "\n"
