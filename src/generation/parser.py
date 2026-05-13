"""Parse the structured LLM response into ``{code, explanation, sources}``.

Our prompt asks the LLM to emit, in order:

1. One fenced ``` ```python ... ``` `` code block.
2. 2-4 sentences of plain-English explanation.
3. A final line of the form ``Sources: [1, 3]`` (or ``Sources: []``).

This module turns that string into a typed dict the downstream caller
(validator, API layer, CLI) can use without re-parsing.

Public API:

- :func:`parse_code_response` — full parse → :class:`ParsedResponse`.
- :func:`extract_code_blocks` — every fenced block in the text.
- :func:`extract_explanation` — prose between last code block and Sources line.
- :func:`extract_sources` — the integer list from the ``Sources:`` line.

The parser is **lenient** — real LLM output drifts (no fence language, missing
Sources line, multiple code blocks, capitalised language tag). We accept the
common drifts and degrade gracefully (empty code/explanation/sources rather
than raising).
"""

from __future__ import annotations

import re
from typing import TypedDict

from src.utils.logger import setup_logger

log = setup_logger("generation.parser")

# Fenced code block. Accepts ``python``, ``py``, ``Python``, or no language tag.
# Greedy on the body until the closing ``` on its own line.
_CODE_BLOCK_RE = re.compile(
    r"```\s*(?:python|py)?\s*\n(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)

# ``Sources: [1, 3]`` — case-insensitive, tolerant of bold/markdown wrappers
# in any of the three plausible positions: ``**Sources:** [1]``,
# ``**Sources**: [1]``, ``Sources: [1]``. We capture the inside-the-brackets
# portion.
_SOURCES_RE = re.compile(
    r"\*{0,2}\s*sources\s*\*{0,2}\s*:\s*\*{0,2}\s*\[([^\]]*)\]",
    re.IGNORECASE,
)


class ParsedResponse(TypedDict):
    code: str
    explanation: str
    sources: list[int]
    raw: str


def extract_code_blocks(text: str) -> list[str]:
    """Return every fenced code block in ``text``, in order, stripped of
    leading / trailing whitespace.
    """
    return [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(text)]


def extract_sources(text: str) -> list[int]:
    """Pull the integer list from the ``Sources: [...]`` line.

    Returns ``[]`` if no Sources line is present, the list is empty, or the
    content isn't all-integer.
    """
    match = _SOURCES_RE.search(text)
    if not match:
        return []
    inside = match.group(1).strip()
    if not inside:
        return []

    nums: list[int] = []
    for part in re.split(r"[,\s]+", inside):
        part = part.strip()
        if part.isdigit():
            nums.append(int(part))
    return nums


def extract_explanation(text: str) -> str:
    """Return the prose between the last fenced code block and the
    ``Sources:`` line (or end of text if there's no Sources line).

    If there is no code block, returns everything before the Sources line.
    """
    # Find the end of the LAST code block — the prompt guarantees one, but
    # if a model emits multiple we treat the final one as the answer's code
    # and any earlier blocks as part of the surrounding prose context.
    last_code_end = 0
    for m in _CODE_BLOCK_RE.finditer(text):
        last_code_end = m.end()

    sources_match = _SOURCES_RE.search(text)
    end = sources_match.start() if sources_match else len(text)

    return text[last_code_end:end].strip()


def parse_code_response(response: str) -> ParsedResponse:
    """Parse a full code-generation response into its three components.

    Always returns a populated :class:`ParsedResponse`; missing components
    come back as empty strings / empty list rather than raising — the
    validator stage downstream is responsible for deciding what to do about
    e.g. an empty code field.
    """
    if not response or not response.strip():
        log.warning("parse_code_response received empty input")
        return ParsedResponse(code="", explanation="", sources=[], raw=response or "")

    code_blocks = extract_code_blocks(response)
    code = code_blocks[0] if code_blocks else ""
    explanation = extract_explanation(response)
    sources = extract_sources(response)

    if not code:
        log.warning("parse_code_response: no fenced code block found in LLM output")
    if not explanation and code:
        log.debug("parse_code_response: code present but explanation empty")

    return ParsedResponse(
        code=code,
        explanation=explanation,
        sources=sources,
        raw=response,
    )
