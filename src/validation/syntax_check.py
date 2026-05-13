"""AST-based syntax validation with an optional regeneration loop.

This module is the gate that turns "the LLM produced something that looks like
code" into "the LLM produced something Python can parse." It runs Python's
``ast.parse`` and structures the result so downstream layers can either:

- pass the code through unchanged (when valid), or
- feed the errors back to the LLM for up to ``MAX_REGENERATION_ATTEMPTS``
  regeneration passes (when invalid).

Public API:

- :func:`validate_syntax` — single-shot AST check, returns
  :class:`ValidationResult`.
- :func:`validate_with_retry` — orchestrates the validate → regenerate →
  validate loop. The caller supplies a ``regenerate`` callback that takes
  ``(previous_code, errors_str)`` and returns the LLM's new code attempt.

Per the project's error-handling rule, the ``ast.parse`` call is wrapped in
try/except even though it's a stdlib call — any non-SyntaxError exception
(e.g. a ``ValueError`` on bizarre inputs) becomes a clean
``ValidationResult(valid=False, ...)`` rather than propagating.
"""

from __future__ import annotations

import ast
from typing import Callable, TypedDict

from config import settings
from src.utils.logger import setup_logger

log = setup_logger("validation.syntax_check")


class SyntaxIssue(TypedDict):
    line: int | None
    column: int | None
    message: str


class ValidationResult(TypedDict):
    valid: bool
    errors: list[SyntaxIssue]
    code: str


def validate_syntax(code: str) -> ValidationResult:
    """Parse ``code`` with :mod:`ast` and return a structured result.

    ``valid`` is ``True`` iff ``ast.parse`` succeeds. On failure, the
    ``errors`` list contains a single :class:`SyntaxIssue` with line, column,
    and message from the raised ``SyntaxError``. Empty / whitespace-only
    input is reported as invalid with a clear message rather than letting
    ``ast.parse`` return an empty Module silently.
    """
    if not code or not code.strip():
        return ValidationResult(
            valid=False,
            errors=[SyntaxIssue(line=None, column=None, message="empty code (LLM returned no code block)")],
            code=code,
        )

    try:
        ast.parse(code)
    except SyntaxError as exc:
        issue = SyntaxIssue(
            line=exc.lineno,
            column=exc.offset,
            message=f"{exc.msg}",
        )
        log.debug(f"syntax error: line {exc.lineno}:{exc.offset} — {exc.msg}")
        return ValidationResult(valid=False, errors=[issue], code=code)
    except Exception as exc:  # extremely unusual — non-string inputs etc.
        log.error(f"unexpected ast.parse failure: {type(exc).__name__}: {exc}")
        return ValidationResult(
            valid=False,
            errors=[SyntaxIssue(line=None, column=None, message=f"{type(exc).__name__}: {exc}")],
            code=code,
        )

    return ValidationResult(valid=True, errors=[], code=code)


def format_errors(errors: list[SyntaxIssue]) -> str:
    """Human/LLM-readable error block for the regeneration prompt.

    Format is ``line N, col C: message`` so the model can locate the issue.
    """
    if not errors:
        return "(no errors)"
    lines = []
    for e in errors:
        loc = (
            f"line {e['line']}, col {e['column']}"
            if e["line"] is not None and e["column"] is not None
            else f"line {e['line']}"
            if e["line"] is not None
            else "(no location)"
        )
        lines.append(f"- {loc}: {e['message']}")
    return "\n".join(lines)


def validate_with_retry(
    code: str,
    regenerate: Callable[[str, str], str],
    max_attempts: int = settings.MAX_REGENERATION_ATTEMPTS,
) -> ValidationResult:
    """Validate ``code``; on failure, call ``regenerate`` and revalidate.

    ``regenerate(previous_code, errors_str)`` is the LLM call that produces
    a new code attempt. The function is called at most ``max_attempts``
    times (default 2 per spec). The first validate also counts as an
    attempt, so a typical run is one initial validation + up to two
    regenerations = three validate calls total in the worst case.

    Returns the **final** :class:`ValidationResult` — valid if any attempt
    succeeded, invalid (with the latest attempt's errors) if all attempts
    failed. The caller can use ``result['code']`` to retrieve the version
    of the code that produced the final result.
    """
    result = validate_syntax(code)
    if result["valid"]:
        return result

    attempts = 0
    current_code = code
    current_errors = result["errors"]

    while attempts < max_attempts:
        attempts += 1
        errors_str = format_errors(current_errors)
        log.info(
            f"syntax_check: regen attempt {attempts}/{max_attempts} — "
            f"{len(current_errors)} error(s)"
        )

        try:
            new_code = regenerate(current_code, errors_str)
        except Exception as exc:  # whatever the LLM client raises
            log.error(f"regenerate callback failed (attempt {attempts}): {exc}")
            # Stop retrying — we can't recover without the callback.
            break

        result = validate_syntax(new_code)
        current_code = new_code
        current_errors = result["errors"]
        if result["valid"]:
            log.info(f"syntax_check: regenerated code is valid (attempt {attempts})")
            return result

    log.warning(f"syntax_check: all {attempts} regen attempts failed")
    return result
