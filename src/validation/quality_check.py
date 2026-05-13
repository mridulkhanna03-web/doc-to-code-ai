"""Code quality and security checks for generated Python.

Three concerns, three functions:

- :func:`run_ruff` — shell out to ``ruff check --output-format json`` and
  return the parsed issue list. Wrapped in try/except per project rule so
  a missing ruff binary or subprocess hiccup doesn't crash the pipeline.
- :func:`check_pep8` — derive a 0–100 PEP-8 compliance score from the ruff
  issue list. Pure compute.
- :func:`security_scan` — walk the AST looking for known-dangerous patterns
  (``os.system``, ``subprocess.*``, ``eval``, ``exec``, ``__import__``).
  We **flag, don't reject** — per the project spec the security check
  surfaces a list of warnings without failing the run.

:func:`quality_check` is the orchestrator that runs all three and returns a
single structured :class:`QualityReport`.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from src.utils.logger import setup_logger

log = setup_logger("validation.quality_check")

_RUFF_TIMEOUT_SECONDS: int = 15


def _find_ruff() -> str | None:
    """Locate the ``ruff`` binary.

    pip installs ruff as a binary at ``<venv>/bin/ruff`` (or ``Scripts\\ruff.exe``
    on Windows). Subprocess by default only searches the system PATH, which
    on a non-activated venv does not include the venv's bin directory — so
    we have to look there explicitly. Falls back to ``PATH`` lookup.
    """
    # 1. Sibling of the current Python interpreter (works for venv + system Python).
    py_dir = Path(sys.executable).parent
    candidate = py_dir / ("ruff.exe" if os.name == "nt" else "ruff")
    if candidate.is_file():
        return str(candidate)

    # 2. System PATH.
    found = shutil.which("ruff")
    if found:
        return found

    return None

# AST node patterns we flag. (function name, message)
_DANGEROUS_CALLS: dict[str, str] = {
    "eval": "dynamic code evaluation",
    "exec": "dynamic code execution",
    "compile": "compile + exec equivalent",
    "__import__": "dynamic import",
}

_DANGEROUS_ATTRS: dict[str, str] = {
    "os.system": "shell command execution",
    "os.popen": "shell command execution",
    "subprocess.run": "subprocess execution",
    "subprocess.call": "subprocess execution",
    "subprocess.Popen": "subprocess execution",
    "subprocess.check_output": "subprocess execution",
    "subprocess.check_call": "subprocess execution",
}


class RuffIssue(TypedDict):
    code: str            # e.g. "E501"
    message: str
    line: int | None
    column: int | None


class SecurityFlag(TypedDict):
    pattern: str         # e.g. "os.system"
    line: int | None
    note: str


class QualityReport(TypedDict):
    valid: bool          # ruff ran successfully (does NOT mean zero issues)
    errors: list[RuffIssue]      # ruff issues coded E* or F* (real bugs)
    warnings: list[RuffIssue]    # ruff issues coded W*/etc. (style only)
    security_flags: list[SecurityFlag]
    pep8_score: float    # 0-100; higher is better
    line_count: int


def run_ruff(code: str) -> tuple[bool, list[RuffIssue]]:
    """Run ``ruff check --output-format=json`` against ``code`` (via stdin).

    Returns ``(ok, issues)`` where ``ok`` is whether ruff ran successfully
    (regardless of whether it found issues). On any subprocess / decode
    failure, returns ``(False, [])`` with the cause logged.
    """
    ruff = _find_ruff()
    if ruff is None:
        log.error(
            "ruff binary not found in venv bin or system PATH. "
            "Install with: pip install ruff>=0.3"
        )
        return False, []

    try:
        proc = subprocess.run(
            [ruff, "check", "--output-format=json", "--stdin-filename", "generated.py", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=_RUFF_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        log.error(f"ruff vanished after path resolution: {exc}")
        return False, []
    except subprocess.TimeoutExpired:
        log.error(f"ruff exceeded {_RUFF_TIMEOUT_SECONDS}s timeout — skipping quality check")
        return False, []
    except OSError as exc:
        log.error(f"ruff subprocess failed: {exc}")
        return False, []

    # ruff returns 0 (no issues), 1 (issues found), or non-zero on internal error.
    if proc.returncode not in (0, 1):
        log.warning(f"ruff exited with code {proc.returncode}: {proc.stderr.strip()[:200]}")
        return False, []

    raw = proc.stdout.strip()
    if not raw:
        return True, []

    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error(f"ruff JSON parse failed: {exc} (raw stdout starts with: {raw[:120]!r})")
        return False, []

    issues: list[RuffIssue] = []
    for it in items:
        loc = it.get("location") or {}
        issues.append(
            RuffIssue(
                code=it.get("code") or "",
                message=it.get("message") or "",
                line=loc.get("row"),
                column=loc.get("column"),
            )
        )
    return True, issues


def check_pep8(issues: list[RuffIssue], line_count: int) -> float:
    """0-100 PEP-8 score.

    We count only E-prefix (PEP 8 errors) and W-prefix (PEP 8 warnings)
    against the score; F-prefix (pyflakes — real bugs) and others don't
    move PEP 8 specifically. ``score = 100 * (1 - pep8_violations / max(line_count, 1))``,
    clamped to ``[0, 100]``.
    """
    if line_count <= 0:
        return 0.0
    pep8 = sum(1 for i in issues if i["code"].startswith(("E", "W")))
    score = 100.0 * (1.0 - pep8 / line_count)
    return max(0.0, min(100.0, score))


def _attr_chain(node: ast.AST) -> str | None:
    """Return ``"a.b.c"`` for ``Attribute`` chains like ``os.path.join``.

    Returns ``None`` if the chain isn't all attribute access on names.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def security_scan(code: str) -> list[SecurityFlag]:
    """AST walk for known-dangerous patterns. Returns a list of flags;
    never raises — bad input becomes an empty list with a debug log.
    """
    if not code or not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        log.debug("security_scan: ast.parse failed; cannot scan invalid code")
        return []
    except Exception as exc:
        log.error(f"security_scan: unexpected ast.parse error: {exc}")
        return []

    flags: list[SecurityFlag] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # Plain-name call: eval(...), exec(...)
        if isinstance(func, ast.Name) and func.id in _DANGEROUS_CALLS:
            flags.append(SecurityFlag(
                pattern=f"{func.id}()",
                line=node.lineno,
                note=_DANGEROUS_CALLS[func.id],
            ))
            continue

        # Attribute call: os.system(...), subprocess.run(...)
        chain = _attr_chain(func)
        if chain and chain in _DANGEROUS_ATTRS:
            flags.append(SecurityFlag(
                pattern=f"{chain}()",
                line=node.lineno,
                note=_DANGEROUS_ATTRS[chain],
            ))
    return flags


def quality_check(code: str) -> QualityReport:
    """Run ruff + security scan and return a single :class:`QualityReport`."""
    line_count = len(code.splitlines()) if code else 0

    ok, issues = run_ruff(code)
    errors = [i for i in issues if i["code"].startswith(("F", "E9"))]  # F = pyflakes, E9 = syntax-ish
    warnings = [i for i in issues if not i["code"].startswith(("F", "E9"))]

    flags = security_scan(code)
    pep8 = check_pep8(issues, line_count)

    log.info(
        f"quality_check: ruff_ok={ok} issues={len(issues)} "
        f"(errors={len(errors)}, warnings={len(warnings)}) "
        f"security_flags={len(flags)} pep8={pep8:.1f}"
    )
    return QualityReport(
        valid=ok,
        errors=errors,
        warnings=warnings,
        security_flags=flags,
        pep8_score=pep8,
        line_count=line_count,
    )
