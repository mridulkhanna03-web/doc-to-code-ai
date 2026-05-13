"""End-to-end RAG query CLI — the Day 2 milestone.

Pipeline:

    query  →  (optional) Gemini rewrite  →  retrieve k chunks
           →  build code prompt  →  Ollama / Gemini generate
           →  parse response into {code, explanation, sources}
           →  AST syntax check (up to 2 regen retries)
           →  ruff + security scan
           →  print structured summary

Usage:

    python scripts/query_cli.py "How to create a basic FastAPI application?"
    python scripts/query_cli.py "How to handle file uploads?" --k 3 --rewrite-query
    python scripts/query_cli.py "..."--json   # machine-readable output

Exit codes:
    0  — query succeeded and code is AST-valid
    1  — query ran but produced invalid code after all regen attempts
    2  — pipeline failure (no chunks, LLM error, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.generation.llm_client import LLMError, generate, rewrite_query  # noqa: E402
from src.generation.parser import parse_code_response  # noqa: E402
from src.generation.prompts import build_code_prompt, build_regeneration_prompt  # noqa: E402
from src.retrieval.retriever import retrieve_context  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402
from src.validation.quality_check import quality_check  # noqa: E402
from src.validation.syntax_check import format_errors, validate_with_retry  # noqa: E402

log = setup_logger("scripts.query_cli")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("query", help="Developer's question, in natural language")
    p.add_argument("--k", type=int, default=settings.K_RESULTS,
                   help=f"top-k chunks to retrieve (default: {settings.K_RESULTS})")
    p.add_argument("--rewrite-query", action="store_true",
                   help="rewrite the query via Gemini before retrieval (Query2Doc-style)")
    p.add_argument("--no-validate", action="store_true",
                   help="skip AST + ruff validation (faster, but no retry on broken code)")
    p.add_argument("--json", action="store_true", dest="json_out",
                   help="print machine-readable JSON instead of human-readable text")
    return p.parse_args()


def _print_human(payload: dict) -> None:
    """Pretty-print the result for a human reading the terminal."""
    print()
    print("=" * 78)
    print(f"QUERY: {payload['query']}")
    if payload.get("rewritten_query"):
        print(f"REWRITTEN: {payload['rewritten_query']}")
    print("=" * 78)

    # Retrieval
    hits = payload["retrieved"]
    print(f"\n--- RETRIEVED {len(hits)} chunk(s) ---")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] dist={h['distance']:.3f}  {h['source_doc']} / {h['section']}")

    # Code
    print(f"\n--- GENERATED CODE  (backend={payload['backend']}) ---")
    print(payload["code"] or "(no code block in response)")

    # Explanation
    if payload.get("explanation"):
        print(f"\n--- EXPLANATION ---")
        print(payload["explanation"])

    # Validation
    val = payload["validation"]
    qc = payload["quality"]
    print(f"\n--- VALIDATION ---")
    print(f"  syntax valid : {val['valid']}  (regen attempts={val.get('regen_attempts', 0)})")
    if not val["valid"]:
        print(f"  syntax errors:")
        for e in val["errors"]:
            loc = f"line {e['line']}" if e.get("line") else "(no loc)"
            print(f"    - {loc}: {e['message']}")
    print(f"  ruff ran ok  : {qc['valid']}")
    print(f"  ruff errors  : {len(qc['errors'])}")
    print(f"  ruff warnings: {len(qc['warnings'])}")
    print(f"  pep8 score   : {qc['pep8_score']:.1f}")
    if qc["security_flags"]:
        print(f"  security flags:")
        for f in qc["security_flags"]:
            print(f"    - line {f['line']}: {f['pattern']}  ({f['note']})")

    print(f"\n--- SOURCES ---")
    print(f"  cited by LLM: {payload['sources']}")

    # Timing
    t = payload["timing"]
    print(f"\n--- TIMING ---")
    for k, v in t.items():
        print(f"  {k:>14}: {v:.2f}s")


def main() -> int:
    args = _parse_args()
    timing: dict[str, float] = {}

    # 1. Optional query rewrite (Gemini)
    t0 = time.time()
    used_query = args.query
    rewritten = None
    if args.rewrite_query:
        rewritten = rewrite_query(args.query)
        if rewritten and rewritten != args.query:
            used_query = rewritten
    timing["rewrite"] = time.time() - t0

    # 2. Retrieve
    t0 = time.time()
    context, hits = retrieve_context(used_query, k=args.k)
    timing["retrieve"] = time.time() - t0
    if not hits:
        log.error("retrieval returned 0 chunks — aborting")
        return 2

    # 3. Generate via LLM
    t0 = time.time()
    system, user = build_code_prompt(context, args.query)
    try:
        text, backend = generate(system, user)
    except LLMError as exc:
        log.error(f"LLM generation failed: {exc}")
        return 2
    timing["generate"] = time.time() - t0

    # 4. Parse
    parsed = parse_code_response(text)

    # 5. Validate (syntax + quality), with regen loop for syntax
    val_result: dict = {"valid": True, "errors": [], "code": parsed["code"], "regen_attempts": 0}
    qc_result: dict = {"valid": True, "errors": [], "warnings": [], "security_flags": [], "pep8_score": 100.0, "line_count": 0}

    if not args.no_validate and parsed["code"]:
        t0 = time.time()
        regen_count = {"n": 0}

        def regen(prev_code: str, errors_str: str) -> str:
            regen_count["n"] += 1
            log.info(f"regenerating after syntax error (attempt {regen_count['n']})")
            user_regen = build_regeneration_prompt(prev_code, errors_str)
            new_text, _ = generate(system, user_regen)
            new_parsed = parse_code_response(new_text)
            return new_parsed["code"]

        val_result = dict(validate_with_retry(parsed["code"], regenerate=regen))
        val_result["regen_attempts"] = regen_count["n"]
        timing["validate_syntax"] = time.time() - t0

        # Run quality check on the final (possibly regenerated) code
        t0 = time.time()
        qc_result = dict(quality_check(val_result["code"]))
        timing["quality_check"] = time.time() - t0

    # 6. Assemble payload
    payload = {
        "query": args.query,
        "rewritten_query": rewritten,
        "backend": backend,
        "retrieved": [
            {
                "source_doc": h["metadata"].get("source_doc", "?"),
                "section": h["metadata"].get("section", "?"),
                "distance": h["distance"],
                "id": h["id"],
            }
            for h in hits
        ],
        "code": val_result["code"],
        "explanation": parsed["explanation"],
        "sources": parsed["sources"],
        "validation": val_result,
        "quality": qc_result,
        "timing": timing,
    }

    # 7. Output
    if args.json_out:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_human(payload)

    # 8. Exit code
    return 0 if val_result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
