"""Download FastAPI's documentation source by shallow-cloning the upstream repo.

Run this once before ``scripts/ingest_docs.py``. It is idempotent — if the
``data/raw_docs/fastapi`` directory already exists, the script just reports
what's there instead of re-cloning.

Why a git clone instead of HTML scraping?

* FastAPI's docs site is generated from markdown in ``docs/en/docs/`` in the
  upstream repo. Pulling the markdown directly gives us cleaner, code-block-
  preserving input than scraping the rendered HTML.
* A shallow clone (``--depth=1``) is ~10 MB and finishes in seconds.
* Idempotent re-runs let us safely re-execute the full pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Allow `python scripts/download_docs.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

log = setup_logger("scripts.download_docs")

REPO_URL: str = "https://github.com/tiangolo/fastapi.git"
CLONE_DIR: Path = settings.RAW_DOCS_DIR / "fastapi"
DOCS_GLOB: str = "docs/en/docs/**/*.md"


def _git_available() -> bool:
    return shutil.which("git") is not None


def _shallow_clone(url: str, dest: Path) -> None:
    log.info(f"git clone --depth=1 {url} -> {dest}")
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--single-branch", url, str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log.error(f"git clone failed (exit {result.returncode}): {result.stderr.strip()}")
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    log.info("clone complete")


def _summarise(clone_dir: Path) -> tuple[int, int]:
    """Return ``(file_count, total_chars)`` for the docs glob.

    Per-file reads are wrapped: a single unreadable doc is logged and skipped
    rather than aborting the whole summary.
    """
    files = sorted(clone_dir.glob(DOCS_GLOB))
    total_chars = 0
    for f in files:
        try:
            total_chars += len(f.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            try:
                total_chars += len(f.read_text(encoding="latin-1"))
            except OSError as exc:
                log.warning(f"latin-1 fallback failed for {f}: {exc}")
        except OSError as exc:
            log.warning(f"could not read {f}: {exc}")
    return len(files), total_chars


def main() -> int:
    if not _git_available():
        log.error("`git` is not on PATH — install git and re-run")
        return 2

    settings.RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if CLONE_DIR.exists():
        log.info(f"{CLONE_DIR} already exists — skipping clone (delete the directory to re-download)")
    else:
        try:
            _shallow_clone(REPO_URL, CLONE_DIR)
        except RuntimeError:
            return 1

    file_count, total_chars = _summarise(CLONE_DIR)
    log.info(
        f"FastAPI docs ready at {CLONE_DIR}: "
        f"{file_count} markdown files, ~{total_chars:,} chars "
        f"(~{total_chars // 4:,} tokens at 4 chars/token)"
    )
    if file_count == 0:
        log.error(f"no markdown files matched glob '{DOCS_GLOB}' under {CLONE_DIR}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
