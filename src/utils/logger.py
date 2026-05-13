"""Loguru-based logger factory.

Usage:

    from src.utils.logger import setup_logger
    log = setup_logger(__name__)
    log.info("hello")

The first call configures loguru globally (console sink + rotating file sink at
``logs/app.log``). Subsequent calls just return a bound logger with the caller's
module name attached as an extra field, so log lines are easy to filter.
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from config import settings

_CONFIGURED: bool = False

_CONSOLE_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{extra[module]}</cyan> "
    "<level>{message}</level>"
)

_FILE_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[module]} | {name}:{function}:{line} - {message}"
)


def _configure() -> None:
    """Install console + file sinks. Called once, idempotently.

    The console sink should never fail. The file sink may fail on permission
    or disk errors — we catch, print a warning to stderr, and continue with
    console-only logging rather than crashing the whole pipeline.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=_CONSOLE_FORMAT,
        colorize=True,
        enqueue=False,
    )

    file_path = settings.LOGS_DIR / "app.log"
    try:
        logger.add(
            file_path,
            level=settings.LOG_LEVEL,
            format=_FILE_FORMAT,
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )
    except (OSError, PermissionError) as exc:
        # Console-only fallback — log it to stderr via the console sink, which is already installed.
        logger.bind(module="utils.logger").warning(
            f"file log sink disabled: cannot open {file_path}: {exc}"
        )

    logger.configure(extra={"module": "app"})
    _CONFIGURED = True


def setup_logger(module_name: str) -> Any:
    """Return a logger bound to ``module_name``.

    First call configures global sinks; later calls are cheap.
    """
    _configure()
    return logger.bind(module=module_name)
