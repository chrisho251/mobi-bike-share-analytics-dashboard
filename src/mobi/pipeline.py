"""Runs one full data-collection cycle: stations, bike status and weather."""

import logging
from collections.abc import Callable
from typing import Any

from mobi.db import init_db
from mobi.ingest import ingest_stations, ingest_status, ingest_weather

logger = logging.getLogger("pipeline")


def run_step(name: str, step: Callable[[], Any]) -> bool:
    """Runs one pipeline step and logs the outcome without raising.

    A failing step must not stop the steps after it, so any exception is
    logged and swallowed.

    Args:
        name: A short label used in the log messages.
        step: A function that takes no arguments and does the work.

    Returns:
        True if the step finished, False if it raised an exception.
    """
    try:
        result = step()
    except Exception:
        logger.exception("%s failed", name)
        return False
    logger.info("%s: %s rows", name, result)
    return True


def run_pipeline() -> bool:
    """Runs every collection step once.

    Returns:
        True if all steps succeeded, False if at least one failed.
    """
    init_db()
    results = [
        run_step("stations", ingest_stations),
        run_step("status", ingest_status),
        run_step("weather", ingest_weather),
    ]
    return all(results)


def main() -> None:
    """Configures logging, runs the pipeline and sets the exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not run_pipeline():
        raise SystemExit(1)


if __name__ == "__main__":
    main()