"""Runs one full cycle: collect stations, bike status, weather, then forecast."""

import logging
from collections.abc import Callable
from typing import Any

from mobi.db import init_db
from mobi.forecast import fill_actuals, predict_next_hour
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
    """Runs every collection and forecasting step once.

    Weather is optional: if it fails, the failure is logged but the run is
    still considered successful, because the bike counts are the essential
    data. The forecast steps run last because they read what was just stored.

    Returns:
        True if every step except weather succeeded, False otherwise.
    """
    init_db()
    critical = [
        run_step("stations", ingest_stations),
        run_step("status", ingest_status),
    ]
    run_step("weather", ingest_weather)
    critical += [
        run_step("fill_actuals", fill_actuals),
        run_step("predict", predict_next_hour),
    ]
    return all(critical)


def main() -> None:
    """Configures logging, runs the pipeline and sets the exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not run_pipeline():
        raise SystemExit(1)


if __name__ == "__main__":
    main()