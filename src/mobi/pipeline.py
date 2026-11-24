"""Runs one full cycle: collect stations, bike status, weather, then forecast."""

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from mobi import config
from mobi.db import delete_before, init_db, latest_fetch_time, predictions_log, station_status
from mobi.forecast import TRAIN_WINDOW, fill_actuals, predict_next_hour
from mobi.ingest import ingest_stations, ingest_status, ingest_weather

logger = logging.getLogger("pipeline")

# Older rows are deleted every run so the database stops growing (each day adds
# about 37,600 snapshots and as many forecasts). Snapshots must cover the
# training window; forecasts feed the dashboard's 7-day views and the 24-hour
# per-station selection. Weather and training runs are small and kept.
STATUS_RETENTION = TRAIN_WINDOW + timedelta(days=2)
PREDICTION_RETENTION = timedelta(days=14)


def prune_history() -> int:
    """Deletes snapshots and forecasts older than their retention period.

    Ages are measured from the newest snapshot, not the clock, so nothing is
    deleted while collection is stopped.

    Returns:
        The number of rows deleted.
    """
    latest = latest_fetch_time(station_status, "snapshot_time")
    if latest is None:
        return 0
    return delete_before(
        station_status, "snapshot_time", latest - STATUS_RETENTION
    ) + delete_before(predictions_log, "target_time", latest - PREDICTION_RETENTION)


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
    data. The forecast steps run after collection because they read what was
    just stored, and old rows are pruned last.

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
        run_step("prune", prune_history),
    ]
    return all(critical)


def main() -> None:
    """Configures logging, runs the pipeline and sets the exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if config.is_simulation():
        # Live snapshots must never be mixed into the synthetic history.
        raise SystemExit(
            "DATABASE_URL is not set, so the project points at the simulation "
            "database. Set DATABASE_URL to collect live data, or run "
            "simulation/replay.py to forecast on simulated data."
        )
    if not run_pipeline():
        raise SystemExit(1)


if __name__ == "__main__":
    main()