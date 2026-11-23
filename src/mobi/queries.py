"""Read-only queries that feed the dashboard.

Every function here only reads. Results are cached for a short time so that
many viewers refreshing at once do not each hit the database.
"""

import time
from datetime import timedelta
from functools import lru_cache

import pandas as pd
from sqlalchemy import func, select

from mobi import features
from mobi.db import get_engine, model_runs, predictions_log, station_status
from mobi.ingest import utcnow

CACHE_SECONDS = 30
LOCAL_TZ = features.LOCAL_TZ


def to_local(values: pd.Series) -> pd.Series:
    """Converts naive UTC timestamps to Vancouver local time for display.

    Args:
        values: A datetime series holding naive UTC values.

    Returns:
        The same instants as timezone-aware Vancouver times.
    """
    return values.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)


def _time_bucket() -> int:
    """Returns a number that changes every CACHE_SECONDS (the cache key)."""
    return int(time.time() // CACHE_SECONDS)


@lru_cache(maxsize=1)
def _overview(bucket: int) -> pd.DataFrame:
    """Builds the live overview for one cache bucket (see overview)."""
    engine = get_engine()
    with engine.connect() as conn:
        latest = conn.execute(select(func.max(station_status.c.snapshot_time))).scalar()
        base_time = conn.execute(select(func.max(predictions_log.c.base_time))).scalar()
    if latest is None:
        return pd.DataFrame()

    status = features.read_query(
        select(
            station_status.c.station_id,
            station_status.c.num_bikes,
            station_status.c.num_docks,
        ).where(station_status.c.snapshot_time == latest)
    )
    frame = features.load_stations().merge(status, on="station_id", how="inner")
    if base_time is not None:
        forecasts = features.read_query(
            select(
                predictions_log.c.station_id,
                predictions_log.c.target_time,
                predictions_log.c.model_version,
                predictions_log.c.pred_baseline,
                predictions_log.c.pred_model,
                predictions_log.c.pred_corrected,
            ).where(predictions_log.c.base_time == base_time)
        )
        frame = frame.merge(forecasts, on="station_id", how="left")
    else:
        for column in ("pred_baseline", "pred_model", "pred_corrected"):
            frame[column] = float("nan")
    frame["snapshot_time"] = latest
    capacity = frame["capacity"].where(frame["capacity"] > 0)
    frame["fill_ratio"] = frame["num_bikes"] / capacity
    return frame


def overview() -> pd.DataFrame:
    """Returns the latest state and forecast of every station.

    Returns:
        One row per station with its metadata, current num_bikes / num_docks /
        fill_ratio, snapshot_time, and the newest forecasts (pred_baseline,
        pred_model, pred_corrected, target_time). Empty if nothing is stored.
    """
    return _overview(_time_bucket()).copy()


@lru_cache(maxsize=1)
def _accuracy(bucket: int) -> pd.DataFrame:
    """Computes hourly forecast errors for one cache bucket (see accuracy)."""
    scored = features.read_query(
        select(
            predictions_log.c.target_time,
            predictions_log.c.actual,
            predictions_log.c.pred_baseline,
            predictions_log.c.pred_model,
            predictions_log.c.pred_corrected,
        ).where(predictions_log.c.actual.is_not(None))
    )
    if scored.empty:
        return pd.DataFrame()
    scored["hour"] = pd.to_datetime(scored["target_time"]).dt.floor("h")
    for name in ("baseline", "model", "corrected"):
        scored[name] = (scored["actual"] - scored[f"pred_{name}"]).abs()
    hourly = (
        scored.groupby("hour")[["baseline", "model", "corrected"]]
        .mean()
        .join(scored.groupby("hour").size().rename("n"))
        .reset_index()
    )
    hourly["hour"] = to_local(hourly["hour"])
    return hourly


def accuracy() -> pd.DataFrame:
    """Returns the mean absolute error of each forecast type, per hour.

    Returns:
        Columns hour (Vancouver time), baseline, model, corrected (mean
        absolute error in bikes) and n (forecasts scored). Empty until the
        first forecasts have been compared with reality.
    """
    return _accuracy(_time_bucket()).copy()


def model_history() -> pd.DataFrame:
    """Returns the most recent model training runs.

    Returns:
        Up to 20 rows of model_runs, newest first, with trained_at in Vancouver
        time. Empty if no model has been trained yet.
    """
    runs = features.read_query(
        select(model_runs).order_by(model_runs.c.trained_at.desc()).limit(20)
    )
    if runs.empty:
        return runs
    runs["trained_at"] = to_local(pd.to_datetime(runs["trained_at"]))
    return runs


def station_history(station_id: str, hours: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns one station's recent bike counts and its forecasts.

    Args:
        station_id: The station to inspect.
        hours: How far back to look.

    Returns:
        A tuple (history, forecasts). history has snapshot_time and num_bikes;
        forecasts has target_time, pred_baseline, pred_corrected and actual.
        Times are in Vancouver local time.
    """
    since = utcnow() - timedelta(hours=hours)
    history = features.read_query(
        select(station_status.c.snapshot_time, station_status.c.num_bikes)
        .where(station_status.c.station_id == station_id)
        .where(station_status.c.snapshot_time >= since)
        .order_by(station_status.c.snapshot_time)
    )
    forecasts = features.read_query(
        select(
            predictions_log.c.target_time,
            predictions_log.c.pred_baseline,
            predictions_log.c.pred_corrected,
            predictions_log.c.actual,
        )
        .where(predictions_log.c.station_id == station_id)
        .where(predictions_log.c.target_time >= since)
        .order_by(predictions_log.c.target_time)
    )
    for frame, column in ((history, "snapshot_time"), (forecasts, "target_time")):
        if not frame.empty:
            frame[column] = to_local(pd.to_datetime(frame[column]))
    return history, forecasts
