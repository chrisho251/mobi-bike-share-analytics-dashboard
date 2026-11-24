"""Read-only queries that feed the dashboard.

Every function only reads. Results are cached for a short time, so many viewers
refreshing at once do not each hit the database; slow-moving analysis queries
are cached longer than live monitoring ones.
"""

import time
from datetime import timedelta
from functools import lru_cache, wraps

import pandas as pd
from sqlalchemy import case, func, select

from mobi import features
from mobi.db import get_engine, model_runs, predictions_log, station_status, stations, weather_hourly

LOCAL_TZ = features.LOCAL_TZ


def ttl_cache(seconds: int):
    """Caches a DataFrame-returning function for a number of seconds.

    Args:
        seconds: How long a result stays valid.

    Returns:
        A decorator. The wrapped function returns a copy, so callers can
        modify the result without corrupting the cache.
    """

    def decorator(function):
        @lru_cache(maxsize=16)
        def cached(bucket, *args):
            return function(*args)

        @wraps(function)
        def wrapper(*args):
            return cached(int(time.time() // seconds), *args).copy()

        return wrapper

    return decorator


def to_local(values: pd.Series) -> pd.Series:
    """Converts naive UTC timestamps to Vancouver local time for display.

    Args:
        values: A series of naive UTC datetimes.

    Returns:
        The same instants as timezone-aware Vancouver times.
    """
    return pd.to_datetime(values).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)


def latest_snapshot_time():
    """Returns the newest snapshot time (naive UTC), or None if empty."""
    with get_engine().connect() as conn:
        return conn.execute(select(func.max(station_status.c.snapshot_time))).scalar()


def _since(hours: float):
    """Returns the naive UTC time `hours` before the newest snapshot.

    Windows are anchored on the newest data rather than the clock, so a paused
    collector (or a replayed simulation) still shows its last window.
    """
    latest = latest_snapshot_time()
    return None if latest is None else latest - timedelta(hours=hours)


@ttl_cache(30)
def overview() -> pd.DataFrame:
    """Returns the latest state and forecast of every station.

    Returns:
        One row per station with metadata, num_bikes, num_docks, fill_ratio,
        snapshot_time and the newest forecasts (pred_baseline, pred_model,
        pred_corrected, target_time). Empty if nothing is stored.
    """
    latest = latest_snapshot_time()
    if latest is None:
        return pd.DataFrame()
    with get_engine().connect() as conn:
        base_time = conn.execute(select(func.max(predictions_log.c.base_time))).scalar()

    status = features.read_query(
        select(station_status.c.station_id, station_status.c.num_bikes, station_status.c.num_docks)
        .where(station_status.c.snapshot_time == latest)
    )
    frame = features.load_stations().merge(status, on="station_id", how="inner")
    forecasts = features.read_query(
        select(
            predictions_log.c.station_id,
            predictions_log.c.target_time,
            predictions_log.c.pred_baseline,
            predictions_log.c.pred_model,
            predictions_log.c.pred_corrected,
        ).where(predictions_log.c.base_time == base_time)
    )
    frame = frame.merge(forecasts, on="station_id", how="left")
    frame["snapshot_time"] = latest
    frame["fill_ratio"] = frame["num_bikes"] / frame["capacity"].where(frame["capacity"] > 0)
    return frame


@ttl_cache(60)
def network_timeline(hours: int = 24) -> pd.DataFrame:
    """Counts empty and full stations at every snapshot.

    Args:
        hours: Length of the window, ending at the newest snapshot.

    Returns:
        Columns time (Vancouver), empty, full and bikes.
    """
    since = _since(hours)
    if since is None:
        return pd.DataFrame()
    c = station_status.c
    frame = features.read_query(
        select(
            c.snapshot_time.label("time"),
            func.sum(case((c.num_bikes == 0, 1), else_=0)).label("empty"),
            func.sum(case((c.num_docks == 0, 1), else_=0)).label("full"),
            func.sum(c.num_bikes).label("bikes"),
        )
        .where(c.snapshot_time >= since)
        .group_by(c.snapshot_time)
        .order_by(c.snapshot_time)
    )
    frame["time"] = to_local(frame["time"])
    return frame


def _scored_forecasts(hours: int) -> pd.DataFrame:
    """Reads forecasts with a known outcome whose target time is recent."""
    since = _since(hours)
    if since is None:
        return pd.DataFrame()
    p = predictions_log.c
    frame = features.read_query(
        select(p.station_id, p.target_time, p.actual, p.pred_baseline, p.pred_model, p.pred_corrected)
        .where(p.actual.is_not(None), p.target_time >= since)
    )
    for name in ("baseline", "model", "corrected"):
        frame[name] = (frame["actual"] - frame[f"pred_{name}"]).abs()
    return frame


@ttl_cache(300)
def accuracy(hours: int = 24 * 7) -> pd.DataFrame:
    """Returns the mean absolute error of each forecast type, per hour.

    Args:
        hours: How far back to look, by forecast target time.

    Returns:
        Columns hour (Vancouver), baseline, model, corrected (mean absolute
        error in bikes) and n (forecasts scored). Empty before any scoring.
    """
    scored = _scored_forecasts(hours)
    if scored.empty:
        return pd.DataFrame()
    scored["hour"] = pd.to_datetime(scored["target_time"]).dt.floor("h")
    grouped = scored.groupby("hour")
    hourly = grouped[["baseline", "model", "corrected"]].mean().join(grouped.size().rename("n"))
    hourly = hourly.reset_index()
    hourly["hour"] = to_local(hourly["hour"])
    return hourly


@ttl_cache(300)
def accuracy_by_area(hours: int = 24) -> pd.DataFrame:
    """Returns the mean absolute error of each forecast type, per area.

    Args:
        hours: How far back to look, by forecast target time.

    Returns:
        Columns area, baseline, model, corrected. Empty before any scoring.
    """
    scored = _scored_forecasts(hours)
    if scored.empty:
        return pd.DataFrame()
    scored = scored.merge(features.load_stations()[["station_id", "area"]], on="station_id")
    return scored.groupby("area")[["baseline", "model", "corrected"]].mean().reset_index()


@ttl_cache(300)
def model_history(limit: int = 200) -> pd.DataFrame:
    """Returns the most recent training runs, oldest first.

    Args:
        limit: Maximum number of runs.

    Returns:
        model_runs rows with trained_at in Vancouver time.
    """
    runs = features.read_query(
        select(model_runs).order_by(model_runs.c.trained_at.desc()).limit(limit)
    )
    if not runs.empty:
        runs["trained_at"] = to_local(runs["trained_at"])
    return runs.iloc[::-1]


def _recent_status(days: int) -> pd.DataFrame:
    """Reads snapshots of the last `days` days with each station's area and capacity."""
    since = _since(24 * days)
    if since is None:
        return pd.DataFrame()
    status = features.load_status(since=since)
    stations = features.load_stations()[["station_id", "area", "capacity"]]
    frame = status.merge(stations, on="station_id")
    local = to_local(frame["snapshot_time"])
    frame["local_hour"] = local.dt.hour
    frame["weekend"] = local.dt.day_of_week >= 5
    return frame


@ttl_cache(600)
def fill_by_area_hour(days: int = 7) -> pd.DataFrame:
    """Returns the average fill level by area and local hour of day.

    Args:
        days: How many days of history to average.

    Returns:
        A frame indexed by area with one column per hour (0-23), values in %.
    """
    frame = _recent_status(days)
    if frame.empty:
        return frame
    frame["fill"] = frame["num_bikes"] / frame["capacity"] * 100
    return frame.pivot_table(index="area", columns="local_hour", values="fill", aggfunc="mean")


@ttl_cache(600)
def activity_by_hour(days: int = 7) -> pd.DataFrame:
    """Measures how busy the network is by hour, in dry versus rainy hours.

    Activity is the average absolute change in bikes per station between
    consecutive snapshots, a proxy for trips since the feed has no trip data.

    Args:
        days: How many days of history to use.

    Returns:
        Columns local_hour, weather ("Dry" or "Rain") and activity.
    """
    frame = _recent_status(days)
    if frame.empty:
        return frame
    frame = frame.sort_values(["station_id", "snapshot_time"])
    frame["change"] = frame.groupby("station_id")["num_bikes"].diff().abs()
    frame["hour"] = frame["snapshot_time"].dt.floor("h")
    rain = features.load_weather()[["time", "precipitation"]].rename(columns={"time": "hour"})
    frame = frame.merge(rain, on="hour", how="left")
    frame["weather"] = (frame["precipitation"] > 0).map({True: "Rain", False: "Dry"})
    return (
        frame.dropna(subset=["change"])
        .groupby(["local_hour", "weather"])["change"]
        .mean()
        .rename("activity")
        .reset_index()
    )


@ttl_cache(300)
def daily_completeness(days: int = 7) -> pd.DataFrame:
    """Measures how much of each day's expected data was collected.

    A full day has 144 snapshots (one every 10 minutes). For the current day
    only the slots up to the newest snapshot are expected.

    Args:
        days: Number of days to show, ending with the current one (fewer if
            collection started more recently).

    Returns:
        Columns day (Vancouver date), snapshots, expected and percent.
    """
    latest = latest_snapshot_time()
    if latest is None:
        return pd.DataFrame()
    times = features.read_query(
        select(station_status.c.snapshot_time)
        .where(station_status.c.snapshot_time >= latest - timedelta(days=days))
        .distinct()
    )
    local = to_local(times["snapshot_time"])
    counts = local.dt.date.value_counts().sort_index()
    latest_local = _local_timestamp(latest)
    # Days before collection started would read as failures, so start there.
    first_day = max(latest_local.date() - timedelta(days=days - 1), counts.index.min())
    frame = pd.DataFrame({"day": pd.date_range(first_day, latest_local.date(), freq="D").date})
    frame["snapshots"] = frame["day"].map(counts).fillna(0).astype(int)
    midnight = latest_local.normalize()
    elapsed_slots = int((latest_local - midnight) / timedelta(minutes=10)) + 1
    frame["expected"] = [elapsed_slots if d == latest_local.date() else 144 for d in frame["day"]]
    frame["percent"] = (frame["snapshots"] / frame["expected"] * 100).clip(upper=100)
    return frame


def _local_timestamp(value) -> pd.Timestamp:
    """Converts one naive UTC timestamp to Vancouver time."""
    return pd.Timestamp(value).tz_localize("UTC").tz_convert(LOCAL_TZ)


@ttl_cache(60)
def stations_per_snapshot(hours: int = 48) -> pd.DataFrame:
    """Counts the stations present in each snapshot.

    Args:
        hours: Length of the window, ending at the newest snapshot.

    Returns:
        Columns time (Vancouver) and stations.
    """
    since = _since(hours)
    if since is None:
        return pd.DataFrame()
    c = station_status.c
    frame = features.read_query(
        select(c.snapshot_time.label("time"), func.count().label("stations"))
        .where(c.snapshot_time >= since)
        .group_by(c.snapshot_time)
        .order_by(c.snapshot_time)
    )
    frame["time"] = to_local(frame["time"])
    return frame


@ttl_cache(60)
def pipeline_status() -> pd.DataFrame:
    """Summarises the freshness and completeness of every pipeline output.

    Returns:
        One row with last_snapshot, snapshots_24h, stations_reporting,
        stations_total, last_weather_fetch, last_model_run, last_model_mae,
        last_baseline_mae and overdue_forecasts (forecasts whose target passed
        more than 15 minutes ago without an actual).
    """
    latest = latest_snapshot_time()
    s, p = station_status.c, predictions_log.c
    with get_engine().connect() as conn:
        last_weather = conn.execute(select(func.max(weather_hourly.c.fetched_at))).scalar()
        last_run = conn.execute(select(model_runs).order_by(model_runs.c.trained_at.desc()).limit(1)).first()
        stations_total = conn.execute(select(func.count()).select_from(stations)).scalar()
        snapshots_24h = reporting = overdue = 0
        if latest is not None:
            snapshots_24h = conn.execute(
                select(func.count(s.snapshot_time.distinct())).where(s.snapshot_time > latest - timedelta(hours=24))
            ).scalar()
            reporting = conn.execute(
                select(func.count()).select_from(station_status).where(s.snapshot_time == latest)
            ).scalar()
            overdue = conn.execute(
                select(func.count()).select_from(predictions_log).where(
                    p.actual.is_(None),
                    p.target_time < latest - timedelta(minutes=15),
                    p.target_time > latest - timedelta(hours=12),
                )
            ).scalar()
    return pd.DataFrame(
        [
            {
                "last_snapshot": latest,
                "snapshots_24h": snapshots_24h,
                "stations_reporting": reporting,
                "stations_total": stations_total,
                "last_weather_fetch": last_weather,
                "last_model_run": last_run.trained_at if last_run else None,
                "last_model_mae": last_run.mae_model if last_run else None,
                "last_baseline_mae": last_run.mae_baseline if last_run else None,
                "overdue_forecasts": overdue,
            }
        ]
    )


def station_history(station_id: str, hours: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns one station's recent bike counts and its forecasts.

    Args:
        station_id: The station to inspect.
        hours: Length of the window, ending at the newest snapshot.

    Returns:
        A tuple (history, forecasts): history has time and num_bikes;
        forecasts has time, pred_baseline, pred_corrected and actual. Times are
        Vancouver local time.
    """
    since = _since(hours)
    if since is None:
        return pd.DataFrame(), pd.DataFrame()
    s, p = station_status.c, predictions_log.c
    history = features.read_query(
        select(s.snapshot_time.label("time"), s.num_bikes)
        .where(s.station_id == station_id, s.snapshot_time >= since)
        .order_by(s.snapshot_time)
    )
    forecasts = features.read_query(
        select(p.target_time.label("time"), p.pred_baseline, p.pred_corrected, p.actual)
        .where(p.station_id == station_id, p.target_time >= since)
        .order_by(p.target_time)
    )
    for frame in (history, forecasts):
        if not frame.empty:
            frame["time"] = to_local(frame["time"])
    return history, forecasts
