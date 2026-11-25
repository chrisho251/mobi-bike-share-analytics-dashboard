"""Turns raw station snapshots into model-ready features.

Every row of the output describes one station at one snapshot time (the
"base time") and holds everything known at that moment, plus the bike count
60 minutes later (the target) when that future snapshot already exists.
"""

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select

from mobi import config
from mobi.db import get_engine, station_status, stations, weather_hourly

# How far ahead we forecast.
HORIZON = timedelta(minutes=60)
# Past observations used as features, in minutes before the base time.
LAGS_MINUTES = (30, 60)
# A snapshot counts as "the value at time T" if it lies within this distance.
MATCH_TOLERANCE = timedelta(minutes=5)
# History needed to build the features of the newest snapshot (its lags).
LOOKBACK = timedelta(minutes=max(LAGS_MINUTES)) + MATCH_TOLERANCE
# Display and calendar features use the dispatcher's local time.
LOCAL_TZ = "America/Vancouver"
AREA_CODES = {name: code for code, name in enumerate(config.AREA_ANCHORS)}

FEATURE_COLUMNS = [
    "bikes_now",
    "capacity",
    "fill_ratio",
    "bikes_lag30",
    "bikes_lag60",
    "trend30",
    "trend60",
    "hour_frac",
    "day_of_week",
    "is_weekend",
    "temp_now",
    "rain_now",
    "wind_now",
    "temp_next",
    "rain_next",
    "wind_next",
    "lat",
    "lon",
    "area_code",
]


def _to_ns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Casts datetime columns to nanosecond precision.

    merge_asof refuses to join keys of different datetime precision, so every
    frame read from the database is normalised first.

    Args:
        frame: The frame to convert (modified in place).
        columns: Names of the datetime columns.

    Returns:
        The same frame, for chaining.
    """
    for column in columns:
        frame[column] = frame[column].astype("datetime64[ns]")
    return frame


def read_query(query) -> pd.DataFrame:
    """Runs a SQLAlchemy select and returns the result as a DataFrame.

    SQLAlchemy labels columns with its own string subclass, which scikit-learn
    rejects as a feature name, so the column names are converted to plain str.

    Args:
        query: A SQLAlchemy select statement.

    Returns:
        The query result with plain-string column names.
    """
    frame = pd.read_sql(query, get_engine())
    frame.columns = [str(column) for column in frame.columns]
    return frame


def load_status(since: datetime | None = None) -> pd.DataFrame:
    """Reads bike counts from the database.

    Args:
        since: If given, only snapshots at or after this UTC time are read.

    Returns:
        A frame with columns station_id, snapshot_time and num_bikes.
    """
    query = select(
        station_status.c.station_id,
        station_status.c.snapshot_time,
        station_status.c.num_bikes,
    )
    if since is not None:
        query = query.where(station_status.c.snapshot_time >= since)
    return _to_ns(read_query(query), ["snapshot_time"])


def load_stations() -> pd.DataFrame:
    """Reads station metadata from the database.

    Returns:
        A frame with columns station_id, name, lat, lon, capacity and area.
    """
    query = select(
        stations.c.station_id,
        stations.c.name,
        stations.c.lat,
        stations.c.lon,
        stations.c.capacity,
        stations.c.area,
    )
    return read_query(query)


def load_weather() -> pd.DataFrame:
    """Reads hourly weather from the database.

    Returns:
        A frame with columns time, temperature, precipitation and wind_speed.
    """
    query = select(
        weather_hourly.c.time,
        weather_hourly.c.temperature,
        weather_hourly.c.precipitation,
        weather_hourly.c.wind_speed,
    )
    return _to_ns(read_query(query), ["time"])


def _value_at_offset(
    base: pd.DataFrame, observations: pd.DataFrame, offset: timedelta
) -> pd.Series:
    """Looks up each station's bike count at base time plus an offset.

    Args:
        base: Rows with station_id and snapshot_time; its index must be unique.
        observations: Columns station_id, obs_time and value, sorted by
            obs_time.
        offset: Time to add to snapshot_time (negative looks into the past).

    Returns:
        A series aligned with base.index holding the nearest observation within
        MATCH_TOLERANCE, or NaN where none exists.
    """
    query = base[["station_id", "snapshot_time"]].copy()
    query["query_time"] = query["snapshot_time"] + offset
    query = query.sort_values("query_time")
    merged = pd.merge_asof(
        query,
        observations,
        left_on="query_time",
        right_on="obs_time",
        by="station_id",
        tolerance=MATCH_TOLERANCE,
        direction="nearest",
    )
    merged.index = query.index
    return merged["value"].reindex(base.index)


def build_features(
    status: pd.DataFrame, station_info: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Builds one feature row per station snapshot.

    Args:
        status: Bike counts (station_id, snapshot_time, num_bikes).
        station_info: Station metadata (station_id, lat, lon, capacity, area).
        weather: Hourly weather (time, temperature, precipitation, wind_speed).

    Returns:
        A frame with the FEATURE_COLUMNS plus station_id, snapshot_time and
        bikes_target. bikes_target is NaN while the snapshot 60 minutes later
        has not been collected yet.
    """
    frame = status.sort_values(["station_id", "snapshot_time"]).reset_index(drop=True)
    frame = frame.rename(columns={"num_bikes": "bikes_now"})

    observations = (
        frame[["station_id", "snapshot_time", "bikes_now"]]
        .rename(columns={"snapshot_time": "obs_time", "bikes_now": "value"})
        .sort_values("obs_time")
    )
    for minutes in LAGS_MINUTES:
        frame[f"bikes_lag{minutes}"] = _value_at_offset(
            frame, observations, -timedelta(minutes=minutes)
        )
    frame["bikes_target"] = _value_at_offset(frame, observations, HORIZON)

    frame = frame.merge(
        station_info[["station_id", "lat", "lon", "capacity", "area"]],
        on="station_id",
        how="left",
    )
    frame["fill_ratio"] = frame["bikes_now"] / frame["capacity"].where(
        frame["capacity"] > 0
    )
    frame["trend30"] = frame["bikes_now"] - frame["bikes_lag30"]
    frame["trend60"] = frame["bikes_now"] - frame["bikes_lag60"]
    frame["area_code"] = frame["area"].map(AREA_CODES).fillna(0).astype(int)

    local = frame["snapshot_time"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    frame["hour_frac"] = local.dt.hour + local.dt.minute / 60
    frame["day_of_week"] = local.dt.day_of_week
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)

    frame["base_hour"] = frame["snapshot_time"].dt.floor("h")
    frame["target_hour"] = (frame["snapshot_time"] + HORIZON).dt.floor("h")
    now_weather = weather.rename(
        columns={
            "time": "base_hour",
            "temperature": "temp_now",
            "precipitation": "rain_now",
            "wind_speed": "wind_now",
        }
    )
    next_weather = weather.rename(
        columns={
            "time": "target_hour",
            "temperature": "temp_next",
            "precipitation": "rain_next",
            "wind_speed": "wind_next",
        }
    )
    frame = frame.merge(now_weather, on="base_hour", how="left")
    frame = frame.merge(next_weather, on="target_hour", how="left")
    return frame
