"""Tests for feature building."""

from datetime import timedelta

import pandas as pd
from conftest import START

from mobi import features


def _status(counts: list[int]) -> pd.DataFrame:
    """One station with a bike count every 10 minutes."""
    times = [START + timedelta(minutes=10 * i) for i in range(len(counts))]
    frame = pd.DataFrame({"station_id": "a", "snapshot_time": times, "num_bikes": counts})
    return features._to_ns(frame, ["snapshot_time"])


def _build(counts: list[int]) -> pd.DataFrame:
    stations = pd.DataFrame(
        {"station_id": ["a"], "lat": [49.28], "lon": [-123.12], "capacity": [20], "area": ["Downtown"]}
    )
    weather = pd.DataFrame(
        {"time": [START], "temperature": [8.0], "precipitation": [0.0], "wind_speed": [5.0]}
    )
    return features.build_features(_status(counts), stations, features._to_ns(weather, ["time"]))


def test_target_is_count_60_minutes_later():
    frame = _build(list(range(10)))

    assert frame.loc[0, "bikes_target"] == 6
    assert pd.isna(frame.loc[4, "bikes_target"])  # 60 minutes later not collected yet


def test_lags_look_back_without_leaking():
    frame = _build(list(range(10)))

    assert frame.loc[6, "bikes_lag30"] == 3
    assert frame.loc[6, "bikes_lag60"] == 0
    assert frame.loc[6, "trend30"] == 3
    assert pd.isna(frame.loc[2, "bikes_lag30"])


def test_missing_snapshot_beyond_tolerance_gives_nan():
    counts = list(range(10))
    status = _status(counts).drop(index=6)  # no snapshot exactly 60 min after row 0
    stations = pd.DataFrame(
        {"station_id": ["a"], "lat": [49.28], "lon": [-123.12], "capacity": [20], "area": ["Downtown"]}
    )
    weather = features._to_ns(
        pd.DataFrame({"time": [START], "temperature": [8.0], "precipitation": [0.0], "wind_speed": [5.0]}),
        ["time"],
    )
    frame = features.build_features(status, stations, weather)

    assert pd.isna(frame.loc[0, "bikes_target"])
