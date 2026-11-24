"""Shared fixtures: an empty SQLite database and a small synthetic network."""

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from mobi import db

START = datetime(2026, 11, 2, 8, 0)
STATIONS = 20


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    """Points the project at a fresh SQLite file with all tables created."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(db, "_engine", None)
    db.init_db()
    yield db.get_engine()
    db.get_engine().dispose()


def station_rows() -> list[dict]:
    """Returns STATIONS stations 100 m apart, capacity 20, all Downtown."""
    return [
        {
            "station_id": f"s{i:02d}",
            "name": f"Station {i}",
            "lat": 49.28 + i * 0.001,
            "lon": -123.12,
            "capacity": 20,
            "area": "Downtown",
            "updated_at": START,
        }
        for i in range(STATIONS)
    ]


def status_rows(start: datetime, end: datetime, seed: int = 0) -> list[dict]:
    """Returns 10-minute snapshots following a daily cycle plus noise."""
    rng = np.random.default_rng(seed)
    rows, time = [], start
    while time <= end:
        for i in range(STATIONS):
            phase = 2 * math.pi * (time.hour + time.minute / 60) / 24 + i
            bikes = int(np.clip(round(10 + 7 * math.sin(phase) + rng.normal(0, 1)), 0, 20))
            rows.append(
                {
                    "station_id": f"s{i:02d}",
                    "snapshot_time": time,
                    "num_bikes": bikes,
                    "num_docks": 20 - bikes,
                    "is_renting": True,
                    "is_returning": True,
                    "fetched_at": time,
                }
            )
        time += timedelta(minutes=10)
    return rows


@pytest.fixture
def network(sqlite_db):
    """Fills the database with two days of synthetic snapshots and weather."""
    end = START + timedelta(days=2)
    with sqlite_db.begin() as conn:
        conn.execute(db.stations.insert(), station_rows())
        conn.execute(db.station_status.insert(), status_rows(START, end))
        conn.execute(
            db.weather_hourly.insert(),
            [
                {"time": START + timedelta(hours=h), "temperature": 8.0,
                 "precipitation": 0.0, "wind_speed": 10.0, "fetched_at": end}
                for h in range(24 * 3)
            ],
        )
    return end
