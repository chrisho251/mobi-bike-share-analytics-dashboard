"""Tests for e-bike collection and hours empty or full."""

from datetime import timedelta

import pytest
from conftest import START, station_rows
from sqlalchemy import create_engine, inspect, select, text

from mobi import db, ingest, queries


def _feed_station(types):
    return {
        "station_id": "0008",
        "num_bikes_available": 11,
        "num_docks_available": 3,
        "is_renting": True,
        "is_returning": True,
        "is_installed": True,
        **({} if types is None else {"vehicle_types_available": types}),
    }


TYPES = [
    {"vehicle_type_id": "1", "count": 9},
    {"vehicle_type_id": "2", "count": 2},
    {"vehicle_type_id": "3", "count": 0},
]


def test_ebikes_are_counted_from_electric_vehicle_types():
    row = ingest.status_row(_feed_station(TYPES), START, START, {"2", "3"})

    assert row["num_bikes"] == 11
    assert row["num_ebikes"] == 2


@pytest.mark.parametrize("types, electric", [(TYPES, None), (None, {"2", "3"})])
def test_ebikes_are_unknown_without_types(types, electric):
    assert ingest.status_row(_feed_station(types), START, START, electric)["num_ebikes"] is None


def test_electric_types_exclude_human_powered(monkeypatch):
    feed = {
        "data": {
            "vehicle_types": [
                {"vehicle_type_id": "1", "propulsion_type": "human"},
                {"vehicle_type_id": "2", "propulsion_type": "electric_assist"},
            ]
        }
    }
    monkeypatch.setattr(ingest, "fetch_json", lambda url: feed)

    assert ingest.electric_type_ids() == {"2"}


def test_init_db_adds_new_columns_to_an_existing_table(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    old = create_engine(url)
    with old.begin() as conn:
        # station_status as it was before num_ebikes existed.
        conn.execute(text(
            "CREATE TABLE station_status (station_id VARCHAR, snapshot_time DATETIME, "
            "num_bikes INTEGER NOT NULL, num_docks INTEGER NOT NULL, is_renting BOOLEAN, "
            "is_returning BOOLEAN, fetched_at DATETIME, PRIMARY KEY (station_id, snapshot_time))"
        ))
        conn.execute(text("INSERT INTO station_status VALUES ('s00', '2026-11-02 08:00:00', 3, 7, 1, 1, NULL)"))
    old.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(db, "_engine", None)

    db.init_db()
    db.init_db()  # a second run has nothing left to add

    engine = db.get_engine()
    columns = {c["name"] for c in inspect(engine).get_columns("station_status")}
    assert "num_ebikes" in columns
    with engine.connect() as conn:
        assert conn.execute(select(db.station_status.c.num_ebikes)).scalar_one() is None
    engine.dispose()


def test_service_hours_measure_time_empty_and_full(sqlite_db):
    rows = []
    for step in range(12):
        time = START + timedelta(minutes=10 * step)
        # s00 is empty for 3 of 12 snapshots, s01 full for 6, s02 never either.
        for station, bikes in (("s00", 0 if step < 3 else 5), ("s01", 20 if step < 6 else 10), ("s02", 8)):
            rows.append({"station_id": station, "snapshot_time": time, "num_bikes": bikes,
                         "num_docks": 20 - bikes, "is_renting": True, "is_returning": True})
    with sqlite_db.begin() as conn:
        conn.execute(db.stations.insert(), station_rows()[:3])
        conn.execute(db.station_status.insert(), rows)

    service = queries.service_hours.__wrapped__().set_index("name")

    assert service.index.tolist() == ["Station 1", "Station 0", "Station 2"]  # worst first
    assert service.loc["Station 0", ["empty_hours", "full_hours"]].tolist() == [6.0, 0.0]
    assert service.loc["Station 1", ["empty_hours", "full_hours"]].tolist() == [0.0, 12.0]
