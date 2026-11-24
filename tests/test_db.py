"""Tests for the upsert helper."""

from conftest import START, station_rows, status_rows

from mobi import db


def test_upsert_skips_duplicate_snapshots(sqlite_db):
    db.upsert(db.stations, station_rows(), ["station_id"])
    rows = status_rows(START, START)
    keys = ["station_id", "snapshot_time"]

    assert db.upsert(db.station_status, rows, keys) == len(rows)
    assert db.upsert(db.station_status, rows, keys) == 0


def test_upsert_overwrites_when_update_columns_given(sqlite_db):
    rows = station_rows()[:1]
    db.upsert(db.stations, rows, ["station_id"])
    rows[0]["name"] = "Renamed"

    assert db.upsert(db.stations, rows, ["station_id"], update_cols=["name"]) == 1
    with sqlite_db.connect() as conn:
        assert conn.execute(db.stations.select()).one().name == "Renamed"
