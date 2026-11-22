"""Demonstrates that upsert is safe to repeat, using throwaway rows."""

from datetime import datetime

from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError

from mobi.db import get_engine, station_status, stations, upsert


def main() -> None:
    """Writes the same rows twice, prints the counts, then deletes the rows."""
    engine = get_engine()
    now = datetime(2026, 11, 22, 14, 0)
    station = [{"station_id": "__t__", "name": "a", "lat": 1.0, "lon": 1.0,
                "capacity": 10, "area": "x", "updated_at": now}]
    status = [{"station_id": "__t__", "snapshot_time": now, "num_bikes": 3,
               "num_docks": 7, "is_renting": True, "is_returning": True,
               "fetched_at": now}]
    key = ["station_id", "snapshot_time"]
    try:
        print("station insert :", upsert(stations, station, ["station_id"]))
        print("status insert 1:", upsert(station_status, status, key))
        print("status insert 2:", upsert(station_status, status, key))
        station[0]["name"] = "b"
        print("station update :",
              upsert(stations, station, ["station_id"], update_cols=["name"]))
        try:
            with engine.begin() as conn:
                conn.execute(insert(station_status), status)
        except IntegrityError as error:
            print("plain INSERT fails:", str(error.orig).splitlines()[0])
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM station_status WHERE station_id = '__t__'"))
            conn.execute(text("DELETE FROM stations WHERE station_id = '__t__'"))
        print("cleanup done")


if __name__ == "__main__":
    main()