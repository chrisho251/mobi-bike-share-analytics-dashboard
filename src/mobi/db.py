"""Database schema and helpers shared by the pipeline and the dashboard."""
from collections.abc import Sequence
from typing import Any
from sqlalchemy.engine import Engine
from mobi.config import get_db_url
from datetime import datetime
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# All timestamps are stored in UTC as naive datetimes (no tzinfo).
metadata = MetaData()

stations = Table(
    "stations",
    metadata,
    Column("station_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("lat", Float, nullable=False),
    Column("lon", Float, nullable=False),
    Column("capacity", Integer),
    Column("area", String),
    Column("updated_at", DateTime),
)

station_status = Table(
    "station_status",
    metadata,
    Column("station_id", String, ForeignKey("stations.station_id"), primary_key=True),
    Column("snapshot_time", DateTime, primary_key=True),
    Column("num_bikes", Integer, nullable=False),
    Column("num_docks", Integer, nullable=False),
    Column("is_renting", Boolean),
    Column("is_returning", Boolean),
    Column("fetched_at", DateTime),
    Index("ix_station_status_time", "snapshot_time"),
)

weather_hourly = Table(
    "weather_hourly",
    metadata,
    Column("time", DateTime, primary_key=True),
    Column("temperature", Float),
    Column("precipitation", Float),
    Column("wind_speed", Float),
    Column("fetched_at", DateTime),
)

_engine: Engine | None = None
def get_engine() -> Engine:
    """Returns the shared SQLAlchemy engine, creating it on first use.

    Returns:
        A SQLAlchemy Engine connected to the database from get_db_url().
    """
    global _engine
    if _engine is None:
        _engine = create_engine(get_db_url(), pool_pre_ping=True)
    return _engine

def upsert(
    table: Table,
    rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
    update_cols: Sequence[str] | None = None,
) -> int:
    """Inserts rows, resolving duplicate keys without raising an error.

    Args:
        table: The table to write into.
        rows: One dict per row, mapping column names to values.
        keys: Names of the columns that form the unique key (normally the
            primary key). A row whose key already exists is a duplicate.
        update_cols: Columns to overwrite when a duplicate is found. If None,
            duplicates are skipped and the existing row is kept unchanged.

    Returns:
        The number of rows actually inserted (or updated, when update_cols is
        given). Skipped duplicates are not counted.
    """
    if not rows:
        return 0
    engine = get_engine()
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    stmt = insert(table).values(list(rows))
    if update_cols:
        stmt = stmt.on_conflict_do_update(
            index_elements=list(keys),
            set_={col: stmt.excluded[col] for col in update_cols},
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=list(keys))
    stmt = stmt.returning(table.c[keys[0]])
    with engine.begin() as conn:
        written = conn.execute(stmt).fetchall()
    return len(written)

def init_db() -> None:
    """Creates any missing tables. Existing tables are left untouched."""
    metadata.create_all(get_engine())

def latest_fetch_time(table: Table) -> datetime | None:
    """Returns the most recent fetched_at value stored in a table.

    Args:
        table: A table that has a fetched_at column.

    Returns:
        The latest fetched_at as a naive UTC datetime, or None if the table
        is empty.
    """
    with get_engine().connect() as conn:
        return conn.execute(select(func.max(table.c.fetched_at))).scalar()