"""Database schema and helpers shared by the pipeline and the dashboard."""

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
)
from sqlalchemy.engine import Engine

from mobi.config import get_db_url

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

def init_db() -> None:
    """Creates any missing tables. Existing tables are left untouched."""
    metadata.create_all(get_engine())