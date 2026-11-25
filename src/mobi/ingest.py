"""Fetches data from the Mobi GBFS feed and Open-Meteo and stores it."""

import math
import time
from datetime import datetime, timezone
from typing import Any
from mobi.db import (
    latest_fetch_time,
    station_status,
    stations,
    upsert,
    weather_hourly,
)
import requests
from sqlalchemy import Table

from mobi import config
from mobi.db import station_status, stations, upsert, weather_hourly

EARTH_RADIUS_KM = 6371.0


def utcnow() -> datetime:
    """Returns the current UTC time as a naive datetime (no tzinfo).

    Returns:
        The current time in UTC without timezone information, which is the
        convention for every timestamp stored in the database.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetched_this_hour(table: Table, column: str = "fetched_at") -> bool:
    """Tells whether a table was already refreshed during the current UTC hour.

    Slow-changing data (weather, station metadata) only needs one download per
    hour, which also keeps us under the free API rate limits.

    Args:
        table: The table to inspect.
        column: Name of the DateTime column holding the write time.

    Returns:
        True if the newest value of the column falls in the current hour.
    """
    this_hour = utcnow().replace(minute=0, second=0, microsecond=0)
    last_fetch = latest_fetch_time(table, column)
    return last_fetch is not None and last_fetch >= this_hour


def fetch_json(
    url: str, params: dict[str, Any] | None = None, retries: int = 3
) -> dict[str, Any]:
    """Downloads a URL and parses the response body as JSON.

    Failed requests are retried with a growing pause (2 s, 4 s, ...).

    Args:
        url: The address to request.
        params: Optional query-string parameters.
        retries: Total number of attempts before giving up.

    Returns:
        The parsed JSON document.

    Raises:
        requests.RequestException: If every attempt fails.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the great-circle distance between two points on Earth.

    Args:
        lat1: Latitude of the first point, in degrees.
        lon1: Longitude of the first point, in degrees.
        lat2: Latitude of the second point, in degrees.
        lon2: Longitude of the second point, in degrees.

    Returns:
        The distance in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def assign_area(lat: float, lon: float) -> str:
    """Finds the neighbourhood whose anchor point is closest to a location.

    Args:
        lat: Latitude of the location, in degrees.
        lon: Longitude of the location, in degrees.

    Returns:
        The name of the nearest area in config.AREA_ANCHORS.
    """
    return min(
        config.AREA_ANCHORS,
        key=lambda name: haversine_km(lat, lon, *config.AREA_ANCHORS[name]),
    )


def ingest_stations() -> int:
    """Downloads station metadata and stores it in the stations table.

    Existing stations are updated, so renamed stations or changed capacities
    are picked up. Station metadata rarely changes, so it is refreshed at most
    once per hour.

    Returns:
        The number of stations written, or 0 if they were already refreshed
        during the current hour.
    """
    if fetched_this_hour(stations, "updated_at"):
        return 0
    data = fetch_json(config.STATION_INFO_URL)
    now = utcnow()
    rows = [
        {
            "station_id": s["station_id"],
            "name": s["name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "capacity": s.get("capacity"),
            "area": assign_area(s["lat"], s["lon"]),
            "updated_at": now,
        }
        for s in data["data"]["stations"]
    ]
    return upsert(
        stations,
        rows,
        ["station_id"],
        update_cols=["name", "lat", "lon", "capacity", "area", "updated_at"],
    )

def electric_type_ids() -> set[str] | None:
    """Reads which vehicle types in the feed are e-bikes.

    E-bike counts are a bonus, so a failure here must not cost the snapshot.

    Returns:
        The ids of every vehicle type that is not human-powered, or None if
        the vehicle types could not be downloaded.
    """
    try:
        data = fetch_json(config.VEHICLE_TYPES_URL)
    except requests.RequestException:
        return None
    return {
        vehicle["vehicle_type_id"]
        for vehicle in data["data"]["vehicle_types"]
        if vehicle.get("propulsion_type", "human") != "human"
    }


def status_row(
    station: dict[str, Any], snapshot_time: datetime, fetched_at: datetime, electric: set[str] | None
) -> dict[str, Any]:
    """Converts one station of the station_status feed into a table row.

    Args:
        station: The station's entry in the feed.
        snapshot_time: When the feed was generated (naive UTC).
        fetched_at: When it was downloaded (naive UTC).
        electric: Output of electric_type_ids().

    Returns:
        A row for the station_status table. num_ebikes is None when the feed
        or the vehicle types do not provide it.
    """
    types = station.get("vehicle_types_available")
    return {
        "station_id": station["station_id"],
        "snapshot_time": snapshot_time,
        "num_bikes": station["num_bikes_available"],
        "num_docks": station["num_docks_available"],
        "is_renting": station["is_renting"],
        "is_returning": station["is_returning"],
        "fetched_at": fetched_at,
        "num_ebikes": (
            None
            if electric is None or types is None
            else sum(t["count"] for t in types if t["vehicle_type_id"] in electric)
        ),
    }


def ingest_status() -> int:
    """Downloads the live bike and dock counts and stores one snapshot.

    Every call adds one row per station, stamped with the time the feed was
    generated. Running it again before the feed refreshes writes nothing new.

    Returns:
        The number of new rows inserted.
    """
    data = fetch_json(config.STATION_STATUS_URL)
    snapshot_time = datetime.fromtimestamp(
        data["last_updated"], tz=timezone.utc
    ).replace(tzinfo=None, second=0, microsecond=0)
    now = utcnow()
    electric = electric_type_ids()
    rows = [
        status_row(s, snapshot_time, now, electric)
        for s in data["data"]["stations"]
        if s["is_installed"]
    ]
    return upsert(station_status, rows, ["station_id", "snapshot_time"])

def ingest_weather() -> int:
    """Downloads hourly weather for Vancouver and stores it.

    The request covers yesterday, today and tomorrow. Hours that already exist
    are overwritten, because forecasts are revised and past hours are replaced
    by observed values.

    Returns:
        The number of hourly rows written, or 0 if the weather was already
        fetched during the current hour.
    """
    if fetched_this_hour(weather_hourly):
        return 0
    params = {
        "latitude": config.VANCOUVER_LAT,
        "longitude": config.VANCOUVER_LON,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "GMT",
    }
    data = fetch_json(config.WEATHER_URL, params)
    hourly = data["hourly"]
    now = utcnow()
    rows = [
        {
            "time": datetime.fromisoformat(time_str),
            "temperature": temperature,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "fetched_at": now,
        }
        for time_str, temperature, precipitation, wind_speed in zip(
            hourly["time"],
            hourly["temperature_2m"],
            hourly["precipitation"],
            hourly["wind_speed_10m"],
        )
    ]
    return upsert(
        weather_hourly,
        rows,
        ["time"],
        update_cols=["temperature", "precipitation", "wind_speed", "fetched_at"],
    )