"""Fetches data from the Mobi GBFS feed and Open-Meteo and stores it."""

import math
import time
from datetime import datetime, timezone
from typing import Any

import requests

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
    are picked up.

    Returns:
        The number of stations written.
    """
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
    rows = [
        {
            "station_id": s["station_id"],
            "snapshot_time": snapshot_time,
            "num_bikes": s["num_bikes_available"],
            "num_docks": s["num_docks_available"],
            "is_renting": s["is_renting"],
            "is_returning": s["is_returning"],
            "fetched_at": now,
        }
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
        The number of hourly rows written.
    """
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