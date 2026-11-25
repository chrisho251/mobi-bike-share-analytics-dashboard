"""Project configuration: API endpoints, constants and database URL """

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env.local")

# Data sources API
GBFS_BASE_URL = "https://gbfs.kappa.fifteen.eu/gbfs/2.3/mobi/en"
STATION_STATUS_URL = f"{GBFS_BASE_URL}/station_status.json"
STATION_INFO_URL = f"{GBFS_BASE_URL}/station_information.json"
VEHICLE_TYPES_URL = f"{GBFS_BASE_URL}/vehicle_types.json"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Vancouver lat/lon
VANCOUVER_LAT = 49.2827
VANCOUVER_LON = -123.1207

# Anchor points used to label each station with a neighbourhood. A station
# belongs to the anchor that is closest to it.
AREA_ANCHORS = {
    "Downtown": (49.2827, -123.1207),
    "West End": (49.2870, -123.1380),
    "Stanley Park & Coal Harbour": (49.2960, -123.1330),
    "Kitsilano": (49.2680, -123.1600),
    "Fairview & Mount Pleasant": (49.2630, -123.1100),
    "East Van": (49.2720, -123.0700),
}

# Without DATABASE_URL the project runs on the simulated network, so anyone can
# explore it without credentials. Set DATABASE_URL to switch to live data.
SIMULATION_DB = Path(__file__).resolve().parents[2] / "simulation" / "data" / "mobi_sim.db"
SIMULATION_DATABASE_URL = f"sqlite:///{SIMULATION_DB.as_posix()}"


def get_db_url() -> str:
    """Returns the SQLAlchemy database URL for the current environment.

    Reads DATABASE_URL (from the environment or .env.local). If it is not set,
    falls back to the simulation database.

    Returns:
        A SQLAlchemy URL. Postgres URLs are rewritten to use the psycopg 3
        driver.
    """
    url = os.getenv("DATABASE_URL", SIMULATION_DATABASE_URL)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def is_simulation() -> bool:
    """Tells whether the project is pointed at the simulation database.

    Returns:
        True if the current database URL is the simulation SQLite file.
    """
    return Path(get_db_url().removeprefix("sqlite:///")).resolve() == SIMULATION_DB
