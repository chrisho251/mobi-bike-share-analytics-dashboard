"""Project configuration: API endpoints, constants and database URL """

import os

from dotenv import load_dotenv

load_dotenv(".env.local")

# Data sources API
GBFS_BASE_URL = "https://gbfs.kappa.fifteen.eu/gbfs/2.3/mobi/en"
STATION_STATUS_URL = f"{GBFS_BASE_URL}/station_status.json"
STATION_INFO_URL = f"{GBFS_BASE_URL}/station_information.json"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Vancouver lat/lon
VANCOUVER_LAT = 49.2827
VANCOUVER_LON = -123.1207

# Default database
DEFAULT_DATABASE_URL = "sqlite:///data/mobi.db"

def get_db_url() -> str:
    """Return the SQLALchemy database URL for the current env
        Reads the DATABASE_URL env variable (set in .env.local by the Neon CLI).
        If it is not set, falls back to a local SQLite file so that anyone can run the project offline without a Neon account

        Returns:
            A SQLAlchemy URL string. Postgres URLs are rewritten to use the psycopg (version 3) driver
    """
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url