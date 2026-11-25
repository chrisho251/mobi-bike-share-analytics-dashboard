"""Generates a synthetic history of the Mobi network in a local SQLite file.

The simulated data only exists to develop and test the forecasting code with
more history than the live collector has gathered. It is never written to the
production database: this script points DATABASE_URL at its own SQLite file
before importing the project modules.

Usage:
    uv run python simulation/generate.py --days 21
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SIM_DIR = Path(__file__).resolve().parent
DEFAULT_DB = SIM_DIR / "data" / "mobi_sim.db"
STATIONS_CSV = SIM_DIR / "stations.csv"
STEP = timedelta(minutes=10)
LOCAL_TZ = ZoneInfo("America/Vancouver")
# Share of docked bikes that are e-bikes (16 % in the live feed, November 2026).
EBIKE_SHARE = 0.16

# Share of each area's trips that start (-1) or end (+1) there, by time of day.
# Residential areas empty in the morning and refill in the evening; downtown
# does the opposite; Stanley Park is driven by leisure rides.
PROFILES = {
    "commuter_home": [(7, 10, -0.6), (16, 19, 0.6)],
    "commuter_work": [(7, 10, 0.6), (16, 19, -0.6)],
    "leisure": [(10, 14, -0.5), (15, 19, 0.5)],
}
AREA_PROFILE = {
    "Downtown": "commuter_work",
    "Stanley Park & Coal Harbour": "leisure",
    "West End": "commuter_home",
    "Kitsilano": "commuter_home",
    "Fairview & Mount Pleasant": "commuter_home",
    "East Van": "commuter_home",
}


def parse_args() -> argparse.Namespace:
    """Reads the command-line options.

    Returns:
        The parsed options (days, end, seed, db).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=21, help="days of history")
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        default=None,
        help="last snapshot, naive UTC (default: now, floored to 10 minutes)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    return parser.parse_args()


def activity_curve(local_hour: float, weekend: bool) -> float:
    """Returns relative trip volume for a local time of day.

    Args:
        local_hour: Hour of day in Vancouver time, with fractions.
        weekend: Whether the day is Saturday or Sunday.

    Returns:
        A multiplier around 1 at busy hours and near 0 at night.
    """
    if weekend:
        return 0.1 + 0.9 * np.exp(-((local_hour - 14) ** 2) / 18)
    morning = np.exp(-((local_hour - 8.5) ** 2) / 2)
    evening = np.exp(-((local_hour - 17.5) ** 2) / 3)
    return 0.08 + 0.5 * np.exp(-((local_hour - 13) ** 2) / 10) + morning + evening


def flow_bias(profile: str, local_hour: float, weekend: bool) -> float:
    """Returns the net direction of trips at a station (-1 out, +1 in).

    Args:
        profile: A key of PROFILES.
        local_hour: Hour of day in Vancouver time.
        weekend: Whether the day is Saturday or Sunday.

    Returns:
        A value in [-1, 1]; 0 means as many arrivals as departures.
    """
    scale = 0.3 if weekend and profile != "leisure" else 1.0
    if weekend and profile == "leisure":
        scale = 1.5
    for start, end, bias in PROFILES[profile]:
        if start <= local_hour < end:
            return float(np.clip(bias * scale, -0.9, 0.9))
    return 0.0


def simulate_weather(hours: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Simulates hourly Vancouver-like autumn weather.

    Rain follows a two-state Markov chain so wet spells last several hours.

    Args:
        hours: Hourly naive UTC timestamps.
        rng: Random number generator.

    Returns:
        A frame with time, temperature, precipitation and wind_speed.
    """
    raining = False
    precipitation, temperature = [], []
    daily_offset = 0.0
    for time in hours:
        if time.hour == 0:
            daily_offset = 0.7 * daily_offset + rng.normal(0, 1.5)
        raining = rng.random() < (0.85 if raining else 0.06)
        precipitation.append(round(float(rng.gamma(1.5, 0.8)), 1) if raining else 0.0)
        local_hour = time.tz_localize("UTC").tz_convert(LOCAL_TZ).hour
        temperature.append(
            round(7 + daily_offset + 3 * np.sin((local_hour - 9) / 24 * 2 * np.pi), 1)
        )
    return pd.DataFrame(
        {
            "time": hours.to_pydatetime(),
            "temperature": temperature,
            "precipitation": precipitation,
            "wind_speed": np.round(rng.gamma(2.0, 6.0, len(hours)), 1),
        }
    )


def simulate_status(
    stations: pd.DataFrame,
    times: pd.DatetimeIndex,
    weather: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulates bike counts for every station at every 10-minute step.

    Each step, arrivals and departures are Poisson draws whose total depends on
    the hour, the weekday and the rain, and whose balance depends on the
    station's area profile. Rebalancing vans occasionally reset empty or full
    stations to half capacity during the day.

    Args:
        stations: Station metadata from stations.csv.
        times: Snapshot times (naive UTC), every 10 minutes.
        weather: Hourly weather from simulate_weather.
        rng: Random number generator.

    Returns:
        A frame shaped like the station_status table.
    """
    capacity = stations["capacity"].to_numpy()
    profiles = stations["area"].map(AREA_PROFILE).fillna("commuter_home").to_numpy()
    # Busier stations draw more trips; scaled so a busy hour moves ~4 bikes.
    activity = rng.lognormal(0, 0.5, len(stations)) * capacity / 18 * 0.7
    bikes = np.round(capacity * rng.uniform(0.3, 0.7, len(stations))).astype(int)
    rain_by_hour = weather.set_index("time")["precipitation"]

    frames = []
    for time in times:
        local = time.tz_localize("UTC").tz_convert(LOCAL_TZ)
        local_hour = local.hour + local.minute / 60
        weekend = local.day_of_week >= 5
        rain = rain_by_hour.get(time.floor("h"), 0.0)
        trips = activity * activity_curve(local_hour, weekend) * np.exp(-0.5 * rain)
        bias = np.array([flow_bias(p, local_hour, weekend) for p in profiles])
        arrivals = rng.poisson(trips * (1 + bias) / 2)
        departures = rng.poisson(trips * (1 - bias) / 2)
        bikes = np.clip(bikes + arrivals - departures, 0, capacity)

        stuck = (bikes <= 1) | (bikes >= capacity - 1)
        van = stuck & (6 <= local_hour < 22) & (rng.random(len(stations)) < 0.08)
        bikes = np.where(van, np.round(capacity / 2).astype(int), bikes)

        frames.append(
            pd.DataFrame(
                {
                    "station_id": stations["station_id"],
                    "snapshot_time": time.to_pydatetime(),
                    "num_bikes": bikes,
                    "num_docks": capacity - bikes,
                    "is_renting": True,
                    "is_returning": True,
                    "fetched_at": time.to_pydatetime(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def write_tables(db_path: Path, stations, status, weather) -> None:
    """Creates a fresh simulation database and stores the generated tables.

    Args:
        db_path: The SQLite file to (re)create.
        stations: Station metadata.
        status: Simulated station_status rows.
        weather: Simulated weather_hourly rows.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"

    # Imported only now so the engine is created for the simulation file.
    from mobi import db

    assert db.get_engine().dialect.name == "sqlite", "simulation must use SQLite"
    db.init_db()
    end = status["snapshot_time"].max()
    with db.get_engine().begin() as conn:
        conn.execute(
            db.stations.insert(),
            stations.assign(updated_at=end).to_dict("records"),
        )
        conn.execute(db.weather_hourly.insert(), weather.assign(fetched_at=end).to_dict("records"))
        conn.execute(db.station_status.insert(), status.to_dict("records"))


def main() -> None:
    """Generates the simulated history and prints a short summary."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    end = args.end or datetime.now(timezone.utc).replace(tzinfo=None)
    end = pd.Timestamp(end).floor("10min")
    times = pd.date_range(end - timedelta(days=args.days), end, freq=STEP)
    hours = pd.date_range(times[0].floor("h"), end.floor("h") + timedelta(days=2), freq="h")

    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    weather = simulate_weather(hours, rng)
    status = simulate_status(stations, times, weather, rng)
    # A separate generator, so adding e-bikes left the simulated history unchanged.
    status["num_ebikes"] = np.random.default_rng(args.seed + 1).binomial(
        status["num_bikes"], EBIKE_SHARE
    )
    write_tables(args.db, stations, status, weather)
    print(
        f"Wrote {len(status):,} snapshots rows for {len(stations)} stations, "
        f"{times[0]} to {times[-1]} UTC, into {args.db}"
    )


if __name__ == "__main__":
    main()
