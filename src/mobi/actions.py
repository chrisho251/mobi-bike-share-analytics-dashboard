"""Turns next-hour forecasts into concrete rebalancing suggestions.

A dispatcher cannot act on "station 4321 will have 0.8 bikes". They need
"take 6 bikes from X to Y and Z". The rules below are deliberately simple so
they can be explained in one sentence each:

* A station is short if its forecast is under 15 % full; it is topped up
  towards 30 %, so riders find a bike.
* A station is full if its forecast is over 90 % full; it is emptied towards
  70 %, so riders can return a bike.
* Short stations are served first, the most bikes needed first. They take
  bikes from the nearest full stations within MAX_KM, then from the nearest
  other stations over 55 % full. A giving station keeps half its capacity.
* Bikes a full station still has to lose go to the nearest stations within
  MAX_KM that are under 50 % full, filling them up to 50 %.
* Moves from the same station are grouped into van trips of at most
  VAN_CAPACITY bikes.

Every planned move updates the forecast bikes of both stations, so no bike is
promised twice.
"""

import math

import pandas as pd

from mobi.ingest import haversine_km

SHORT_FILL = 0.15
FULL_FILL = 0.90
TARGET_FILL = 0.30
FULL_TARGET_FILL = 0.70
DONOR_FILL = 0.55
DONOR_KEEP = 0.50
RECEIVER_FILL = 0.50
MAX_KM = 2.0
# Bikes one van carries per trip. An assumption: set it to the real fleet's.
VAN_CAPACITY = 20

EMPTY_SOON = "Empty soon"
FULL_SOON = "Full soon"
MOVE_COLUMNS = ["from_station", "to_station", "bikes", "distance_km", "to_area", "reason"]
TRIP_COLUMNS = ["trip", "pickup", "bikes", "drop_offs", "farthest_km", "reason"]


def _usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Keeps the stations with a forecast and a known capacity.

    Args:
        frame: The output of queries.overview().

    Returns:
        The usable rows with a fresh index and a forecast fill column.
    """
    if frame.empty or frame["pred_corrected"].isna().all():
        return pd.DataFrame()
    usable = frame.dropna(subset=["pred_corrected", "capacity"])
    usable = usable[usable["capacity"] > 0].reset_index(drop=True)
    return usable.assign(fill=usable["pred_corrected"] / usable["capacity"])


def at_risk_stations(frame: pd.DataFrame) -> pd.DataFrame:
    """Lists stations forecast to run empty or full within the hour.

    Args:
        frame: The output of queries.overview().

    Returns:
        One row per station at risk with its name, area, current bikes, the
        corrected forecast, capacity, forecast fill and a risk label. Stations
        running empty come first, then full ones, each most urgent first.
        Empty if there are no forecasts.
    """
    stations = _usable(frame)
    if stations.empty:
        return pd.DataFrame()
    risky = stations[(stations["fill"] < SHORT_FILL) | (stations["fill"] > FULL_FILL)].copy()
    risky["risk"] = (risky["fill"] < SHORT_FILL).map({True: EMPTY_SOON, False: FULL_SOON})
    # Emptiest first among the short stations, fullest first among the full.
    risky["urgency"] = risky["fill"].where(risky["risk"] == EMPTY_SOON, -risky["fill"])
    risky = risky.sort_values(["risk", "urgency"])
    risky["forecast_bikes"] = risky["pred_corrected"].round(1)
    risky["forecast_full_pct"] = (risky["fill"] * 100).round().astype(int)
    result = risky[
        ["name", "area", "num_bikes", "forecast_bikes", "capacity", "forecast_full_pct", "risk"]
    ].rename(columns={"name": "station", "num_bikes": "bikes_now"})
    return result.reset_index(drop=True)


def plan_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """Suggests bike moves that refill short stations and relieve full ones.

    Args:
        frame: The output of queries.overview().

    Returns:
        One row per suggested move with from_station, to_station, bikes,
        distance_km, to_area and reason (the risks it fixes: "Empty soon",
        "Full soon", or both joined by " + " when a full station refills a
        short one), most urgent first. Empty if no move is needed.
    """
    stations = _usable(frame)
    if stations.empty:
        return pd.DataFrame(columns=MOVE_COLUMNS)
    capacity = stations["capacity"].astype(float).tolist()
    bikes = stations["pred_corrected"].astype(float).tolist()  # updated as moves are planned
    short = stations.index[stations["fill"] < SHORT_FILL]
    full = set(stations.index[stations["fill"] > FULL_FILL])
    donors = set(stations.index[stations["fill"] > DONOR_FILL])

    points = list(zip(stations["lat"], stations["lon"]))

    def nearby(i: int) -> list[tuple[float, int]]:
        """Other stations within MAX_KM of station i as (distance, index), nearest first."""
        pairs = (
            (haversine_km(*points[i], *point), j) for j, point in enumerate(points) if j != i
        )
        return sorted(pair for pair in pairs if pair[0] <= MAX_KM)

    moves = []

    def move(source: int, target: int, count: int, distance: float, reason: str) -> None:
        bikes[source] -= count
        bikes[target] += count
        moves.append(
            {
                "from_station": stations.at[source, "name"],
                "to_station": stations.at[target, "name"],
                "bikes": count,
                "distance_km": round(distance, 2),
                "to_area": stations.at[target, "area"],
                "reason": reason,
            }
        )

    need = {i: max(1, round(TARGET_FILL * capacity[i] - bikes[i])) for i in short}
    for target in sorted(short, key=lambda i: -need[i]):
        remaining = need[target]
        candidates = [(d, j) for d, j in nearby(target) if j in donors]
        # Full stations give first: the same trip then fixes both stations.
        for distance, source in sorted(candidates, key=lambda c: (c[1] not in full, c[0])):
            if remaining <= 0:
                break
            spare = math.floor(bikes[source] - DONOR_KEEP * capacity[source])
            count = min(remaining, spare)
            if count > 0:
                reason = f"{EMPTY_SOON} + {FULL_SOON}" if source in full else EMPTY_SOON
                move(source, target, count, distance, reason)
                remaining -= count

    def excess(i: int) -> int:
        return math.floor(bikes[i] - FULL_TARGET_FILL * capacity[i])

    for source in sorted(full, key=lambda i: -excess(i)):
        for distance, target in nearby(source):
            surplus = excess(source)
            if surplus <= 0:
                break
            if target in full:
                continue
            room = math.floor(RECEIVER_FILL * capacity[target] - bikes[target])
            count = min(surplus, room)
            if count > 0:
                move(source, target, count, distance, FULL_SOON)
    return pd.DataFrame(moves, columns=MOVE_COLUMNS)


def plan_trips(moves: pd.DataFrame) -> pd.DataFrame:
    """Groups moves into van trips: one pickup station, up to VAN_CAPACITY bikes.

    Moves keep their order of urgency: a trip is numbered by its most urgent
    move, and a pickup with more bikes than a van holds becomes several trips.

    Args:
        moves: The output of plan_moves().

    Returns:
        One row per trip with trip number, pickup station, bikes, the drop-offs
        as text ("Station A (4), Station B (2)"), the farthest drop-off in km
        and the reasons. Empty if there are no moves.
    """
    if moves.empty:
        return pd.DataFrame(columns=TRIP_COLUMNS)
    trips: list[dict] = []
    open_trip: dict[str, dict] = {}
    for move in moves.itertuples():
        remaining = int(move.bikes)
        while remaining > 0:
            trip = open_trip.get(move.from_station)
            if trip is None or trip["bikes"] == VAN_CAPACITY:
                trip = {"pickup": move.from_station, "bikes": 0, "stops": [], "km": 0.0, "reasons": []}
                trips.append(trip)
                open_trip[move.from_station] = trip
            count = min(remaining, VAN_CAPACITY - trip["bikes"])
            trip["bikes"] += count
            trip["stops"].append(f"{move.to_station} ({count})")
            trip["km"] = max(trip["km"], move.distance_km)
            for reason in move.reason.split(" + "):
                if reason not in trip["reasons"]:
                    trip["reasons"].append(reason)
            remaining -= count
    return pd.DataFrame(
        [
            {
                "trip": number,
                "pickup": trip["pickup"],
                "bikes": trip["bikes"],
                "drop_offs": ", ".join(trip["stops"]),
                "farthest_km": trip["km"],
                "reason": " + ".join(trip["reasons"]),
            }
            for number, trip in enumerate(trips, start=1)
        ],
        columns=TRIP_COLUMNS,
    )
