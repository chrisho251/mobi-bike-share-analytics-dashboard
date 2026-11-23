"""Turns next-hour forecasts into concrete rebalancing suggestions.

A dispatcher cannot act on "station 4321 will have 0.8 bikes". They need
"move 4 bikes from X to Y". The rules below are deliberately simple so they can
be explained in one sentence each:

* A station is short if its forecast is under 15 % full.
* A station can donate if its forecast is over 55 % full; it keeps half.
* Each short station is topped up towards 30 % from the nearest donors within
  MAX_KM kilometres.
"""

import math

import pandas as pd

from mobi.ingest import haversine_km

SHORT_FILL = 0.15
FULL_FILL = 0.90
TARGET_FILL = 0.30
DONOR_FILL = 0.55
DONOR_KEEP = 0.50
MAX_KM = 2.0


def at_risk_stations(frame: pd.DataFrame) -> pd.DataFrame:
    """Lists stations forecast to run empty or full within the hour.

    Args:
        frame: The output of queries.overview().

    Returns:
        One row per station at risk with its name, area, current bikes, the
        corrected forecast, capacity and a risk label. Empty if there are no
        forecasts.
    """
    if frame.empty or frame["pred_corrected"].isna().all():
        return pd.DataFrame()
    usable = frame.dropna(subset=["pred_corrected", "capacity"])
    usable = usable[usable["capacity"] > 0]
    fill = usable["pred_corrected"] / usable["capacity"]
    risky = usable[(fill < SHORT_FILL) | (fill > FULL_FILL)].copy()
    risky["risk"] = (risky["pred_corrected"] / risky["capacity"] < SHORT_FILL).map(
        {True: "Empty soon", False: "Full soon"}
    )
    risky["forecast_bikes"] = risky["pred_corrected"].round(1)
    result = risky[
        ["name", "area", "num_bikes", "forecast_bikes", "capacity", "risk"]
    ].rename(columns={"name": "station", "num_bikes": "bikes_now"})
    return result.sort_values(["risk", "forecast_bikes"]).reset_index(drop=True)


def plan_moves(frame: pd.DataFrame) -> pd.DataFrame:
    """Suggests bike moves from stations with spare bikes to short stations.

    Args:
        frame: The output of queries.overview().

    Returns:
        One row per suggested move with from_station, to_station, bikes,
        distance_km and to_area, most urgent first. Empty if no move is needed.
    """
    if frame.empty or frame["pred_corrected"].isna().all():
        return pd.DataFrame()
    stations = frame.dropna(subset=["pred_corrected", "capacity"]).copy()
    stations = stations[stations["capacity"] > 0].reset_index(drop=True)
    stations["fill"] = stations["pred_corrected"] / stations["capacity"]

    short = stations[stations["fill"] < SHORT_FILL].copy()
    short["need"] = (
        (TARGET_FILL * short["capacity"] - short["pred_corrected"]).round().clip(lower=1)
    )
    donors = stations[stations["fill"] > DONOR_FILL].copy()
    donors["spare"] = (
        (donors["pred_corrected"] - DONOR_KEEP * donors["capacity"]).apply(math.floor)
    ).clip(lower=0)

    moves = []
    for target in short.sort_values("need", ascending=False).itertuples():
        need = int(target.need)
        distance = donors.apply(
            lambda d: haversine_km(target.lat, target.lon, d["lat"], d["lon"]), axis=1
        )
        nearby = donors[(distance <= MAX_KM) & (donors["spare"] > 0)]
        for index in distance[nearby.index].sort_values().index:
            if need <= 0:
                break
            take = int(min(need, donors.at[index, "spare"]))
            if take <= 0:
                continue
            donors.at[index, "spare"] -= take
            need -= take
            moves.append(
                {
                    "from_station": donors.at[index, "name"],
                    "to_station": target.name,
                    "bikes": take,
                    "distance_km": round(float(distance[index]), 2),
                    "to_area": target.area,
                }
            )
    return pd.DataFrame(moves)
