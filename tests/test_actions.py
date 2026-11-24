"""Tests for the rebalancing suggestions."""

import pandas as pd

from mobi import actions


def _overview(forecasts: list[float], lats: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"S{i}" for i in range(len(forecasts))],
            "area": "Downtown",
            "lat": lats,
            "lon": -123.12,
            "capacity": 20,
            "num_bikes": [round(f) for f in forecasts],
            "pred_corrected": forecasts,
        }
    )


def test_short_station_is_refilled_from_nearest_donor():
    frame = _overview([1.0, 18.0, 18.0], [49.280, 49.281, 49.290])

    moves = actions.plan_moves(frame)

    assert moves.iloc[0]["from_station"] == "S1"  # closest donor first
    assert moves.iloc[0]["to_station"] == "S0"
    assert moves["bikes"].sum() == 5  # 30 % of 20 minus the forecast of 1


def test_donor_beyond_max_distance_is_ignored():
    frame = _overview([1.0, 18.0], [49.28, 49.40])  # about 13 km apart

    assert actions.plan_moves(frame).empty


def test_at_risk_labels_empty_and_full():
    frame = _overview([1.0, 10.0, 19.5], [49.28, 49.29, 49.30])

    risks = actions.at_risk_stations(frame).set_index("station")["risk"]

    assert risks.to_dict() == {"S0": "Empty soon", "S2": "Full soon"}
