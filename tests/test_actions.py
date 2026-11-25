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


def test_at_risk_lists_the_most_urgent_first():
    frame = _overview([2.0, 0.5, 19.0, 20.0], [49.28, 49.29, 49.30, 49.31])

    risks = actions.at_risk_stations(frame)

    assert risks["station"].tolist() == ["S1", "S0", "S3", "S2"]
    assert risks["forecast_full_pct"].tolist() == [2, 10, 100, 95]


def test_full_station_is_emptied_into_the_nearest_station_with_room():
    frame = _overview([19.0, 4.0, 4.0], [49.280, 49.282, 49.281])

    moves = actions.plan_moves(frame)

    # Down to 70 % of 20 = 14, so 5 bikes go to the nearest station (S2).
    assert moves.to_dict("records") == [
        {"from_station": "S0", "to_station": "S2", "bikes": 5, "distance_km": 0.11,
         "to_area": "Downtown", "reason": "Full soon"}
    ]


def test_short_station_takes_from_a_full_station_first():
    # S1 is nearer and could give, but taking from the full S2 fixes both.
    frame = _overview([1.0, 15.0, 19.0], [49.280, 49.281, 49.283])

    moves = actions.plan_moves(frame)

    assert moves[["from_station", "to_station", "bikes", "reason"]].to_dict("records") == [
        {"from_station": "S2", "to_station": "S0", "bikes": 5, "reason": "Empty soon + Full soon"}
    ]


def test_moves_are_grouped_into_van_trips(monkeypatch):
    monkeypatch.setattr(actions, "VAN_CAPACITY", 5)
    moves = pd.DataFrame(
        {
            "from_station": ["A", "A", "B"],
            "to_station": ["X", "Y", "Z"],
            "bikes": [4, 3, 2],
            "distance_km": [0.5, 1.2, 0.3],
            "to_area": "Downtown",
            "reason": ["Empty soon", "Full soon", "Empty soon"],
        }
    )

    trips = actions.plan_trips(moves)

    assert trips[["pickup", "bikes", "drop_offs", "farthest_km", "reason"]].to_dict("records") == [
        {"pickup": "A", "bikes": 5, "drop_offs": "X (4), Y (1)", "farthest_km": 1.2,
         "reason": "Empty soon + Full soon"},
        {"pickup": "A", "bikes": 2, "drop_offs": "Y (2)", "farthest_km": 1.2, "reason": "Full soon"},
        {"pickup": "B", "bikes": 2, "drop_offs": "Z (2)", "farthest_km": 0.3, "reason": "Empty soon"},
    ]
    assert trips["trip"].tolist() == [1, 2, 3]


def test_no_forecast_means_no_moves_or_trips():
    frame = _overview([1.0, 18.0], [49.28, 49.281]).assign(pred_corrected=None)

    assert actions.plan_moves(frame).empty
    assert actions.plan_trips(actions.plan_moves(frame)).empty
