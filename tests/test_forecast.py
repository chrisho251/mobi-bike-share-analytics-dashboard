"""Tests for training, forecasting, correction and scoring."""

from datetime import timedelta

import pandas as pd
from conftest import status_rows
from sqlalchemy import func, select

from mobi import db, forecast


def _scored(station: str, model_errors: list[float], baseline_errors: list[float]):
    return pd.DataFrame(
        {
            "station_id": station,
            "actual": 10.0,
            "pred_model": [10.0 + e for e in model_errors],
            "pred_baseline": [10.0 + e for e in baseline_errors],
        }
    )


def test_model_is_trusted_only_where_it_beat_the_baseline():
    scored = pd.concat(
        [
            _scored("good", [0, 0, 0], [2, 2, 2]),
            _scored("bad", [2, 2, 2], [0, 0, 0]),
            _scored("too_few", [0], [2]),
        ]
    )

    assert forecast.trusted_stations(scored) == {"good"}


def test_correct_falls_back_to_baseline_for_untrusted_stations():
    rows = pd.DataFrame(
        {"station_id": ["a", "b"], "pred_model": [5.0, 5.0], "pred_baseline": [8.0, 8.0]}
    )

    corrected = forecast.correct(rows, {"a"})

    assert corrected["pred_corrected"].tolist() == [5.0, 8.0]


def test_forecasts_stay_within_capacity():
    values = pd.Series([-2.0, 5.0, 25.0])
    capacity = pd.Series([20, 20, 20])

    assert forecast._clip(values, capacity).tolist() == [0.0, 5.0, 20.0]


def test_pipeline_forecasts_then_scores_against_reality(network):
    assert forecast.predict_next_hour() == 20
    assert forecast.predict_next_hour() == 0  # same snapshot is not forecast twice

    later = status_rows(network + timedelta(minutes=10), network + timedelta(minutes=60), seed=1)
    with db.get_engine().begin() as conn:
        conn.execute(db.station_status.insert(), later)

    assert forecast.fill_actuals() == 20
    with db.get_engine().connect() as conn:
        assert conn.execute(select(func.count()).select_from(db.model_runs)).scalar() == 1
