"""Tests for training, forecasting, correction and scoring."""

from datetime import datetime, timedelta

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
            _scored("good", [0] * 6, [2] * 6),
            _scored("bad", [2] * 6, [0] * 6),
            _scored("too_few", [0] * 5, [2] * 5),
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


def _add_snapshot(time: datetime, seed: int) -> None:
    with db.get_engine().begin() as conn:
        conn.execute(db.station_status.insert(), status_rows(time, time, seed=seed))


def _count(table) -> int:
    with db.get_engine().connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar()


def _versions_at(base_time: datetime) -> set[str]:
    with db.get_engine().connect() as conn:
        return set(
            conn.execute(
                select(db.predictions_log.c.model_version).where(
                    db.predictions_log.c.base_time == base_time
                )
            ).scalars()
        )


def test_stored_model_is_reused_until_retraining_is_due(network, monkeypatch):
    clock = datetime(2026, 11, 24, 12, 0)
    monkeypatch.setattr(forecast, "utcnow", lambda: clock)
    forecast.predict_next_hour()
    first_version = forecast.load_model()[1]

    snapshot = network + timedelta(minutes=10)
    _add_snapshot(snapshot, seed=2)
    monkeypatch.setattr(forecast, "utcnow", lambda: clock + timedelta(minutes=30))
    assert forecast.predict_next_hour() == 20
    assert _count(db.model_runs) == 1
    assert _versions_at(snapshot) == {first_version}

    snapshot = network + timedelta(minutes=20)
    _add_snapshot(snapshot, seed=3)
    monkeypatch.setattr(forecast, "utcnow", lambda: clock + forecast.RETRAIN_EVERY)
    assert forecast.predict_next_hour() == 20
    assert _count(db.model_runs) == 2
    assert _versions_at(snapshot) == {forecast.load_model()[1]} != {first_version}


def test_model_trains_only_on_the_recent_window(network, monkeypatch):
    monkeypatch.setattr(forecast, "TRAIN_WINDOW", timedelta(hours=18))
    forecast.predict_next_hour()

    with db.get_engine().connect() as conn:
        run = conn.execute(select(db.model_runs)).one()
    # 109 snapshots lie in the window; the last 6 have no outcome yet.
    assert run.n_train + run.n_valid == 20 * (109 - 6)
