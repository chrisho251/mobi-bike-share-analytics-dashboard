"""Tests for the retention step of the pipeline."""

from datetime import timedelta

from conftest import START
from sqlalchemy import func, select

from mobi import db, pipeline


def _prediction(base_time) -> dict:
    return {
        "station_id": "s00",
        "base_time": base_time,
        "target_time": base_time + timedelta(hours=1),
        "model_version": "baseline-only",
        "bikes_now": 5,
        "pred_baseline": 5.0,
        "pred_model": 5.0,
        "pred_corrected": 5.0,
    }


def test_prune_deletes_only_rows_past_their_retention(network, monkeypatch):
    monkeypatch.setattr(pipeline, "STATUS_RETENTION", timedelta(days=1))
    monkeypatch.setattr(pipeline, "PREDICTION_RETENTION", timedelta(hours=6))
    with db.get_engine().begin() as conn:
        conn.execute(
            db.predictions_log.insert(),
            [_prediction(START), _prediction(network - timedelta(hours=2))],
        )

    # One day of snapshots (144 per station) and the old forecast go.
    assert pipeline.prune_history() == 20 * 144 + 1

    with db.get_engine().connect() as conn:
        oldest = conn.execute(select(func.min(db.station_status.c.snapshot_time))).scalar()
        kept = conn.execute(select(db.predictions_log.c.base_time)).scalars().all()
    assert oldest == network - timedelta(days=1)
    assert kept == [network - timedelta(hours=2)]
    assert pipeline.prune_history() == 0


def test_snapshots_outlive_the_training_window():
    assert pipeline.STATUS_RETENTION > pipeline.TRAIN_WINDOW
