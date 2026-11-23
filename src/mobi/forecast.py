"""Forecasts bikes per station 60 minutes ahead and corrects its own mistakes.

Each pipeline run does two things:

1. ``fill_actuals`` looks up what really happened for forecasts whose target
   time has passed, so every forecast can be scored.
2. ``predict_next_hour`` retrains the model on all history, forecasts the next
   hour for every station and stores three competing forecasts:

   * baseline:  the bike count stays as it is now (persistence);
   * model:     gradient-boosted trees predicting the change in bike count;
   * corrected: the model plus that station's average recent error, so
     systematic mistakes are cancelled out as actuals arrive.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, func, select, update

from mobi import features
from mobi.db import (
    get_engine,
    model_runs,
    predictions_log,
    station_status,
    upsert,
)
from mobi.ingest import utcnow

logger = logging.getLogger("forecast")

# Below this many labelled examples the model is skipped and only the
# baseline is logged, so the tables start filling from the very first run.
MIN_TRAIN_ROWS = 2000
# Share of the most recent snapshot times held out to score the model.
VALID_FRACTION = 0.2
# Only errors made in this window are used to correct the next forecasts.
BIAS_WINDOW = timedelta(hours=3)
# Forecasts older than this are not scanned for actuals any more.
ACTUAL_LOOKBACK = timedelta(hours=12)


def _latest_snapshot_time():
    """Returns the newest snapshot time in the database, or None if empty."""
    with get_engine().connect() as conn:
        return conn.execute(select(func.max(station_status.c.snapshot_time))).scalar()


def fill_actuals() -> int:
    """Stores the real bike count for forecasts whose target time has passed.

    For every forecast without an actual value and with target_time up to the
    newest snapshot, the snapshot closest to target_time (within the matching
    tolerance) is looked up and written into the actual column.

    Returns:
        The number of forecasts that received an actual value.
    """
    latest = _latest_snapshot_time()
    if latest is None:
        return 0
    oldest = latest - ACTUAL_LOOKBACK
    pending = features.read_query(
        select(
            predictions_log.c.station_id,
            predictions_log.c.base_time,
            predictions_log.c.target_time,
        ).where(
            predictions_log.c.actual.is_(None),
            predictions_log.c.target_time <= latest,
            predictions_log.c.target_time >= oldest,
        )
    )
    if pending.empty:
        return 0
    pending = features._to_ns(pending, ["base_time", "target_time"])
    status = features.load_status(since=oldest - features.MATCH_TOLERANCE)
    observations = (
        status.rename(columns={"snapshot_time": "obs_time", "num_bikes": "value"})
        .sort_values("obs_time")
    )
    matched = pd.merge_asof(
        pending.sort_values("target_time"),
        observations,
        left_on="target_time",
        right_on="obs_time",
        by="station_id",
        tolerance=features.MATCH_TOLERANCE,
        direction="nearest",
    ).dropna(subset=["value"])
    if matched.empty:
        return 0

    records = [
        {
            "b_station": row.station_id,
            "b_base": row.base_time.to_pydatetime(),
            "b_actual": int(row.value),
        }
        for row in matched.itertuples()
    ]
    statement = (
        update(predictions_log)
        .where(
            predictions_log.c.station_id == bindparam("b_station"),
            predictions_log.c.base_time == bindparam("b_base"),
        )
        .values(actual=bindparam("b_actual"))
    )
    with get_engine().begin() as conn:
        conn.execute(statement, records)
    return len(records)


def _train(frame: pd.DataFrame):
    """Trains the change-in-bikes model and scores it on the newest data.

    The model learns delta = bikes in 60 minutes - bikes now. It is first fit
    on the older part of the history and scored on the newest part (a fair
    test, since the future is never used to predict the past), then refit on
    everything for the actual forecasts.

    Args:
        frame: Feature rows that already have a bikes_target.

    Returns:
        A tuple (model, metrics). The model is None, and metrics is empty, when
        there are fewer than MIN_TRAIN_ROWS examples or the history is still too
        short to compute every feature.
    """
    # Imported here so the dashboard, which never trains, does not need it.
    from sklearn.ensemble import HistGradientBoostingRegressor

    if len(frame) < MIN_TRAIN_ROWS:
        return None, {}

    frame = frame.sort_values("snapshot_time")
    cutoff = frame["snapshot_time"].quantile(1 - VALID_FRACTION)
    train = frame[frame["snapshot_time"] < cutoff]
    valid = frame[frame["snapshot_time"] >= cutoff]
    if not train[features.FEATURE_COLUMNS].notna().any().all():
        # A feature with no value at all (history too short for the lags yet)
        # cannot be binned by the model.
        return None, {}

    def make_model():
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            categorical_features=["area_code"],
            early_stopping=False,
            random_state=0,
        )

    metrics = {"n_train": len(train), "n_valid": len(valid)}
    if len(train) >= MIN_TRAIN_ROWS // 2 and not valid.empty:
        scored = make_model().fit(
            train[features.FEATURE_COLUMNS], train["bikes_target"] - train["bikes_now"]
        )
        predicted = _clip(
            valid["bikes_now"] + scored.predict(valid[features.FEATURE_COLUMNS]),
            valid["capacity"],
        )
        metrics["mae_model"] = float((valid["bikes_target"] - predicted).abs().mean())
        metrics["mae_baseline"] = float(
            (valid["bikes_target"] - valid["bikes_now"]).abs().mean()
        )

    model = make_model().fit(
        frame[features.FEATURE_COLUMNS], frame["bikes_target"] - frame["bikes_now"]
    )
    return model, metrics


def _clip(values: pd.Series, capacity: pd.Series) -> pd.Series:
    """Limits forecasts to the physically possible range 0..capacity.

    Args:
        values: Forecast bike counts.
        capacity: Dock capacity per row; NaN means unknown (no upper limit).

    Returns:
        The clipped forecasts.
    """
    upper = capacity.where(capacity > 0, np.inf).fillna(np.inf)
    return values.clip(lower=0).clip(upper=upper)


def _recent_bias(latest) -> pd.Series:
    """Computes each station's average recent model error.

    Args:
        latest: The newest snapshot time; errors are taken from the BIAS_WINDOW
            before it.

    Returns:
        A series indexed by station_id with mean(actual - pred_model), for
        stations that have at least one scored forecast in the window.
    """
    scored = features.read_query(
        select(
            predictions_log.c.station_id,
            predictions_log.c.pred_model,
            predictions_log.c.actual,
        ).where(
            predictions_log.c.actual.is_not(None),
            predictions_log.c.target_time > latest - BIAS_WINDOW,
        )
    )
    if scored.empty:
        return pd.Series(dtype=float)
    scored["error"] = scored["actual"] - scored["pred_model"]
    return scored.groupby("station_id")["error"].mean()


def predict_next_hour() -> int:
    """Retrains the model and logs the 60-minute forecast for every station.

    Returns:
        The number of forecast rows written (0 if this snapshot was already
        forecast or no data exists yet).
    """
    latest = _latest_snapshot_time()
    if latest is None:
        return 0
    now = utcnow()

    frame = features.build_features(
        features.load_status(), features.load_stations(), features.load_weather()
    )
    model, metrics = _train(frame[frame["bikes_target"].notna()])
    version = f"hgb-{now:%Y%m%d%H%M}" if model is not None else "baseline-only"
    if model is not None:
        logger.info("trained %s: %s", version, metrics)
        upsert(
            model_runs,
            [{"model_version": version, "trained_at": now, **metrics}],
            ["model_version"],
        )

    current = frame[frame["snapshot_time"] == pd.Timestamp(latest)].copy()
    if current.empty:
        return 0
    current["pred_baseline"] = current["bikes_now"].astype(float)
    if model is not None:
        delta = model.predict(current[features.FEATURE_COLUMNS])
        current["pred_model"] = _clip(current["bikes_now"] + delta, current["capacity"])
    else:
        current["pred_model"] = current["pred_baseline"]
    bias = current["station_id"].map(_recent_bias(latest)).fillna(0.0)
    current["pred_corrected"] = _clip(current["pred_model"] + bias, current["capacity"])

    rows = [
        {
            "station_id": row.station_id,
            "base_time": latest,
            "target_time": latest + features.HORIZON,
            "model_version": version,
            "bikes_now": int(row.bikes_now),
            "pred_baseline": float(row.pred_baseline),
            "pred_model": float(row.pred_model),
            "pred_corrected": float(row.pred_corrected),
            "created_at": now,
        }
        for row in current.itertuples()
    ]
    return upsert(predictions_log, rows, ["station_id", "base_time"])
