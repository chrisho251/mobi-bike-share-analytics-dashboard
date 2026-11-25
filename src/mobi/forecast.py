"""Forecasts bikes per station 60 minutes ahead and corrects its own mistakes.

Each pipeline run does two things:

1. ``fill_actuals`` looks up what really happened for forecasts whose target
   time has passed, so every forecast can be scored.
2. ``predict_next_hour`` retrains the model on the last TRAIN_WINDOW of history
   once per RETRAIN_EVERY (reusing the stored model in between), forecasts the
   next hour for every station and stores three competing forecasts:

   * baseline:  the bike count stays as it is now (persistence);
   * model:     gradient-boosted trees predicting the change in bike count;
   * corrected: per station, the model if it beat the baseline over the last
     24 hours, otherwise the baseline. Stations where the model keeps being
     wrong are therefore handed back to the baseline automatically, and
     handed to the model again once it does better.

Adding each station's average recent error to the model ("bias correction")
was tried first and rejected: on real data it raised the error from 0.66 to
0.88 bikes, because most stations barely change within an hour, so the
recent error is mostly noise.
"""

import logging
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, func, select, update

from mobi import features
from mobi.db import (
    get_engine,
    model_runs,
    model_store,
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
# Only forecasts scored in this window decide, per station, whether the model
# or the baseline is trusted.
# 24 hours kept 93 % of the model's gain on simulated data and limited the
# loss to 1 % on real data where the model was worse; 6 hours was too noisy.
SELECTION_WINDOW = timedelta(hours=24)
# A station needs at least this many scored forecasts in the window before the
# model can be trusted there.
SELECTION_MIN_FORECASTS = 6
# Forecasts older than this are not scanned for actuals any more.
ACTUAL_LOOKBACK = timedelta(hours=12)
# The model learns from this much recent history only. Four weeks hold four of
# each weekday, and the cost of a training run stops growing with the history.
TRAIN_WINDOW = timedelta(days=28)
# How often the model is retrained. The runs in between reuse the stored model
# and read only the last hour of snapshots.
RETRAIN_EVERY = timedelta(hours=1)


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


def _make_model():
    """Creates an untrained change-in-bikes regressor.

    Absolute-error loss predicts the median change, which is 0 for most
    stations; squared error predicted small changes everywhere and lost to
    persistence. Early stopping sizes the model to the signal in the data: on
    two days of real data it stops after a few trees (about persistence), on
    three weeks of simulated data it grows to ~200 trees and beats it by ~9 %.
    """
    # Imported here so the dashboard, which never trains, does not need it.
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        categorical_features=["area_code"],
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=0,
    )


def _fit(frame: pd.DataFrame):
    """Fits a model on the feature columns that hold at least one value.

    Lag features are empty while the history is shorter than the lag, and the
    model cannot bin an empty column, so such columns are left out. The fitted
    model remembers its columns in feature_names_in_.

    Args:
        frame: Feature rows with bikes_target.

    Returns:
        The fitted model.
    """
    columns = [c for c in features.FEATURE_COLUMNS if frame[c].notna().any()]
    return _make_model().fit(frame[columns], frame["bikes_target"] - frame["bikes_now"])


def train_model(frame: pd.DataFrame):
    """Trains the change-in-bikes model and scores it on the newest data.

    The model learns delta = bikes in 60 minutes - bikes now. It is first fit
    on the older part of the history and scored on the newest part (the future
    is never used to predict the past), then refit on everything.

    Args:
        frame: Feature rows that already have a bikes_target.

    Returns:
        A tuple (model, metrics). The model is None and metrics is empty when
        there are fewer than MIN_TRAIN_ROWS examples.
    """
    if len(frame) < MIN_TRAIN_ROWS:
        return None, {}

    cutoff = frame["snapshot_time"].quantile(1 - VALID_FRACTION)
    train = frame[frame["snapshot_time"] < cutoff]
    valid = frame[frame["snapshot_time"] >= cutoff]
    metrics = {"n_train": len(train), "n_valid": len(valid)}
    if len(train) >= MIN_TRAIN_ROWS // 2 and not valid.empty:
        scored = forecast(_fit(train), valid)
        metrics["mae_model"] = float((valid["bikes_target"] - scored["pred_model"]).abs().mean())
        metrics["mae_baseline"] = float((valid["bikes_target"] - valid["bikes_now"]).abs().mean())
    return _fit(frame), metrics


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


def forecast(model, rows: pd.DataFrame) -> pd.DataFrame:
    """Adds the baseline and model forecasts to feature rows.

    Args:
        model: A model from train_model, or None to fall back to the baseline.
        rows: Feature rows (one per station and base time).

    Returns:
        A copy of rows with pred_baseline and pred_model columns.
    """
    rows = rows.copy()
    rows["pred_baseline"] = rows["bikes_now"].astype(float)
    if model is None:
        rows["pred_model"] = rows["pred_baseline"]
    else:
        delta = model.predict(rows[list(model.feature_names_in_)])
        rows["pred_model"] = _clip(rows["bikes_now"] + delta, rows["capacity"])
    return rows


def trusted_stations(scored: pd.DataFrame) -> set[str]:
    """Finds the stations where the model recently beat the baseline.

    Args:
        scored: Recent forecasts with station_id, pred_baseline, pred_model and
            a known actual.

    Returns:
        The ids of stations with at least SELECTION_MIN_FORECASTS forecasts
        whose mean model error is lower than their mean baseline error.
    """
    if scored.empty:
        return set()
    errors = pd.DataFrame(
        {
            "station_id": scored["station_id"],
            "model": (scored["actual"] - scored["pred_model"]).abs(),
            "baseline": (scored["actual"] - scored["pred_baseline"]).abs(),
        }
    )
    by_station = errors.groupby("station_id").agg(
        model=("model", "mean"), baseline=("baseline", "mean"), n=("model", "size")
    )
    trusted = (by_station["n"] >= SELECTION_MIN_FORECASTS) & (
        by_station["model"] < by_station["baseline"]
    )
    return set(by_station.index[trusted])


def correct(rows: pd.DataFrame, trusted: set[str]) -> pd.DataFrame:
    """Adds pred_corrected: the model where trusted, the baseline elsewhere.

    Args:
        rows: Output of forecast().
        trusted: Output of trusted_stations().

    Returns:
        A copy of rows with a pred_corrected column.
    """
    rows = rows.copy()
    use_model = rows["station_id"].isin(trusted)
    rows["pred_corrected"] = rows["pred_model"].where(use_model, rows["pred_baseline"])
    return rows


def to_log_rows(rows: pd.DataFrame, version: str, created_at) -> list[dict]:
    """Converts corrected forecasts into predictions_log records.

    Args:
        rows: Output of correct(), with station_id and snapshot_time.
        version: The model_version label.
        created_at: When the forecasts were made.

    Returns:
        One dict per row, ready for upsert into predictions_log.
    """
    return [
        {
            "station_id": row.station_id,
            "base_time": row.snapshot_time.to_pydatetime(),
            "target_time": (row.snapshot_time + features.HORIZON).to_pydatetime(),
            "model_version": version,
            "bikes_now": int(row.bikes_now),
            "pred_baseline": float(row.pred_baseline),
            "pred_model": float(row.pred_model),
            "pred_corrected": float(row.pred_corrected),
            "created_at": created_at,
        }
        for row in rows.itertuples()
    ]


def _recent_scored(latest) -> pd.DataFrame:
    """Reads the forecasts scored within SELECTION_WINDOW before latest."""
    return features.read_query(
        select(
            predictions_log.c.station_id,
            predictions_log.c.pred_baseline,
            predictions_log.c.pred_model,
            predictions_log.c.actual,
        ).where(
            predictions_log.c.actual.is_not(None),
            predictions_log.c.target_time > latest - SELECTION_WINDOW,
        )
    )


def _library_version() -> str:
    """Returns the installed scikit-learn version."""
    import sklearn

    return sklearn.__version__


def save_model(model, version: str, trained_at: datetime) -> None:
    """Stores a trained model as the current one, replacing the previous.

    Args:
        model: A model from train_model (not None).
        version: Its model_version label.
        trained_at: When it was trained (naive UTC).
    """
    upsert(
        model_store,
        [
            {
                "name": "current",
                "model_version": version,
                "trained_at": trained_at,
                "library_version": _library_version(),
                "model": pickle.dumps(model),
            }
        ],
        ["name"],
        update_cols=["model_version", "trained_at", "library_version", "model"],
    )


def load_model():
    """Reads the current model written by save_model.

    The pickle was written by this pipeline into its own database, so it is
    trusted.

    Returns:
        A tuple (model, version, trained_at), or None if no model is stored or
        it was saved by another scikit-learn version.
    """
    with get_engine().connect() as conn:
        row = conn.execute(select(model_store).where(model_store.c.name == "current")).first()
    if row is None:
        return None
    if row.library_version != _library_version():
        logger.info(
            "stored model %s needs scikit-learn %s; retraining",
            row.model_version, row.library_version,
        )
        return None
    return pickle.loads(row.model), row.model_version, row.trained_at


def _build_features(since) -> pd.DataFrame:
    """Builds feature rows from the snapshots at or after since."""
    return features.build_features(
        features.load_status(since=since), features.load_stations(), features.load_weather()
    )


def predict_next_hour() -> int:
    """Logs the 60-minute forecast for every station, retraining when due.

    The model is retrained on the last TRAIN_WINDOW of snapshots when no model
    is stored or the stored one is at least RETRAIN_EVERY old. Otherwise the
    stored model is reused and only the snapshots needed for the newest
    features are read.

    Returns:
        The number of forecast rows written (0 if this snapshot was already
        forecast or no data exists yet).
    """
    latest = _latest_snapshot_time()
    if latest is None:
        return 0
    with get_engine().connect() as conn:
        already_forecast = conn.execute(
            select(func.count()).select_from(predictions_log).where(predictions_log.c.base_time == latest)
        ).scalar()
    if already_forecast:
        # The feed has not refreshed since the last run: nothing new to forecast.
        return 0
    now = utcnow()

    stored = load_model()
    if stored is not None and now - stored[2] < RETRAIN_EVERY:
        model, version, _ = stored
        frame = _build_features(since=latest - features.LOOKBACK)
    else:
        frame = _build_features(since=latest - TRAIN_WINDOW)
        model, metrics = train_model(frame[frame["bikes_target"].notna()])
        version = f"hgb-{now:%Y%m%d%H%M}" if model is not None else "baseline-only"
        if model is not None:
            logger.info("trained %s: %s", version, metrics)
            upsert(
                model_runs,
                [{"model_version": version, "trained_at": now, **metrics}],
                ["model_version"],
            )
            save_model(model, version, now)

    current = frame[frame["snapshot_time"] == pd.Timestamp(latest)]
    if current.empty:
        return 0
    rows = correct(forecast(model, current), trusted_stations(_recent_scored(latest)))
    return upsert(predictions_log, to_log_rows(rows, version, now), ["station_id", "base_time"])
