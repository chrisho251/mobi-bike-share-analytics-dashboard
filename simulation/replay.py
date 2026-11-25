"""Replays the live forecasting loop over the simulated history.

Every 10-minute snapshot is forecast exactly as the pipeline would have done
live: the model only sees data whose outcome was known at that moment, it is
retrained on a fixed cadence, and the per-station model/baseline selection uses
only forecasts already scored. Results are written to the simulation
database's predictions_log and model_runs, so the dashboard can show them.

Usage:
    uv run python simulation/replay.py
"""

import argparse
import os
from collections import deque
from datetime import timedelta
from pathlib import Path

import pandas as pd

SIM_DIR = Path(__file__).resolve().parent
DEFAULT_DB = SIM_DIR / "data" / "mobi_sim.db"


def parse_args() -> argparse.Namespace:
    """Reads the command-line options.

    Returns:
        The parsed options (db, retrain_hours, days).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--retrain-hours", type=int, default=24)
    parser.add_argument(
        "--days", type=int, default=None, help="replay only the last N days"
    )
    return parser.parse_args()


def replay(frame: pd.DataFrame, times, retrain_every: timedelta):
    """Runs the forecast loop over the given snapshot times.

    Args:
        frame: Output of features.build_features for the whole history.
        times: Sorted snapshot times to forecast from.
        retrain_every: How often the model is retrained.

    Returns:
        A tuple (forecasts, runs): one row per station and snapshot with the
        three forecasts and the outcome, and one row per training run.
    """
    from mobi import forecast
    from mobi.features import HORIZON

    labelled = frame[frame["bikes_target"].notna()]
    by_time = dict(tuple(frame.groupby("snapshot_time")))
    # Forecasts can be scored once their target time has passed, so the
    # selection window reaches back SELECTION_WINDOW plus the horizon.
    recent = deque(maxlen=int((forecast.SELECTION_WINDOW + HORIZON) / timedelta(minutes=10)) + 1)
    model, last_train, outputs, runs = None, None, [], []

    for time in times:
        if last_train is None or time - last_train >= retrain_every:
            known = labelled[
                (labelled["snapshot_time"] + HORIZON <= time)
                & (labelled["snapshot_time"] >= time - forecast.TRAIN_WINDOW)
            ]
            model, metrics = forecast.train_model(known)
            last_train = time
            if model is not None:
                runs.append({"model_version": f"sim-{time:%Y%m%d%H%M}", "trained_at": time, **metrics})
        version = runs[-1]["model_version"] if model is not None else "baseline-only"

        scored = pd.concat(list(recent)) if recent else pd.DataFrame()
        if not scored.empty:
            scored = scored[scored["snapshot_time"] + HORIZON <= time].rename(
                columns={"bikes_target": "actual"}
            )
            scored = scored[scored["actual"].notna()]
        rows = forecast.correct(
            forecast.forecast(model, by_time[time]), forecast.trusted_stations(scored)
        )
        rows["model_version"] = version
        kept = rows[
            ["station_id", "snapshot_time", "bikes_now", "bikes_target", "capacity",
             "pred_baseline", "pred_model", "pred_corrected", "model_version"]
        ]
        recent.append(kept)
        outputs.append(kept)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(runs)


def save(forecasts: pd.DataFrame, runs: pd.DataFrame) -> None:
    """Replaces predictions_log and model_runs with the replay results.

    Args:
        forecasts: First output of replay().
        runs: Second output of replay().
    """
    from mobi import db, forecast

    records = []
    for version, group in forecasts.groupby("model_version"):
        rows = forecast.to_log_rows(group, version, None)
        for record, actual in zip(rows, group["bikes_target"]):
            record["created_at"] = record["base_time"]
            record["actual"] = None if pd.isna(actual) else int(actual)
        records.extend(rows)
    with db.get_engine().begin() as conn:
        conn.execute(db.predictions_log.delete())
        conn.execute(db.model_runs.delete())
        conn.execute(db.predictions_log.insert(), records)
        if not runs.empty:
            conn.execute(
                db.model_runs.insert(),
                [{**r, "trained_at": pd.Timestamp(r["trained_at"]).to_pydatetime()} for r in runs.to_dict("records")],
            )


def summarise(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Computes the daily mean absolute error of each forecast type.

    Args:
        forecasts: First output of replay().

    Returns:
        One row per day with baseline, model and corrected errors (bikes).
    """
    scored = forecasts[forecasts["bikes_target"].notna()].copy()
    scored["day"] = scored["snapshot_time"].dt.date
    for name in ("baseline", "model", "corrected"):
        scored[name] = (scored["bikes_target"] - scored[f"pred_{name}"]).abs()
    return scored.groupby("day")[["baseline", "model", "corrected"]].mean().round(3)


def main() -> None:
    """Replays the forecasts and prints the daily error table."""
    args = parse_args()
    os.environ["DATABASE_URL"] = f"sqlite:///{args.db.as_posix()}"
    from mobi import db, features

    assert db.get_engine().dialect.name == "sqlite", "replay must use the simulation DB"
    frame = features.build_features(
        features.load_status(), features.load_stations(), features.load_weather()
    )
    times = sorted(frame["snapshot_time"].unique())
    if args.days is not None:
        start = times[-1] - pd.Timedelta(days=args.days)
        times = [t for t in times if t >= start]
    forecasts, runs = replay(frame, [pd.Timestamp(t) for t in times], timedelta(hours=args.retrain_hours))
    save(forecasts, runs)
    print(summarise(forecasts).to_string())
    print(f"{len(runs)} training runs, {len(forecasts):,} forecasts written to {args.db}")


if __name__ == "__main__":
    main()
