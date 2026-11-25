"""Keeps a daily record of forecast accuracy that outlives the raw forecasts.

Retention deletes forecasts after two weeks, so every finished day is
summarised once into accuracy_daily: how many forecasts were scored, the mean
error of each forecast, and how the corrected forecast's at-risk alerts turned
out. Days are Vancouver days, as on the dashboard.

A day is summarised once no forecast in it can still be scored: scoring looks
back forecast.ACTUAL_LOOKBACK, so the day must have ended at least that long
before the newest snapshot.
"""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, inspect, select

from mobi import actions, features
from mobi.db import accuracy_daily, get_engine, predictions_log, station_status, upsert
from mobi.forecast import ACTUAL_LOOKBACK
from mobi.ingest import utcnow

LOCAL = ZoneInfo(features.LOCAL_TZ)


def _local_date(utc: datetime) -> date:
    """Returns the Vancouver date of a naive UTC time."""
    return utc.replace(tzinfo=timezone.utc).astimezone(LOCAL).date()


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """Returns the start and end of a Vancouver day as naive UTC times.

    Days are 23 or 25 hours long when the clocks change.
    """
    start, end = (
        datetime.combine(d, time(), tzinfo=LOCAL).astimezone(timezone.utc).replace(tzinfo=None)
        for d in (day, day + timedelta(days=1))
    )
    return start, end


def summarise_day(scored: pd.DataFrame) -> dict:
    """Summarises one day of scored forecasts.

    Args:
        scored: Forecasts with actual, pred_baseline, pred_model,
            pred_corrected and capacity.

    Returns:
        The accuracy_daily columns except day and created_at.
    """
    row = {"forecasts": len(scored)}
    for name in ("baseline", "model", "corrected"):
        row[f"mae_{name}"] = float((scored["actual"] - scored[f"pred_{name}"]).abs().mean())
    usable = scored[scored["capacity"] > 0]
    actual = usable["actual"] / usable["capacity"]
    forecast = usable["pred_corrected"] / usable["capacity"]
    alerts = {
        "empty": lambda fill: fill < actions.SHORT_FILL,
        "full": lambda fill: fill > actions.FULL_FILL,
    }
    for name, at_risk in alerts.items():
        raised, happened = at_risk(forecast), at_risk(actual)
        row[f"{name}_alerts"] = int(raised.sum())
        row[f"{name}_alerts_true"] = int((raised & happened).sum())
        row[f"{name}_events"] = int(happened.sum())
    return row


def _scored_between(start: datetime, end: datetime) -> pd.DataFrame:
    """Reads the scored forecasts whose target time is in [start, end)."""
    p = predictions_log.c
    scored = features.read_query(
        select(p.station_id, p.actual, p.pred_baseline, p.pred_model, p.pred_corrected).where(
            p.actual.is_not(None), p.target_time >= start, p.target_time < end
        )
    )
    capacity = features.load_stations()[["station_id", "capacity"]]
    return scored.merge(capacity, on="station_id", how="left").fillna({"capacity": 0})


def summarise_days() -> int:
    """Summarises every finished day that has no summary yet.

    Starts after the newest summarised day, or at the first forecast when there
    is none, so a new database is backfilled from whatever forecasts it holds.

    Returns:
        The number of days written.
    """
    with get_engine().connect() as conn:
        latest = conn.execute(select(func.max(station_status.c.snapshot_time))).scalar()
        done = conn.execute(select(func.max(accuracy_daily.c.day))).scalar()
        first_target = conn.execute(select(func.min(predictions_log.c.target_time))).scalar()
    if latest is None or first_target is None:
        return 0
    last_finished = _local_date(latest - ACTUAL_LOOKBACK) - timedelta(days=1)
    day = done + timedelta(days=1) if done is not None else _local_date(first_target)
    now = utcnow()
    rows = []
    while day <= last_finished:
        scored = _scored_between(*day_bounds(day))
        if not scored.empty:
            rows.append({"day": day, **summarise_day(scored), "created_at": now})
        day += timedelta(days=1)
    return upsert(accuracy_daily, rows, ["day"])


def load_daily() -> pd.DataFrame:
    """Reads the daily accuracy record for the dashboard, oldest day first.

    The dashboard only reads, so it cannot create the table: a database the
    pipeline has not updated yet simply has no record.

    Returns:
        The accuracy_daily rows, or an empty frame if there are none.
    """
    if not inspect(get_engine()).has_table(accuracy_daily.name):
        return pd.DataFrame()
    return features.read_query(select(accuracy_daily).order_by(accuracy_daily.c.day))
