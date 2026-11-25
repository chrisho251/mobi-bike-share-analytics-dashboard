"""Tests for the daily accuracy summary."""

from datetime import date, datetime, timedelta

import pandas as pd
from conftest import START, status_rows
from sqlalchemy import select

from mobi import db, summary


def test_days_follow_vancouver_time_including_clock_changes():
    # British Columbia has stayed on UTC-7 since the clocks went forward in
    # March 2026 (time zone database 2026).
    start, end = summary.day_bounds(date(2026, 11, 2))
    assert (start, end) == (datetime(2026, 11, 2, 7), datetime(2026, 11, 3, 7))

    start, end = summary.day_bounds(date(2026, 3, 8))  # clocks went forward
    assert end - start == timedelta(hours=23)


def test_a_day_is_summarised_into_errors_and_alert_outcomes():
    scored = pd.DataFrame(
        {
            "capacity": 20,
            "actual": [0, 10, 20, 10],
            "pred_baseline": [4, 10, 16, 10],
            "pred_model": [2, 10, 18, 12],
            "pred_corrected": [1, 2, 19, 10],
        }
    )

    row = summary.summarise_day(scored)

    assert row["forecasts"] == 4
    assert (row["mae_baseline"], row["mae_model"], row["mae_corrected"]) == (2.0, 1.5, 2.5)
    # Empty soon: two alerts (1 and 2 bikes), one came true; one station ran empty.
    assert (row["empty_alerts"], row["empty_alerts_true"], row["empty_events"]) == (2, 1, 1)
    assert (row["full_alerts"], row["full_alerts_true"], row["full_events"]) == (1, 1, 1)


def _forecasts(day_start: datetime, hours: int) -> list[dict]:
    return [
        {
            "station_id": "s00",
            "base_time": day_start + timedelta(hours=h),
            "target_time": day_start + timedelta(hours=h + 1),
            "model_version": "test",
            "bikes_now": 5,
            "pred_baseline": 5.0,
            "pred_model": 6.0,
            "pred_corrected": 6.0,
            "actual": 7,
        }
        for h in range(hours)
    ]


def _days() -> list:
    with db.get_engine().connect() as conn:
        return conn.execute(select(db.accuracy_daily.c.day, db.accuracy_daily.c.forecasts)).all()


def test_only_finished_days_are_summarised_once(network):
    # START is 01:00 in Vancouver on 2 November; data runs to 01:00 on the 4th.
    with db.get_engine().begin() as conn:
        conn.execute(db.predictions_log.insert(), _forecasts(START, 47))

    # The 3rd ended less than 12 hours before the newest snapshot: still open.
    assert summary.summarise_days() == 1
    assert _days() == [(date(2026, 11, 2), 22)]  # forecasts due 02:00 to 23:00
    assert summary.summarise_days() == 0

    later = network + timedelta(hours=12)
    with db.get_engine().begin() as conn:
        conn.execute(db.station_status.insert(), status_rows(later, later))

    assert summary.summarise_days() == 1
    assert _days()[-1] == (date(2026, 11, 3), 24)


def test_dashboard_reads_nothing_before_the_table_exists(sqlite_db):
    with sqlite_db.begin() as conn:
        db.accuracy_daily.drop(conn)

    assert summary.load_daily().empty
