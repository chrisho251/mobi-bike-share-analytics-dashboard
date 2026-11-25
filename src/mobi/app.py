"""Gradio dashboard for Mobi dispatchers and the team running the pipeline.

Eleven panels in six tabs:

* Live Overview     - station map; empty/full stations over the last 24 h.
* Next-Hour Actions - suggested bike moves; stations at risk.
* Forecast Accuracy - error per hour; error per area; model training runs.
* Station Explorer  - one station's bikes and forecasts.
* Demand Patterns   - fill level by area and hour; activity in dry vs rain.
* Monitoring        - status board; daily data completeness; stations reporting.

Run with: uv run --extra app python -m mobi.app
"""

import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mobi import actions, config, features, queries, summary
from mobi.ingest import utcnow

REFRESH_SECONDS = 60
MAX_MOVES_SHOWN = 25
IS_SIMULATION = config.is_simulation()

# Forecast colours are fixed per series: the baseline is a neutral reference.
SERIES_COLORS = {"baseline": "#8a8984", "model": "#2a78d6", "corrected": "#eb6834"}
WEATHER_COLORS = {"Dry": "#eda100", "Rain": "#2a78d6"}
PERCENT_COLORBAR = {
    "title": {"text": "Full (%)", "side": "right"},
    "tickvals": [0, 25, 50, 75, 100],
}
EMPTY_MESSAGE = "No data yet: the collector has not stored any snapshot."
NO_FORECAST_MESSAGE = (
    "No scored forecasts yet. Forecasts are compared with reality one hour "
    "after they are made."
)


def _style(figure: go.Figure, height: int = 380) -> go.Figure:
    """Applies the shared layout: white background, light grid, legend top right.

    Args:
        figure: The figure to style.
        height: Height in pixels.

    Returns:
        The same figure.
    """
    figure.update_layout(
        height=height,
        template="plotly_white",
        margin={"l": 50, "r": 20, "t": 30, "b": 40},
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1, "xanchor": "right", "title_text": ""},
        hovermode="x unified",
    )
    return figure


def _local(timestamp) -> pd.Timestamp:
    """Converts a naive UTC timestamp to Vancouver time."""
    return pd.Timestamp(timestamp).tz_localize("UTC").tz_convert(queries.LOCAL_TZ)


def _minutes_ago(timestamp) -> int:
    """Returns how many minutes before now a naive UTC timestamp is."""
    return int((utcnow() - pd.Timestamp(timestamp)).total_seconds() // 60)


# ---------------------------------------------------------------- Live Overview


def live_kpis(frame: pd.DataFrame) -> str:
    """Summarises the network state in one Markdown line.

    Args:
        frame: Output of queries.overview().

    Returns:
        Data age, bikes available and empty/full station counts.
    """
    if frame.empty:
        return EMPTY_MESSAGE
    latest = frame["snapshot_time"].iloc[0]
    return (
        f"**Data from {_local(latest):%a %H:%M}** ({_minutes_ago(latest)} min ago) | "
        f"{len(frame)} stations | **{int(frame['num_bikes'].sum())}** bikes available | "
        f"**{int((frame['num_bikes'] == 0).sum())}** empty | "
        f"**{int((frame['num_docks'] == 0).sum())}** full"
    )


def station_map(frame: pd.DataFrame) -> go.Figure | None:
    """Draws every station on a map, coloured by how full it is.

    Args:
        frame: Output of queries.overview().

    Returns:
        A Plotly map, or None without data.
    """
    if frame.empty:
        return None
    frame = frame.assign(fill_percent=(frame["fill_ratio"] * 100).round())
    figure = px.scatter_map(
        frame,
        lat="lat",
        lon="lon",
        color="fill_percent",
        range_color=(0, 100),
        color_continuous_scale="Viridis",
        hover_name="name",
        hover_data={"num_bikes": True, "num_docks": True, "lat": False, "lon": False},
        labels={"fill_percent": "Full (%)", "num_bikes": "Bikes", "num_docks": "Free docks"},
        center={"lat": config.VANCOUVER_LAT, "lon": config.VANCOUVER_LON},
        zoom=11.6,
        height=520,
        map_style="open-street-map",
    )
    figure.update_traces(marker={"size": 11})
    figure.update_layout(margin={"l": 0, "r": 0, "t": 36, "b": 0}, coloraxis_colorbar=PERCENT_COLORBAR)
    return figure


def empty_full_chart() -> go.Figure | None:
    """Plots how many stations were empty or full over the last 24 hours.

    Returns:
        A line chart, or None without data.
    """
    timeline = queries.network_timeline()
    if timeline.empty:
        return None
    long = timeline.melt(id_vars="time", value_vars=["empty", "full"], var_name="state", value_name="stations")
    long["state"] = long["state"].map({"empty": "Empty (no bikes)", "full": "Full (no docks)"})
    figure = px.line(
        long,
        x="time",
        y="stations",
        color="state",
        color_discrete_sequence=["#e34948", "#6250d6"],
        labels={"time": "", "stations": "Stations"},
    )
    return _style(figure, 300)


# ------------------------------------------------------------ Next-Hour Actions


def action_tables(frame: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Builds the headline and both tables of the actions tab.

    Args:
        frame: Output of queries.overview().

    Returns:
        A tuple (headline Markdown, suggested moves, stations at risk).
    """
    if frame.empty or frame["pred_corrected"].isna().all():
        return "No forecast yet.", pd.DataFrame(), pd.DataFrame()
    risks = actions.at_risk_stations(frame)
    moves = actions.plan_moves(frame)
    counts = risks["risk"].value_counts() if not risks.empty else pd.Series(dtype=int)
    target = _local(frame["target_time"].dropna().iloc[0])
    headline = (
        f"**Forecast for {target:%H:%M}** | **{counts.get('Empty soon', 0)}** stations "
        f"running low, **{counts.get('Full soon', 0)}** nearly full | **{len(moves)}** "
        f"moves suggested, the {min(len(moves), MAX_MOVES_SHOWN)} most urgent shown"
    )
    return headline, moves.head(MAX_MOVES_SHOWN), risks


# ------------------------------------------------------------ Forecast Accuracy


def accuracy_headline(hourly: pd.DataFrame) -> str:
    """Summarises the error of each forecast type over the last 7 days.

    Args:
        hourly: Output of queries.accuracy().

    Returns:
        A Markdown line with the weighted mean errors and the gain over the
        baseline.
    """
    if hourly.empty:
        return NO_FORECAST_MESSAGE
    weights = hourly["n"]
    mae = hourly[["baseline", "model", "corrected"]].multiply(weights, axis=0).sum() / weights.sum()
    gain = (1 - mae["corrected"] / mae["baseline"]) * 100
    return (
        f"**Mean error over 7 days (bikes, lower is better)** | baseline "
        f"**{mae['baseline']:.3f}** | model **{mae['model']:.3f}** | corrected "
        f"**{mae['corrected']:.3f}** ({gain:+.1f}% vs baseline) | "
        f"{int(weights.sum()):,} forecasts scored"
    )


def accuracy_chart(hourly: pd.DataFrame) -> go.Figure | None:
    """Plots each forecast type's error per hour.

    Args:
        hourly: Output of queries.accuracy().

    Returns:
        A line chart, or None without data.
    """
    if hourly.empty:
        return None
    long = hourly.melt(id_vars="hour", value_vars=list(SERIES_COLORS), var_name="forecast", value_name="error")
    figure = px.line(
        long,
        x="hour",
        y="error",
        color="forecast",
        color_discrete_map=SERIES_COLORS,
        labels={"hour": "", "error": "Mean absolute error (bikes)"},
    )
    return _style(figure)


def area_error_chart() -> go.Figure | None:
    """Compares each forecast type's error by area over the last 24 hours.

    Returns:
        A grouped bar chart, or None without data.
    """
    by_area = queries.accuracy_by_area()
    if by_area.empty:
        return None
    long = by_area.melt(id_vars="area", var_name="forecast", value_name="error")
    figure = px.bar(
        long,
        x="area",
        y="error",
        color="forecast",
        barmode="group",
        color_discrete_map=SERIES_COLORS,
        labels={"area": "", "error": "Mean absolute error (bikes)"},
    )
    figure.update_layout(bargap=0.25, bargroupgap=0.05)
    return _style(figure, 340)


def training_chart() -> go.Figure | None:
    """Plots the held-out error of each training run, model versus baseline.

    Returns:
        A line chart, or None before the first training run.
    """
    runs = queries.model_history().dropna(subset=["mae_model"])
    if runs.empty:
        return None
    long = runs.melt(
        id_vars="trained_at", value_vars=["mae_baseline", "mae_model"], var_name="forecast", value_name="error"
    )
    long["forecast"] = long["forecast"].str.removeprefix("mae_")
    figure = px.line(
        long,
        x="trained_at",
        y="error",
        color="forecast",
        markers=True,
        color_discrete_map=SERIES_COLORS,
        labels={"trained_at": "Training run", "error": "Held-out error (bikes)"},
    )
    return _style(figure, 300)


def daily_chart() -> go.Figure | None:
    """Plots each forecast type's error per day since collection began.

    Reads the daily summary, which is kept after the raw forecasts are
    deleted, so this is the long-term record.

    Returns:
        A line chart, or None before the first full day is summarised.
    """
    daily = summary.load_daily()
    if daily.empty:
        return None
    long = daily.rename(columns=lambda c: c.removeprefix("mae_")).melt(
        id_vars="day", value_vars=list(SERIES_COLORS), var_name="forecast", value_name="error"
    )
    figure = px.line(
        long,
        x="day",
        y="error",
        color="forecast",
        markers=True,
        color_discrete_map=SERIES_COLORS,
        labels={"day": "", "error": "Mean absolute error (bikes)"},
    )
    return _style(figure, 300)


# ------------------------------------------------------------- Station Explorer


def station_chart(station_name: str) -> go.Figure | None:
    """Plots one station's bikes over the last 24 hours with its forecasts.

    Args:
        station_name: The station picked in the dropdown.

    Returns:
        A line-and-marker chart, or None if the station is unknown.
    """
    info = features.load_stations()
    match = info[info["name"] == station_name]
    if match.empty:
        return None
    history, forecasts = queries.station_history(match["station_id"].iloc[0])
    figure = go.Figure()
    figure.add_scatter(
        x=history.get("time"), y=history.get("num_bikes"), mode="lines",
        name="Actual bikes", line={"color": "#0b0b0b", "width": 2},
    )
    if not forecasts.empty:
        figure.add_scatter(
            x=forecasts["time"], y=forecasts["pred_corrected"], mode="markers",
            name="Forecast made 1 h earlier", marker={"color": SERIES_COLORS["corrected"], "size": 8},
        )
    figure.add_hline(
        y=match["capacity"].iloc[0], line_dash="dot", line_color="#8a8984",
        annotation_text="capacity",
    )
    figure.update_layout(yaxis_title="Bikes at station", yaxis_rangemode="tozero")
    return _style(figure, 420)


# -------------------------------------------------------------- Demand Patterns


def fill_heatmap() -> go.Figure | None:
    """Shows the average fill level of each area at each hour of the day.

    Returns:
        A heatmap, or None without data.
    """
    pivot = queries.fill_by_area_hour()
    if pivot.empty:
        return None
    figure = px.imshow(
        pivot.round(0),
        color_continuous_scale="Blues",
        range_color=(0, 100),
        aspect="auto",
        labels={"x": "Hour of day (Vancouver)", "y": "", "color": "Full (%)"},
        text_auto=True,
    )
    figure.update_xaxes(dtick=1)
    figure.update_layout(height=380, margin={"l": 20, "r": 20, "t": 44, "b": 40}, coloraxis_colorbar=PERCENT_COLORBAR)
    return figure


def activity_chart() -> go.Figure | None:
    """Compares network activity by hour in dry and rainy hours.

    Returns:
        A line chart, or None without data.
    """
    activity = queries.activity_by_hour()
    if activity.empty:
        return None
    figure = px.line(
        activity,
        x="local_hour",
        y="activity",
        color="weather",
        markers=True,
        color_discrete_map=WEATHER_COLORS,
        labels={"local_hour": "Hour of day (Vancouver)", "activity": "Avg. bikes changed per station, per 10 min"},
    )
    figure.update_xaxes(dtick=2)
    return _style(figure, 340)


# -------------------------------------------------------------------- Monitoring

# Status colours are reserved for health states and always paired with an
# icon and a word, so the state never depends on colour alone.
STATES = {
    "ok": {"color": "#0ca30c", "icon": "✓", "word": "OK"},
    "warn": {"color": "#fab219", "icon": "!", "word": "Check"},
    "bad": {"color": "#d03b3b", "icon": "✕", "word": "Problem"},
}
CARD_CSS = """
<style>
.mon-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px;}
.mon-card {border:1px solid #e3e2de; border-left:6px solid var(--state); border-radius:8px;
           padding:12px 14px; background:var(--block-background-fill, #fff);}
.mon-title {font-size:13px; opacity:.75;}
.mon-value {font-size:26px; font-weight:600; margin:2px 0;}
.mon-detail {font-size:12px; opacity:.75;}
.mon-state {font-size:12px; font-weight:600;}
</style>
"""


def _state(value: float, ok: float, warn: float, higher_is_better: bool = True) -> str:
    """Classifies a measure as ok, warn or bad against two thresholds.

    Args:
        value: The measured value.
        ok: Threshold for "ok".
        warn: Threshold for "warn"; anything beyond it is "bad".
        higher_is_better: Direction of the thresholds.

    Returns:
        "ok", "warn" or "bad".
    """
    if not higher_is_better:
        value, ok, warn = -value, -ok, -warn
    if value >= ok:
        return "ok"
    return "warn" if value >= warn else "bad"


def _card(title: str, value: str, detail: str, state: str) -> str:
    """Renders one status tile as HTML.

    Args:
        title: What is measured.
        value: The headline number.
        detail: One line of context.
        state: A key of STATES.

    Returns:
        An HTML fragment.
    """
    s = STATES[state]
    return (
        f'<div class="mon-card" style="--state:{s["color"]}">'
        f'<div class="mon-title">{title}</div><div class="mon-value">{value}</div>'
        f'<div class="mon-state" style="color:{s["color"]}">{s["icon"]} {s["word"]}</div>'
        f'<div class="mon-detail">{detail}</div></div>'
    )


def monitoring_cards() -> str:
    """Builds the status board: one tile per pipeline output.

    Returns:
        HTML with six tiles (data freshness, completeness, stations, weather,
        model, scoring backlog).
    """
    status = queries.pipeline_status().iloc[0]
    if pd.isna(status["last_snapshot"]):
        return f"<p>{EMPTY_MESSAGE}</p>"

    data_age = _minutes_ago(status["last_snapshot"])
    complete = min(status["snapshots_24h"] / 144 * 100, 100)
    cards = [
        _card(
            "Latest data", f"{data_age} min ago",
            f"Snapshot of {_local(status['last_snapshot']):%a %H:%M}; expected every 10 min",
            _state(data_age, 20, 60, higher_is_better=False),
        ),
        _card(
            "Collection, last 24 h", f"{complete:.0f}%",
            f"{int(status['snapshots_24h'])} snapshots (144 expected)",
            _state(complete, 95, 80),
        ),
        _card(
            "Stations reporting", f"{int(status['stations_reporting'])} / {int(status['stations_total'])}",
            "Stations in the latest snapshot",
            _state(status["stations_reporting"] / max(status["stations_total"], 1) * 100, 95, 80),
        ),
    ]
    if pd.isna(status["last_weather_fetch"]):
        cards.append(_card("Weather", "never", "Open-Meteo, expected hourly", "bad"))
    else:
        weather_age = _minutes_ago(status["last_weather_fetch"])
        cards.append(
            _card("Weather", f"{weather_age} min ago", "Open-Meteo, expected hourly",
                  _state(weather_age, 90, 180, higher_is_better=False))
        )
    if pd.isna(status["last_model_mae"]):
        cards.append(_card("Model error vs baseline", "not trained", "Needs about 2 hours of data", "warn"))
    else:
        gain = (1 - status["last_model_mae"] / status["last_baseline_mae"]) * 100
        run_age = _minutes_ago(status["last_model_run"])
        state = "bad" if run_age > 120 else ("ok" if gain >= -2 else "warn")
        cards.append(
            _card(
                "Model error vs baseline", f"{gain:+.1f}%",
                f"Held-out error {status['last_model_mae']:.2f} vs {status['last_baseline_mae']:.2f} "
                f"bikes (negative = worse); trained {run_age} min ago",
                state,
            )
        )
    overdue = int(status["overdue_forecasts"])
    cards.append(
        _card("Forecasts awaiting a score", f"{overdue}",
              "Due more than 15 min ago, no actual yet", _state(overdue, 0, 261, higher_is_better=False))
    )
    return CARD_CSS + '<div class="mon-grid">' + "".join(cards) + "</div>"


def completeness_chart() -> go.Figure | None:
    """Shows, day by day, what share of the expected snapshots was collected.

    Returns:
        A labelled bar chart of the last 7 days, or None without data.
    """
    daily = queries.daily_completeness()
    if daily.empty:
        return None
    today = daily["day"].iloc[-1]
    labels = [f"{pd.Timestamp(d):%a %d %b}" + (" (today)" if d == today else "") for d in daily["day"]]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=daily["percent"],
            text=[f"{p:.0f}%" for p in daily["percent"]],
            textposition="outside",
            customdata=daily[["snapshots", "expected"]],
            hovertemplate="%{x}<br>%{customdata[0]} snapshots (%{customdata[1]} expected)<extra></extra>",
            marker_color="#2a78d6",
            width=0.5,
        )
    )
    figure.update_layout(yaxis={"range": [0, 115], "title": "Collected (% of expected)", "ticksuffix": "%"})
    figure = _style(figure, 320)
    figure.update_layout(hovermode="closest")
    return figure


def stations_chart() -> go.Figure | None:
    """Plots how many stations reported in each snapshot over 48 hours.

    Returns:
        A line chart with the network size as reference, or None.
    """
    counts = queries.stations_per_snapshot()
    if counts.empty:
        return None
    total = int(queries.pipeline_status().iloc[0]["stations_total"])
    figure = go.Figure()
    figure.add_scatter(
        x=counts["time"], y=[total] * len(counts), mode="lines", name=f"All stations ({total})",
        line={"color": "#8a8984", "dash": "dot", "width": 2}, hoverinfo="skip",
    )
    figure.add_scatter(
        x=counts["time"], y=counts["stations"], mode="lines", name="Stations reporting",
        line={"color": "#2a78d6", "width": 2},
    )
    figure.update_layout(yaxis={"range": [0, total * 1.1], "title": "Stations"})
    return _style(figure, 300)


# ------------------------------------------------------------------ Layout


def refresh_all() -> tuple:
    """Recomputes every panel except the station explorer.

    Returns:
        The values for the outputs listed in build_app, in order.
    """
    frame = queries.overview()
    hourly = queries.accuracy()
    headline, moves, risks = action_tables(frame)
    return (
        live_kpis(frame),
        station_map(frame),
        empty_full_chart(),
        headline,
        moves,
        risks,
        accuracy_headline(hourly),
        accuracy_chart(hourly),
        area_error_chart(),
        training_chart(),
        fill_heatmap(),
        activity_chart(),
        monitoring_cards(),
        completeness_chart(),
        stations_chart(),
        daily_chart(),
    )


def build_app() -> gr.Blocks:
    """Builds the dashboard layout and wires its refresh logic.

    Returns:
        The Gradio app, ready to launch.
    """
    names = sorted(features.load_stations()["name"].tolist())
    with gr.Blocks(title="Mobi Rebalancing Dashboard") as demo:
        gr.Markdown(
            "# Mobi bike-share rebalancing dashboard\n"
            "Where will bikes run out in the next hour, what should be moved, and "
            "can the forecasts be trusted? Refreshes every minute."
        )
        if IS_SIMULATION:
            gr.Markdown("**SIMULATION MODE:** showing synthetic data from `simulation/`, not the live network.")
        timer = gr.Timer(REFRESH_SECONDS)

        with gr.Tab("Live Overview"):
            kpis = gr.Markdown()
            map_plot = gr.Plot(label="Stations by fill level")
            empty_full_plot = gr.Plot(label="Empty and full stations, last 24 h")

        with gr.Tab("Next-Hour Actions"):
            actions_headline = gr.Markdown()
            moves_table = gr.Dataframe(label="Suggested moves", interactive=False)
            risk_table = gr.Dataframe(label="Stations at risk in the next hour", interactive=False)

        with gr.Tab("Forecast Accuracy"):
            gr.Markdown(
                "Three forecasts are logged for every station each 10 minutes: the "
                "**baseline** (bikes stay as they are), the **model**, and the "
                "**corrected** forecast, which uses the model only at stations where it "
                "beat the baseline over the last 24 hours."
            )
            acc_headline = gr.Markdown()
            acc_plot = gr.Plot(label="Error per hour, last 7 days")
            area_plot = gr.Plot(label="Error by area, last 24 h")
            training_plot = gr.Plot(label="Training runs: held-out error")
            daily_plot = gr.Plot(label="Error per day, since collection began")

        with gr.Tab("Station Explorer"):
            picker = gr.Dropdown(choices=names, value=names[0] if names else None, label="Station")
            station_plot = gr.Plot(label="Bikes and forecasts, last 24 h")

        with gr.Tab("Demand Patterns"):
            heatmap_plot = gr.Plot(label="Average fill level by area and hour, last 7 days")
            activity_plot = gr.Plot(label="Network activity by hour: dry vs rain, last 7 days")

        with gr.Tab("Monitoring"):
            gr.Markdown(
                "Is the pipeline healthy? Green tiles need no action; amber and red "
                "ones say what to check. Times are Vancouver time."
            )
            status_board = gr.HTML()
            completeness_plot = gr.Plot(label="Data collected per day (100% = a snapshot every 10 minutes)")
            stations_plot = gr.Plot(label="Stations reporting per snapshot, last 48 h")

        outputs = [
            kpis, map_plot, empty_full_plot,
            actions_headline, moves_table, risk_table,
            acc_headline, acc_plot, area_plot, training_plot,
            heatmap_plot, activity_plot,
            status_board, completeness_plot, stations_plot,
            daily_plot,
        ]
        demo.load(refresh_all, outputs=outputs)
        timer.tick(refresh_all, outputs=outputs)
        demo.load(station_chart, inputs=picker, outputs=station_plot)
        picker.change(station_chart, inputs=picker, outputs=station_plot)
    return demo


def main() -> None:
    """Starts the web server; hosts such as Render set the PORT variable."""
    if IS_SIMULATION and not config.SIMULATION_DB.exists():
        raise SystemExit(
            "No simulation data yet. Run:\n"
            "  uv run python simulation/generate.py\n"
            "  uv run python simulation/replay.py\n"
            "or set DATABASE_URL to use live data."
        )
    build_app().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
