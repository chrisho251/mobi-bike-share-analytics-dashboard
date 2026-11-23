"""Gradio dashboard for Mobi dispatchers.

Four tabs answer four questions:

* Live Overview     - what is the network doing right now?
* Next-Hour Actions - where will bikes run out or overflow, and what to move?
* Forecast Accuracy - can the forecasts be trusted, and are they improving?
* Station Explorer  - what happened at one station?

Run with: uv run --extra app python -m mobi.app
"""

import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from mobi import actions, config, features, queries
from mobi.ingest import utcnow

REFRESH_SECONDS = 60
MAX_MOVES_SHOWN = 25
EMPTY_MESSAGE = "No data yet. The collector has not stored any snapshot."
NO_FORECAST_MESSAGE = (
    "No forecasts yet. They appear after the first model run, then fill in "
    "with real outcomes one hour later."
)


def _kpi_text(frame: pd.DataFrame) -> str:
    """Summarises the network state as Markdown.

    Args:
        frame: The output of queries.overview().

    Returns:
        A short Markdown block with data age and empty/full station counts.
    """
    if frame.empty:
        return EMPTY_MESSAGE
    latest = frame["snapshot_time"].iloc[0]
    age_min = int((utcnow() - latest).total_seconds() // 60)
    local = pd.Timestamp(latest).tz_localize("UTC").tz_convert(queries.LOCAL_TZ)
    empty = int((frame["num_bikes"] == 0).sum())
    full = int((frame["num_docks"] == 0).sum())
    return (
        f"**Data from {local:%H:%M} Vancouver time ({age_min} min ago)** "
        f"| {len(frame)} stations | **{int(frame['num_bikes'].sum())}** bikes available "
        f"| **{empty}** empty stations | **{full}** full stations"
    )


def _map_figure(frame: pd.DataFrame) -> go.Figure:
    """Draws every station on a map, coloured by how full it is.

    Args:
        frame: The output of queries.overview().

    Returns:
        A Plotly map figure (empty if there is no data).
    """
    if frame.empty:
        return go.Figure()
    frame = frame.copy()
    frame["fill_percent"] = (frame["fill_ratio"] * 100).round(0)
    figure = px.scatter_map(
        frame,
        lat="lat",
        lon="lon",
        color="fill_percent",
        range_color=(0, 100),
        color_continuous_scale="Viridis",
        hover_name="name",
        hover_data={"num_bikes": True, "num_docks": True, "lat": False, "lon": False},
        center={"lat": config.VANCOUVER_LAT, "lon": config.VANCOUVER_LON},
        zoom=11.6,
        height=560,
        map_style="open-street-map",
        labels={"fill_percent": "Full (%)"},
    )
    figure.update_traces(marker={"size": 11})
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return figure


def refresh_overview() -> tuple[str, go.Figure]:
    """Loads the Live Overview tab.

    Returns:
        A tuple (KPI Markdown, map figure).
    """
    frame = queries.overview()
    return _kpi_text(frame), _map_figure(frame)


def refresh_actions() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Loads the Next-Hour Actions tab.

    Returns:
        A tuple (headline Markdown, suggested moves, stations at risk).
    """
    frame = queries.overview()
    risks = actions.at_risk_stations(frame)
    if risks.empty and (frame.empty or frame["pred_corrected"].isna().all()):
        return NO_FORECAST_MESSAGE, pd.DataFrame(), pd.DataFrame()
    moves = actions.plan_moves(frame)
    empty_soon = int((risks["risk"] == "Empty soon").sum()) if not risks.empty else 0
    full_soon = int((risks["risk"] == "Full soon").sum()) if not risks.empty else 0
    target = pd.Timestamp(frame["target_time"].dropna().iloc[0])
    target_local = target.tz_localize("UTC").tz_convert(queries.LOCAL_TZ)
    headline = (
        f"**Forecast for {target_local:%H:%M}** | **{empty_soon}** stations will run "
        f"low, **{full_soon}** will be nearly full | **{len(moves)}** moves "
        f"suggested, showing the {min(len(moves), MAX_MOVES_SHOWN)} most urgent"
    )
    return headline, moves.head(MAX_MOVES_SHOWN), risks


def refresh_accuracy() -> tuple[str, go.Figure | None, pd.DataFrame]:
    """Loads the Forecast Accuracy tab.

    Returns:
        A tuple (headline Markdown, error-over-time figure or None when there
        is nothing to plot yet, model run table).
    """
    hourly = queries.accuracy()
    runs = queries.model_history()
    if hourly.empty:
        return NO_FORECAST_MESSAGE, None, runs
    totals = hourly[["baseline", "model", "corrected"]].multiply(hourly["n"], axis=0)
    mae = totals.sum() / hourly["n"].sum()
    gain = (1 - mae["corrected"] / mae["baseline"]) * 100
    headline = (
        f"**Overall error (bikes, lower is better)** | no-change baseline "
        f"**{mae['baseline']:.2f}** | model **{mae['model']:.2f}** | "
        f"corrected **{mae['corrected']:.2f}** | corrected is **{gain:.0f}%** "
        f"better than the baseline | {int(hourly['n'].sum())} forecasts scored"
    )
    long = hourly.melt(
        id_vars=["hour", "n"],
        value_vars=["baseline", "model", "corrected"],
        var_name="forecast",
        value_name="error",
    )
    figure = px.line(
        long,
        x="hour",
        y="error",
        color="forecast",
        markers=True,
        labels={"hour": "Hour (Vancouver)", "error": "Mean absolute error (bikes)"},
        color_discrete_map={
            "baseline": "#9aa0a6",
            "model": "#1f77b4",
            "corrected": "#d95f02",
        },
    )
    figure.update_layout(height=420, legend_title_text="")
    return headline, figure, runs


def station_figure(station_name: str) -> go.Figure:
    """Draws one station's recent bikes together with its forecasts.

    Args:
        station_name: The station name picked in the dropdown.

    Returns:
        A Plotly figure with actual counts and forecast markers.
    """
    figure = go.Figure()
    info = features.load_stations()
    match = info[info["name"] == station_name]
    if match.empty:
        return figure
    station_id = match["station_id"].iloc[0]
    history, forecasts = queries.station_history(station_id)
    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["snapshot_time"],
                y=history["num_bikes"],
                mode="lines",
                name="Actual bikes",
                line={"color": "#1f77b4"},
            )
        )
    if not forecasts.empty:
        figure.add_trace(
            go.Scatter(
                x=forecasts["target_time"],
                y=forecasts["pred_corrected"],
                mode="markers",
                name="Forecast (corrected)",
                marker={"color": "#d95f02", "size": 6},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecasts["target_time"],
                y=forecasts["pred_baseline"],
                mode="markers",
                name="No-change baseline",
                marker={"color": "#9aa0a6", "size": 5, "symbol": "x"},
            )
        )
    capacity = match["capacity"].iloc[0]
    if pd.notna(capacity):
        figure.add_hline(y=capacity, line_dash="dot", annotation_text="capacity")
    figure.update_layout(
        height=440,
        xaxis_title="Time (Vancouver)",
        yaxis_title="Bikes at station",
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def build_app() -> gr.Blocks:
    """Builds the dashboard layout and connects its refresh logic.

    Returns:
        The Gradio Blocks app, ready to launch.
    """
    names = sorted(features.load_stations()["name"].tolist())
    with gr.Blocks(title="Mobi Rebalancing Dashboard") as demo:
        gr.Markdown(
            "# Mobi bike-share rebalancing dashboard\n"
            "For dispatchers: see the network now, know where bikes will run out "
            "in the next hour, and decide what to move. Refreshes every minute."
        )
        timer = gr.Timer(REFRESH_SECONDS)

        with gr.Tab("Live Overview"):
            overview_kpi = gr.Markdown()
            overview_map = gr.Plot()

        with gr.Tab("Next-Hour Actions"):
            actions_headline = gr.Markdown()
            gr.Markdown("### Suggested moves")
            moves_table = gr.Dataframe(interactive=False)
            gr.Markdown("### Stations at risk in the next hour")
            risk_table = gr.Dataframe(interactive=False)

        with gr.Tab("Forecast Accuracy"):
            accuracy_headline = gr.Markdown()
            accuracy_plot = gr.Plot()
            gr.Markdown("### Model training runs")
            runs_table = gr.Dataframe(interactive=False)

        with gr.Tab("Station Explorer"):
            station_picker = gr.Dropdown(
                choices=names, value=names[0] if names else None, label="Station"
            )
            station_plot = gr.Plot()

        def refresh_all():
            """Refreshes every tab at once (used on load and on each timer tick)."""
            kpi, overview_figure = refresh_overview()
            headline, moves, risks = refresh_actions()
            acc_headline, acc_figure, runs = refresh_accuracy()
            return (
                kpi, overview_figure, headline, moves, risks,
                acc_headline, acc_figure, runs,
            )

        outputs = [
            overview_kpi, overview_map, actions_headline, moves_table, risk_table,
            accuracy_headline, accuracy_plot, runs_table,
        ]
        demo.load(refresh_all, outputs=outputs)
        demo.load(station_figure, inputs=station_picker, outputs=station_plot)
        timer.tick(refresh_all, outputs=outputs)
        station_picker.change(station_figure, inputs=station_picker, outputs=station_plot)
    return demo


def main() -> None:
    """Starts the web server (the PORT variable is set by the host, e.g. Render)."""
    build_app().launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860"))
    )


if __name__ == "__main__":
    main()
