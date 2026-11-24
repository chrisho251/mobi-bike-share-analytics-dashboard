# Mobi bike-share rebalancing dashboard

Next-hour bike availability forecasts and rebalancing suggestions for
[Mobi by Rogers](https://www.mobibikes.ca/), Vancouver's public bike-share
system (261 stations).

Every 10 minutes the platform records the bikes and free docks at every
station, together with the weather, forecasts how many bikes each station will
hold in 60 minutes, and checks each forecast against what actually happened
an hour later. A live dashboard turns this into a short list of bike moves for
the rebalancing dispatcher and tracks forecast accuracy and pipeline health
for the data team.

![Architecture](docs/architecture.png)

## Features

- **Live collection** of the Mobi GBFS feed every 10 minutes and Open-Meteo
  weather every hour into Postgres, with idempotent upserts.
- **Next-hour forecasts** for every station: a persistence baseline, a
  gradient-boosted model, and a corrected forecast that uses the model only
  at stations where it recently beat the baseline.
- **Self-scoring**: every forecast is compared with the observed count one
  hour later; accuracy is tracked per hour, per area and per training run.
- **Rebalancing suggestions**: stations about to run empty or full, and which
  nearby stations to take bikes from.
- **Dashboard** with 12 panels in 6 tabs: Live Overview, Next-Hour Actions,
  Forecast Accuracy, Station Explorer, Demand Patterns and Pipeline Health.
- **Simulation** of weeks of network history to develop and validate the
  forecasting code offline.

## Quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/chrisho251/mobi-bike-share-analytics-dashboard.git
cd mobi-bike-share-analytics-dashboard
uv sync --all-extras
```

### Option A: explore with simulated data (no account needed)

```bash
uv run python simulation/generate.py --days 21          # synthetic history
uv run python simulation/replay.py --retrain-hours 24   # forecasts + scores
DATABASE_URL=sqlite:///simulation/data/mobi_sim.db uv run python -m mobi.app
```

Open http://localhost:7860. A banner marks the dashboard as simulation mode.
On Windows PowerShell, set the variable first:
`$env:DATABASE_URL = "sqlite:///simulation/data/mobi_sim.db"`.

### Option B: collect live data

Without `DATABASE_URL`, data goes to a local SQLite file, `data/mobi.db`.

```bash
uv run python -m mobi.pipeline   # one cycle: ingest, score, train, forecast
uv run python -m mobi.app        # dashboard on http://localhost:7860
```

Run the pipeline repeatedly (every 10 minutes) to build history. Forecasts
are logged from the first run; the model starts once 2,000 labelled
snapshots exist (about two hours) and the first forecasts are scored one hour
after they are made.

To use Postgres, put the connection string in `.env.local` (or the
environment):

```bash
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

## Deployment

| Part | Service | Setup |
|---|---|---|
| Database | [Neon](https://neon.tech) Postgres | Create a project; copy the pooled connection string. |
| Pipeline | GitHub Actions (`.github/workflows/pipeline.yml`) | Add the repository secret `DATABASE_URL`. |
| Schedule | [cron-job.org](https://cron-job.org) | Every 10 minutes, `POST https://api.github.com/repos/<owner>/<repo>/actions/workflows/pipeline.yml/dispatches` with body `{"ref":"main"}` and headers `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`. Use a fine-grained token limited to this repository with *Actions: read and write* only. |
| Dashboard | [Render](https://render.com) web service | Build: `pip install ".[app]"`. Start: `python -m mobi.app`. Environment: `DATABASE_URL` (a read-only database role is recommended). |

The workflow also has a GitHub cron schedule as a fallback, but GitHub does
not guarantee scheduled runs and may delay them by hours on new repositories.

## How the forecast works

The model predicts the **change** in bikes over the next hour from the
current count, the counts 30 and 60 minutes earlier, the local hour and
weekday, the weather now and in one hour, and the station's location and
area. It uses absolute-error loss, so it predicts "no change" wherever there
is no reliable signal, and early stopping, so its size adapts to the data
available. It is retrained on every pipeline run.

The **corrected** forecast is chosen per station from the last 24 hours of
scored forecasts: the model where it beat the baseline, the baseline
elsewhere. See [docs/architecture.md](docs/architecture.md) for the
alternatives that were evaluated and rejected.

### Results

Mean absolute error in bikes, forecasting 60 minutes ahead (lower is better).

**Simulated network**, 21 days, replayed as if live (`simulation/replay.py`,
model retrained daily). The gain over persistence grows as history
accumulates:

| Week | Baseline | Model | Corrected | Corrected vs baseline |
|---|---|---|---|---|
| 1 | 1.120 | 1.082 | 1.081 | -3.5 % |
| 2 | 1.142 | 1.051 | 1.063 | -6.9 % |
| 3 | 1.149 | 1.041 | 1.050 | -8.6 % |

On the one simulated day where the model was worse than persistence, the
correction limited the loss (1.083 vs 1.108 for the model alone).

**Live network**, first two days: changes within an hour are small (58 % of
observations show no change, mean error of persistence 0.42 bikes), and the
model has not yet seen a full week of patterns. Early models were worse than
persistence (0.54); the correction kept the logged forecast within about 1 %
of persistence (0.43). The Forecast Accuracy tab tracks this as data
accumulates.

## Project structure

```
src/mobi/
  config.py      endpoints, constants, database URL
  db.py          schema (5 tables), engine, upsert
  ingest.py      GBFS and Open-Meteo collection
  features.py    feature building (lags, targets, calendar, weather)
  forecast.py    training, forecasting, correction, scoring
  pipeline.py    one collection and forecasting cycle (entry point)
  actions.py     rebalancing rules
  queries.py     read-only, cached dashboard queries
  app.py         Gradio dashboard
simulation/      synthetic history generator and forecast replay
tests/           pytest suite (SQLite, no network)
docs/            architecture and tech stack
```

## Development

```bash
uv run pytest           # tests run on temporary SQLite databases
```

## Data sources and attribution

- Station data: Mobi by Rogers [GBFS feed](https://gbfs.kappa.fifteen.eu/gbfs/2.3/mobi/en/gbfs.json).
- Weather data: [Open-Meteo](https://open-meteo.com/), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Map tiles: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.
