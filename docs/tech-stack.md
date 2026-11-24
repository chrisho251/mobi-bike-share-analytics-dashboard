# Tech stack

Every component runs on a free tier.

| Layer | Technology | Why |
|---|---|---|
| Language and packaging | Python 3.12, [uv](https://docs.astral.sh/uv/) | Reproducible installs from `uv.lock`; optional extras keep each environment lean (`ml` for training, `app` for the dashboard). |
| Data access | requests, pandas | Feed download with retries; time-series joins (`merge_asof`) for lags and targets. |
| Database layer | SQLAlchemy 2 Core, psycopg 3 | One schema for Postgres and SQLite; dialect-specific upserts (`ON CONFLICT`). |
| Database | Neon serverless Postgres 18 | Managed Postgres that scales to zero between runs. |
| Machine learning | scikit-learn `HistGradientBoostingRegressor` | Fast on hundreds of thousands of rows, handles missing lags natively, absolute-error loss. |
| Scheduling | GitHub Actions + cron-job.org | Actions runs the pipeline on a clean runner; cron-job.org triggers it reliably every 10 minutes. |
| Dashboard | Gradio 6, Plotly | Python-only web UI with timers and tabs; interactive charts and an OpenStreetMap station map. |
| Hosting | Render (web service) | Public URL for the dashboard. |
| Testing | pytest | Unit and end-to-end tests on throwaway SQLite databases. |
| Data sources | Mobi GBFS 2.3, Open-Meteo | Public, no key required. Open-Meteo data is CC BY 4.0. |

## Dependency groups

| Install | Contents | Used by |
|---|---|---|
| `uv sync` | pandas, requests, SQLAlchemy, psycopg, python-dotenv | ingestion only |
| `uv sync --extra ml` | + scikit-learn | pipeline with forecasting (GitHub Actions) |
| `uv sync --extra app` | + Gradio, Plotly | dashboard |
| `uv sync --all-extras` | everything, plus pytest (dev group) | development |
