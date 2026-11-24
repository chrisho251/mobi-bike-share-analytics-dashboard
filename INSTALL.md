# Installation

From a fresh clone to a running dashboard in about 10 minutes. The steps use
simulated data first, so no database or account is needed; connecting live
data is the last section.

Tested with Python 3.12 on Windows 11. The commands for macOS and Linux are
given where they differ.

## 1. Requirements

- Git
- Python 3.12, or [uv](https://docs.astral.sh/uv/), which downloads Python
  3.12 for you
- Internet access during installation and for live data

## 2. Get the code

```bash
git clone https://github.com/chrisho251/mobi-bike-share-analytics-dashboard.git
cd mobi-bike-share-analytics-dashboard
```

All later commands run from this folder.

## 3. Install the dependencies

Choose one of the two options.

### Option A: uv (recommended)

Install uv once:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then install the project with every optional part
(forecasting, dashboard, tests) from the lock file:

```bash
uv sync --all-extras
```

With uv, prefix the commands in the rest of this guide with `uv run`, for
example `uv run python -m mobi.app`.

### Option B: pip and a virtual environment

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, allow scripts for this terminal
only with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then
activate again.

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins every package to the versions in `uv.lock` and installs
the project itself (`-e .`). To run the tests with pip, also install pytest:
`python -m pip install pytest`.

## 4. Check the installation

```bash
python -m pytest
```

All tests should pass in under a minute. They use temporary databases and do
not need network access.

## 5. Run on simulated data

Without a `DATABASE_URL`, every component uses the simulation database in
`simulation/data/`. Make sure the variable is not set in your terminal
(PowerShell: `Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue`;
bash: `unset DATABASE_URL`).

```bash
python simulation/generate.py    # 21 days of synthetic history, ~20 seconds
python simulation/replay.py      # forecasts and scores, ~6 minutes
python -m mobi.app               # dashboard
```

Open http://localhost:7860. A **SIMULATION MODE** banner confirms the data
source; stop the server with `Ctrl+C`. For a faster first look, generate fewer
days (`python simulation/generate.py --days 3`, replay in about a minute);
with that little history the model is only level with the baseline.
[docs/run-with-simulation.md](docs/run-with-simulation.md) walks through each
dashboard tab.

## 6. Connect live data

Create a file named `.env.local` in the repository root (it is git-ignored)
with a Postgres connection string:

```bash
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

For a local database without Postgres, use
`DATABASE_URL=sqlite:///data/mobi.db` instead.

Then collect one cycle of live data and start the dashboard:

```bash
python -m mobi.pipeline   # ingest stations, bikes and weather; score; forecast
python -m mobi.app
```

The first run logs stations, a snapshot and baseline forecasts. Run the
pipeline every 10 minutes to build history: the model starts after about two
hours of snapshots, and each forecast is scored one hour after it is made.
[README.md](README.md#deployment) describes the scheduled setup.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No simulation data yet` when starting the dashboard | Run step 5 (`generate.py`, then `replay.py`). |
| `ModuleNotFoundError: No module named 'mobi'` | The virtual environment is not active, or dependencies were installed elsewhere. Activate `.venv` (option B) or use `uv run` (option A). |
| Dashboard shows live data instead of the simulation | `DATABASE_URL` is set in the terminal or in `.env.local`; see step 5. |
| `python -m mobi.pipeline` stops with "points at the simulation database" | Intended: live data is never written into the simulation. Set `DATABASE_URL` first (step 6). |
| `weather failed` in the pipeline log | Open-Meteo was briefly unreachable. Weather is optional; the next run retries. |
| Port 7860 already in use | Stop the other server, or set another port: `$env:PORT = "7861"` (PowerShell) or `PORT=7861 python -m mobi.app` (bash). |
| `py -3.12` not found (Windows) | Install Python 3.12 from python.org, or use option A. |
