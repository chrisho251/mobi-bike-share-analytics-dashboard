# Run the project on simulated data

This guide runs the full forecasting loop and the dashboard on a simulated
Mobi network: real station locations, synthetic bike counts and weather. No
account, password or database server is needed. It takes about 10 minutes.

## 1. Prerequisites

- **Git**
- **uv**, the Python package manager (it installs Python 3.12 for you):

  Windows (PowerShell):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  macOS / Linux:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  Open a new terminal afterwards and check with `uv --version`.

## 2. Get the code and install

```bash
git clone https://github.com/chrisho251/mobi-bike-share-analytics-dashboard.git
cd mobi-bike-share-analytics-dashboard
uv sync --all-extras
```

Already cloned? Update instead:

```bash
git pull
uv sync --all-extras
```

## 3. Make sure you are in simulation mode

Simulation mode is the default **when `DATABASE_URL` is not set**. Check both
places it can come from:

- No `.env.local` file in the repository root (or one without `DATABASE_URL`).
- No `DATABASE_URL` in your terminal session:
  - PowerShell: `Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue`
  - bash / zsh: `unset DATABASE_URL`

If you do have a `.env.local` for live data and want to keep it, point the
current terminal at the simulation instead (an existing variable wins over
`.env.local`):

- PowerShell: `$env:DATABASE_URL = "sqlite:///simulation/data/mobi_sim.db"`
- bash / zsh: `export DATABASE_URL=sqlite:///simulation/data/mobi_sim.db`

## 4. Generate the simulated history (~20 seconds)

```bash
uv run python simulation/generate.py
```

Expected last line:

```
Wrote 789,525 snapshots rows for 261 stations, ... into ...simulation/data/mobi_sim.db
```

Options: `--days 21` (length of history), `--seed 42` (change for a different
network). Rerunning replaces the previous simulation.

## 5. Replay the forecasts (~6 minutes)

```bash
uv run python simulation/replay.py
```

This forecasts every station every 10 minutes over the whole history, exactly
as the live pipeline would, retrains the model once per simulated day, and
scores each forecast one hour later. It prints the daily error of the three
forecasts (lower is better), for example:

```
            baseline  model  corrected
day
2026-11-04     1.015  0.975      0.989
...
21 training runs, 789,525 forecasts written to ...
```

## 6. Open the dashboard

```bash
uv run python -m mobi.app
```

Open http://localhost:7860. A **SIMULATION MODE** banner confirms the data
source. Stop the server with `Ctrl+C`.

| Tab | What to look at |
|---|---|
| Live Overview | Station map coloured by fill level; empty and full stations over 24 h. |
| Next-Hour Actions | Suggested bike moves and the stations at risk in the next hour. |
| Forecast Accuracy | Error of baseline, model and corrected forecasts per hour and per area; held-out error of each training run. |
| Station Explorer | Pick a station: its bikes over 24 h with the forecasts made an hour earlier. |
| Demand Patterns | Fill level by area and hour; activity in dry vs rainy hours. |
| Monitoring | Status board (green / amber / red tiles), hour-by-hour collection and scoring, stations reporting. |

The simulation stops at the moment it was generated, so "minutes ago" keeps
growing and the Monitoring tab turns the "Latest data" tile red: expected, not an error.
Regenerate (steps 4 and 5) for fresh timestamps.

## 7. Run the tests (optional)

```bash
uv run pytest
```

All tests use temporary databases and never touch the simulation or live data.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No simulation data yet. Run: ...` | Steps 4 and 5 have not been run in this clone. |
| Dashboard has no banner and shows live data, or asks for a password | `DATABASE_URL` is set; see step 3. |
| `python -m mobi.pipeline` exits with "points at the simulation database" | By design: the live collector never writes into simulated data. Use `simulation/replay.py`. |
| Port 7860 already in use | Another dashboard is running; stop it, or start with a different port: `PORT=7861` (bash) or `$env:PORT = "7861"` (PowerShell). |
| `uv: command not found` | Open a new terminal after installing uv. |

## Switching to live data

Create `.env.local` in the repository root with the connection string you
were given (ask for a read-only one), then restart the dashboard:

```bash
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

Never commit `.env.local`; it is git-ignored.
