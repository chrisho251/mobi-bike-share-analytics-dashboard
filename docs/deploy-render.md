# Deploy the dashboard on Render

The dashboard runs as a free Render web service and reads the live database
(Neon Postgres). It never writes, so it can use a read-only database role.
Count on about 20 minutes.

## Before you start

- The repository is on GitHub and `main` is up to date.
- The pipeline is already collecting data into Neon (see the main README).
- You have a [Render](https://render.com) account. Sign up with GitHub; the
  free web service should not ask for a card. If Render asks for payment
  details, stop and choose another host.

## 1. Create a read-only database role (recommended)

The dashboard needs `SELECT` only. In the Neon console open **SQL Editor**,
pick the production branch and database, and run the following with your own
password (at least 20 random characters):

```sql
CREATE ROLE dashboard_ro LOGIN PASSWORD 'replace-with-a-long-random-password';
GRANT USAGE ON SCHEMA public TO dashboard_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro;
```

Build the connection string from the one the pipeline uses: keep the host,
database name and `?sslmode=require`, and replace the user and password.

```
postgresql://dashboard_ro:<password>@<same-host>/<same-database>?sslmode=require
```

Use the **pooled** host (the one with `-pooler` in its name). Check the role
works before going on:

```bash
DATABASE_URL="<the string above>" python -c "from mobi import queries; print(queries.pipeline_status())"
```

Writing must fail with it: `INSERT` into any table should answer
`permission denied`.

## 2. Create the service from the Blueprint

1. In Render choose **New → Blueprint** and connect the GitHub repository.
2. Render reads [render.yaml](../render.yaml) and proposes one web service,
   `mobi-dashboard`, on the **Free** plan.
3. When asked for `DATABASE_URL`, paste the string from step 1.
4. Apply. The first build installs the dependencies and takes 3 to 5 minutes.

Without the Blueprint, create a **Web Service** by hand with: runtime Python,
build command `pip install -r requirements-dashboard.txt`, start command `python -m mobi.app`,
instance type Free, and environment variables `PYTHON_VERSION=3.12` and
`DATABASE_URL`.

## 3. Check it

Open the `https://mobi-dashboard-xxxx.onrender.com` URL Render shows. The
Live Overview map should load with the latest snapshot, and there must be no
**SIMULATION MODE** banner. Open every tab once.

If the page shows the simulation banner or "No simulation data yet", the
`DATABASE_URL` variable is missing or empty: add it under **Environment** and
redeploy.

## Free plan behaviour

- The service sleeps after about 15 minutes without visitors and takes up to a
  minute to wake. Open the URL a few minutes before a demo.
- The database also scales to zero; the first query after a quiet period is
  slower.
- Render redeploys automatically on every push to `main`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Build fails on Python or wheels | Confirm `PYTHON_VERSION` is `3.12`. |
| Out of memory, service restarts | Free instances have 512 MB. In the dashboard code, lower the history window of the Demand Patterns queries (`days` in `queries.py`). |
| `permission denied for table` | Re-run the `GRANT` statements of step 1 on the right database. |
| Charts empty after a restart | The first load queries Neon after it woke up; reload after a few seconds. |
| Site works but data is old | The pipeline stopped, not the dashboard: check the Monitoring tab and the GitHub Actions runs. |
