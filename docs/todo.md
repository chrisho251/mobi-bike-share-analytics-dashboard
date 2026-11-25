# Work in progress

What is left to do, and findings worth keeping. Last updated 24 November
2026.

## To do

### Housekeeping

- [x] **Run the tests in CI.** `.github/workflows/tests.yml` runs `pytest` on
  every pull request and push to main.
- [x] **Make `uv run pytest` work without extras.** The `dev` group now
  includes the `ml` extra, which the tests need.
- [ ] **Remove leftover files.** `neon.ts`, `package.json` and
  `package-lock.json` (Neon CLI config the Python code does not use), and
  `src/test/check_upsert.py` and `src/scripts/check_db.py`, which overlap with
  `tests/` (the second prints "servier:").

### Follow-ups to recent changes

- [x] **Daily accuracy summary, before 6 December.** `accuracy_daily` keeps
  one row per day (forecasts scored, error of each forecast, alert outcomes)
  after retention deletes the forecasts. Shown on the Forecast Accuracy tab.
- [ ] **Database size on the Monitoring tab.** The free-tier estimate behind
  the retention periods is rough. A tile with the real size against Neon's
  0.5 GB would confirm that retention works.
- [ ] **Real van capacity.** `actions.VAN_CAPACITY` is set to 20 as a guess.
  Set it to what Mobi's vans carry. The "Mobi's own van moves" statistic below
  can estimate it.
- [ ] **E-bike statistics, once there is history.** E-bike counts are now
  stored with each snapshot. After a week or two: e-bike availability over the
  day, the hours each station has no e-bike, and whether e-bikes run out faster
  than classic bikes.

### Needs time or a decision

- [ ] **Check whether the live model is useful (mid-December).** With 3–4 weeks
  of live data, compare live accuracy and update the README's Results table. If
  the model only matches the baseline, the features need work.
- [ ] **Confirm the dashboard deployment.** It is not known whether the Render
  dashboard is live. If not, finish the README's Deployment table.
- [ ] **Better trips, if dispatchers use them.** Trips that collect from more
  than one station, and drop-offs ordered as a route.

### More statistics

- [ ] **Where bikes flow from and to.** The change in bikes between snapshots
  shows which stations lose and which gain bikes, by hour. Morning and evening
  maps would show the commute directly. It undercounts trips (a pickup and a
  return in the same 10 minutes cancel out), so label it a lower bound.
- [ ] **Mobi's own van moves.** A jump of about 8 or more bikes within 10
  minutes is almost certainly a van, not riders. Counting these shows where and
  when Mobi already rebalances, and roughly how much a van carries.
- [ ] **Weather in more detail.** Activity against temperature and rain amount,
  and weekdays against weekends, instead of only dry against rainy hours.
- [ ] **Where the model wins and loses.** Error by time of day, by how full the
  station was and by station size, plus how many stations the correction
  currently trusts the model at.
- [ ] **Look into `free_bike_status`.** It lists about 2,160 bikes and may give
  every bike's position, which would mean fleet size and bikes in use. Check
  what it contains first.

## Findings

- **`last_reported` is not usable.** Every station's `last_reported` equals
  the feed's own timestamp, so the provider fills in the current time. It
  cannot reveal a station that stopped reporting and is not stored. A station
  that drops out of the feed is already visible on the Monitoring tab.
- **Vancouver no longer changes its clocks.** The time zone database
  (tzdata 2026.6) has British Columbia on UTC-7 all year since March 2026.
  The code uses that database, so local times are right, but anything written
  assuming a winter offset of UTC-8 is not.
- **About 16 % of docked bikes are e-bikes** (311 of 1,893 on 24 November).
  The simulation uses the same share.
