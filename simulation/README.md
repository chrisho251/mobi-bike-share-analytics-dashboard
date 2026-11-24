# Simulation

Synthetic history of the Mobi network for developing and testing the forecast
code. It is **not real data** and is never written to the production database.

- `stations.csv`: real station locations and capacities (snapshot of the GBFS
  `station_information` feed), used as the base of the simulation.
- `generate.py`: simulates 10-minute bike counts and hourly weather into
  `simulation/data/mobi_sim.db` (git-ignored).

```bash
uv run python simulation/generate.py --days 21
```

## Model of the network

- Trips per station follow a weekday commute curve (peaks around 08:30 and
  17:30) or a weekend midday curve, and drop when it rains.
- Each area has a direction profile: residential areas empty in the morning and
  refill in the evening, Downtown does the opposite, Stanley Park follows
  leisure rides.
- Arrivals and departures are Poisson draws, so short-term noise is realistic.
- Rebalancing vans occasionally reset empty or full stations to half capacity
  during the day, a source of changes no forecast can anticipate.
