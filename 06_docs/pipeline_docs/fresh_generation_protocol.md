# Fresh Dataset Generation Protocol

- Global seed: `5491`
- Input: `02_processed_data/classic/base/*.base.json`
- Output: `02_processed_data/fresh/fresh/*.fresh.json` and `02_processed_data/fresh/fresh_meta/*.fresh.meta.json`

## Rules

- service time: `service_time_min = 5 + ceil(0.2 * demand)`
- depot start time: `08:00` (480 minutes)
- time windows by distance band (to depot):
  - near: `[480+j, 720+j]`
  - mid: `[540+j, 840+j]`
  - far: `[600+j, 960+j]`
  - jitter `j` sampled from `[-20, 20]` with deterministic instance seed
- freshness class by demand quantiles (30/70): `low`, `medium`, `high`
- max travel time by freshness: low=300, medium=180, high=90
- temp zone by class-conditional probability:
  - high: frozen 0.35, chilled 0.50, ambient 0.15
  - medium: frozen 0.20, chilled 0.50, ambient 0.30
  - low: frozen 0.10, chilled 0.25, ambient 0.65
- penalties:
  - `late_penalty_per_min = 0.8 * demand`
  - `spoilage_penalty = 2.0 * demand * freshness_weight`
  - freshness_weight: high=1.5, medium=1.0, low=0.7

## Objective Weights

- distance: 1.0
- lateness: 1.0
- spoilage: 1.0
