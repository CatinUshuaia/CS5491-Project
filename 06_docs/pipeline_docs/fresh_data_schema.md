# Fresh Data Schema

## File: `02_processed_data/fresh/fresh/<instance_id>.fresh.json`

- Includes all fields from classic `base.json`
- Additional instance-level fields:
  - `problem_variant`
  - `depot_start_time_min`
  - `objective_weights`
- Additional customer-level fields:
  - `distance_to_depot`
  - `distance_band`
  - `service_time_min`
  - `ready_time_min`, `due_time_min`
  - `freshness_class`
  - `max_travel_time_min`
  - `temp_zone`
  - `late_penalty_per_min`
  - `spoilage_penalty`

## File: `02_processed_data/fresh/fresh_meta/<instance_id>.fresh.meta.json`

- Generation rules, seeds, and source mapping for full reproducibility.
