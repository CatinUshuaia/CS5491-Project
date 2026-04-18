# Data Schema (Classic CVRP)

## File: `02_processed_data/classic/base/<instance_id>.base.json`

- `instance_id` (string): instance name, e.g. `A-n32-k5`
- `set_id` (string): one of `A/B/E/F/M/P`
- `type` (string): `CVRP`
- `dimension` (int): total nodes including depot
- `vehicle_capacity` (int): truck capacity
- `vehicle_count_hint` (int|null): parsed from instance name suffix `-kX`
- `depot` (object): `{id, x, y, demand}`
- `customers` (array): list of `{id, x, y, demand}` excluding depot
- `node_id_order` (array[int]): node order used by distance matrix
- `distance_matrix` (array[array[int]]): full symmetric matrix
- `distance_metric` (string): `EUC_2D` or `EXPLICIT`
- `known_opt_cost` (number|null): from `.sol` if available
- `known_opt_routes` (array[array[int]]): parsed route sequence from `.sol`

## File: `02_processed_data/classic/meta/<instance_id>.meta.json`

- `instance_id` (string)
- `source.vrp_file` (string): absolute path to source `.vrp`
- `source.sol_file` (string|null): absolute path to source `.sol` if available
- `parser_version` (string): parser version label
- `raw_comment` (string|null): original `COMMENT` field
- `edge_weight_type` (string)
- `edge_weight_format` (string|null)
- `depot_ids` (array[int])

## File: `02_processed_data/classic/index.csv`

One row per instance for quick filtering and statistics.
