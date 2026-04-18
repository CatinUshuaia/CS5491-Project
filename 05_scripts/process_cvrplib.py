import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATASET_SETS = ["A", "B", "E", "F", "M", "P"]


@dataclass
class VrpInstance:
    instance_id: str
    set_id: str
    dimension: int
    capacity: int
    edge_weight_type: str
    edge_weight_format: Optional[str]
    depot_ids: List[int]
    coords: Dict[int, Tuple[Optional[float], Optional[float]]]
    demands: Dict[int, int]
    distance_matrix: List[List[int]]
    vehicle_count_hint: Optional[int]
    comment: Optional[str]


@dataclass
class SolData:
    known_opt_cost: Optional[float]
    routes: List[List[int]]


def nint(value: float) -> int:
    return int(math.floor(value + 0.5))


def parse_key_value(line: str) -> Tuple[str, str]:
    if ":" in line:
        key, value = line.split(":", 1)
    else:
        parts = line.split()
        key = parts[0]
        value = " ".join(parts[1:]) if len(parts) > 1 else ""
    return key.strip().upper(), value.strip()


def parse_vrp_file(vrp_path: Path, set_id: str) -> VrpInstance:
    lines = [ln.rstrip() for ln in vrp_path.read_text(encoding="utf-8").splitlines()]

    metadata: Dict[str, str] = {}
    coords: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    demands: Dict[int, int] = {}
    depot_ids: List[int] = []
    edge_weight_values: List[float] = []

    section: Optional[str] = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()

        if upper in {
            "NODE_COORD_SECTION",
            "DEMAND_SECTION",
            "DEPOT_SECTION",
            "EDGE_WEIGHT_SECTION",
            "EOF",
        }:
            section = upper
            if upper == "EOF":
                break
            continue

        if section is None:
            key, value = parse_key_value(line)
            metadata[key] = value
            continue

        if section == "NODE_COORD_SECTION":
            parts = line.split()
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            coords[node_id] = (x, y)
        elif section == "DEMAND_SECTION":
            parts = line.split()
            node_id = int(parts[0])
            demand = int(float(parts[1]))
            demands[node_id] = demand
        elif section == "DEPOT_SECTION":
            if line == "-1":
                continue
            depot_ids.append(int(line))
        elif section == "EDGE_WEIGHT_SECTION":
            edge_weight_values.extend(float(x) for x in line.split())

    dimension = int(metadata["DIMENSION"])
    capacity = int(float(metadata["CAPACITY"]))
    instance_id = metadata.get("NAME", vrp_path.stem)
    edge_weight_type = metadata.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    edge_weight_format = metadata.get("EDGE_WEIGHT_FORMAT")
    if edge_weight_format:
        edge_weight_format = edge_weight_format.upper()

    for node_id in range(1, dimension + 1):
        if node_id not in demands:
            raise ValueError(f"{instance_id}: missing demand for node {node_id}")
        if node_id not in coords:
            coords[node_id] = (None, None)

    if not depot_ids:
        raise ValueError(f"{instance_id}: missing depot section")

    distance_matrix = build_distance_matrix(
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        coords=coords,
        edge_weight_values=edge_weight_values,
        instance_id=instance_id,
    )

    vehicle_count_hint = parse_vehicle_count_from_name(instance_id)
    return VrpInstance(
        instance_id=instance_id,
        set_id=set_id,
        dimension=dimension,
        capacity=capacity,
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        depot_ids=depot_ids,
        coords=coords,
        demands=demands,
        distance_matrix=distance_matrix,
        vehicle_count_hint=vehicle_count_hint,
        comment=metadata.get("COMMENT"),
    )


def build_distance_matrix(
    dimension: int,
    edge_weight_type: str,
    edge_weight_format: Optional[str],
    coords: Dict[int, Tuple[Optional[float], Optional[float]]],
    edge_weight_values: List[float],
    instance_id: str,
) -> List[List[int]]:
    if edge_weight_type == "EUC_2D":
        matrix = [[0] * dimension for _ in range(dimension)]
        for i in range(1, dimension + 1):
            xi, yi = coords[i]
            if xi is None or yi is None:
                raise ValueError(f"{instance_id}: missing coordinates for node {i}")
            for j in range(i + 1, dimension + 1):
                xj, yj = coords[j]
                if xj is None or yj is None:
                    raise ValueError(f"{instance_id}: missing coordinates for node {j}")
                dist = nint(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
                matrix[i - 1][j - 1] = dist
                matrix[j - 1][i - 1] = dist
        return matrix

    if edge_weight_type == "EXPLICIT" and edge_weight_format == "LOWER_ROW":
        needed = dimension * (dimension - 1) // 2
        if len(edge_weight_values) != needed:
            raise ValueError(
                f"{instance_id}: EDGE_WEIGHT_SECTION size mismatch, expected {needed}, got {len(edge_weight_values)}"
            )
        matrix = [[0] * dimension for _ in range(dimension)]
        idx = 0
        for i in range(1, dimension):
            for j in range(0, i):
                value = int(round(edge_weight_values[idx]))
                idx += 1
                matrix[i][j] = value
                matrix[j][i] = value
        return matrix

    raise ValueError(
        f"{instance_id}: unsupported EDGE_WEIGHT_TYPE/FORMAT: {edge_weight_type}/{edge_weight_format}"
    )


def parse_vehicle_count_from_name(instance_id: str) -> Optional[int]:
    match = re.search(r"-k(\d+)$", instance_id, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def parse_sol_file(sol_path: Path) -> SolData:
    if not sol_path.exists():
        return SolData(known_opt_cost=None, routes=[])

    lines = [ln.strip() for ln in sol_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    routes: List[List[int]] = []
    known_opt_cost: Optional[float] = None
    for line in lines:
        if line.lower().startswith("route"):
            route_text = line.split(":", 1)[1].strip()
            route = [int(x) for x in route_text.split()] if route_text else []
            routes.append(route)
        elif line.lower().startswith("cost"):
            cost_text = line.split(maxsplit=1)[1]
            known_opt_cost = float(cost_text)

    return SolData(known_opt_cost=known_opt_cost, routes=routes)


def to_base_json(instance: VrpInstance, sol: SolData) -> Dict:
    depot_id = instance.depot_ids[0]
    customers = []
    for node_id in range(1, instance.dimension + 1):
        if node_id == depot_id:
            continue
        x, y = instance.coords[node_id]
        customers.append(
            {
                "id": node_id,
                "x": x,
                "y": y,
                "demand": instance.demands[node_id],
            }
        )

    depot_x, depot_y = instance.coords[depot_id]
    return {
        "instance_id": instance.instance_id,
        "set_id": instance.set_id,
        "type": "CVRP",
        "dimension": instance.dimension,
        "vehicle_capacity": instance.capacity,
        "vehicle_count_hint": instance.vehicle_count_hint,
        "depot": {"id": depot_id, "x": depot_x, "y": depot_y, "demand": instance.demands[depot_id]},
        "customers": customers,
        "node_id_order": list(range(1, instance.dimension + 1)),
        "distance_matrix": instance.distance_matrix,
        "distance_metric": instance.edge_weight_type,
        "known_opt_cost": sol.known_opt_cost,
        "known_opt_routes": sol.routes,
    }


def to_meta_json(instance: VrpInstance, source_vrp: Path, source_sol: Path) -> Dict:
    return {
        "instance_id": instance.instance_id,
        "source": {
            "vrp_file": str(source_vrp),
            "sol_file": str(source_sol) if source_sol.exists() else None,
        },
        "parser_version": "1.0",
        "raw_comment": instance.comment,
        "edge_weight_type": instance.edge_weight_type,
        "edge_weight_format": instance.edge_weight_format,
        "depot_ids": instance.depot_ids,
    }


def validate_instance(instance_json: Dict) -> List[str]:
    errors: List[str] = []
    dim = instance_json["dimension"]
    depot_id = instance_json["depot"]["id"]
    customers = instance_json["customers"]
    matrix = instance_json["distance_matrix"]

    if len(customers) != dim - 1:
        errors.append(f"customer_count mismatch: expected {dim - 1}, got {len(customers)}")

    if instance_json["depot"]["demand"] != 0:
        errors.append("depot demand must be 0")

    for c in customers:
        if c["demand"] <= 0:
            errors.append(f"customer {c['id']} demand must be > 0")

    if depot_id < 1 or depot_id > dim:
        errors.append("invalid depot id")

    if len(matrix) != dim:
        errors.append("distance matrix row size mismatch")
        return errors

    for i in range(dim):
        if len(matrix[i]) != dim:
            errors.append(f"distance matrix col size mismatch at row {i + 1}")
            continue
        if matrix[i][i] != 0:
            errors.append(f"distance diagonal at node {i + 1} is not 0")
        for j in range(i + 1, dim):
            if matrix[i][j] != matrix[j][i]:
                errors.append(f"distance asymmetry at ({i + 1},{j + 1})")
                break

    return errors


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_sets_root = project_root / "01_raw_data" / "cvrplib_sets"
    processed_base_dir = project_root / "02_processed_data" / "classic" / "base"
    processed_meta_dir = project_root / "02_processed_data" / "classic" / "meta"
    raw_export_dir = project_root / "01_raw_data" / "raw_snapshot" / "raw"
    docs_dir = project_root / "06_docs" / "pipeline_docs"
    index_csv_path = project_root / "02_processed_data" / "classic" / "index.csv"
    qa_report_path = docs_dir / "qa_report.md"
    schema_path = docs_dir / "data_schema.md"

    processed_base_dir.mkdir(parents=True, exist_ok=True)
    processed_meta_dir.mkdir(parents=True, exist_ok=True)
    raw_export_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict] = []
    qa_errors: Dict[str, List[str]] = {}

    total_instances = 0
    for set_id in DATASET_SETS:
        source_dir = raw_sets_root / set_id
        if not source_dir.exists():
            continue
        for vrp_path in sorted(source_dir.glob("*.vrp")):
            total_instances += 1
            sol_path = vrp_path.with_suffix(".sol")
            instance = parse_vrp_file(vrp_path=vrp_path, set_id=set_id)
            sol = parse_sol_file(sol_path)
            base_json = to_base_json(instance, sol)
            meta_json = to_meta_json(instance, vrp_path, sol_path)

            base_out_path = processed_base_dir / f"{instance.instance_id}.base.json"
            meta_out_path = processed_meta_dir / f"{instance.instance_id}.meta.json"
            write_json(base_out_path, base_json)
            write_json(meta_out_path, meta_json)

            errors = validate_instance(base_json)
            if errors:
                qa_errors[instance.instance_id] = errors

            total_demand = sum(c["demand"] for c in base_json["customers"])
            max_distance = max(max(row) for row in base_json["distance_matrix"])
            index_rows.append(
                {
                    "instance_id": instance.instance_id,
                    "set_id": set_id,
                    "dimension": instance.dimension,
                    "customers": instance.dimension - 1,
                    "vehicle_count_hint": instance.vehicle_count_hint or "",
                    "vehicle_capacity": instance.capacity,
                    "edge_weight_type": instance.edge_weight_type,
                    "edge_weight_format": instance.edge_weight_format or "",
                    "has_coordinates": int(any(c["x"] is not None for c in base_json["customers"])),
                    "has_opt_solution": int(sol.known_opt_cost is not None),
                    "known_opt_cost": sol.known_opt_cost if sol.known_opt_cost is not None else "",
                    "known_opt_routes_count": len(sol.routes),
                    "total_demand": total_demand,
                    "max_distance": max_distance,
                }
            )

            # Keep exact raw copies for traceability.
            exported_vrp = raw_export_dir / set_id / vrp_path.name
            exported_vrp.parent.mkdir(parents=True, exist_ok=True)
            exported_vrp.write_text(vrp_path.read_text(encoding="utf-8"), encoding="utf-8")
            if sol_path.exists():
                exported_sol = raw_export_dir / set_id / sol_path.name
                exported_sol.write_text(sol_path.read_text(encoding="utf-8"), encoding="utf-8")

    index_rows.sort(key=lambda r: r["instance_id"])
    index_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with index_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "set_id",
                "dimension",
                "customers",
                "vehicle_count_hint",
                "vehicle_capacity",
                "edge_weight_type",
                "edge_weight_format",
                "has_coordinates",
                "has_opt_solution",
                "known_opt_cost",
                "known_opt_routes_count",
                "total_demand",
                "max_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    by_set_counts: Dict[str, int] = {}
    by_set_dim_stats: Dict[str, List[int]] = {}
    for row in index_rows:
        sid = row["set_id"]
        by_set_counts[sid] = by_set_counts.get(sid, 0) + 1
        by_set_dim_stats.setdefault(sid, []).append(int(row["dimension"]))

    qa_lines = [
        "# QA Report (Classic CVRP)",
        "",
        f"- Total instances processed: {total_instances}",
        f"- Total index rows: {len(index_rows)}",
        f"- Instances with parser/validation errors: {len(qa_errors)}",
        "",
        "## Coverage by Set",
        "",
    ]
    for sid in sorted(by_set_counts):
        dims = by_set_dim_stats[sid]
        qa_lines.append(
            f"- {sid}: count={by_set_counts[sid]}, dimension_range=[{min(dims)}, {max(dims)}]"
        )

    qa_lines += [
        "",
        "## Validation Checklist",
        "",
        "- depot demand equals 0",
        "- all customer demands > 0",
        "- customer count equals dimension - 1",
        "- distance matrix shape equals dimension x dimension",
        "- distance matrix is symmetric",
        "- distance matrix diagonal is zero",
        "",
        "## Error Details",
        "",
    ]
    if not qa_errors:
        qa_lines.append("- No validation errors detected.")
    else:
        for instance_id in sorted(qa_errors):
            qa_lines.append(f"- {instance_id}: {'; '.join(qa_errors[instance_id])}")

    qa_report_path.write_text("\n".join(qa_lines) + "\n", encoding="utf-8")

    schema_lines = [
        "# Data Schema (Classic CVRP)",
        "",
        "## File: `02_processed_data/classic/base/<instance_id>.base.json`",
        "",
        "- `instance_id` (string): instance name, e.g. `A-n32-k5`",
        "- `set_id` (string): one of `A/B/E/F/M/P`",
        "- `type` (string): `CVRP`",
        "- `dimension` (int): total nodes including depot",
        "- `vehicle_capacity` (int): truck capacity",
        "- `vehicle_count_hint` (int|null): parsed from instance name suffix `-kX`",
        "- `depot` (object): `{id, x, y, demand}`",
        "- `customers` (array): list of `{id, x, y, demand}` excluding depot",
        "- `node_id_order` (array[int]): node order used by distance matrix",
        "- `distance_matrix` (array[array[int]]): full symmetric matrix",
        "- `distance_metric` (string): `EUC_2D` or `EXPLICIT`",
        "- `known_opt_cost` (number|null): from `.sol` if available",
        "- `known_opt_routes` (array[array[int]]): parsed route sequence from `.sol`",
        "",
        "## File: `02_processed_data/classic/meta/<instance_id>.meta.json`",
        "",
        "- `instance_id` (string)",
        "- `source.vrp_file` (string): absolute path to source `.vrp`",
        "- `source.sol_file` (string|null): absolute path to source `.sol` if available",
        "- `parser_version` (string): parser version label",
        "- `raw_comment` (string|null): original `COMMENT` field",
        "- `edge_weight_type` (string)",
        "- `edge_weight_format` (string|null)",
        "- `depot_ids` (array[int])",
        "",
        "## File: `02_processed_data/classic/index.csv`",
        "",
        "One row per instance for quick filtering and statistics.",
    ]
    schema_path.write_text("\n".join(schema_lines) + "\n", encoding="utf-8")

    print(f"Processed instances: {total_instances}")
    print(f"Index path: {index_csv_path}")
    print(f"QA report path: {qa_report_path}")


if __name__ == "__main__":
    main()
