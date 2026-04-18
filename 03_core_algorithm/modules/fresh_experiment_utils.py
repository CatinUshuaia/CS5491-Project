import ast
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import sys

from ortools_compat import add_ortools_dll_directory

add_ortools_dll_directory()
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
    ORTOOLS_IMPORT_ERROR = ""
except (ImportError, OSError) as exc:
    pywrapcp = None
    routing_enums_pb2 = None
    ORTOOLS_AVAILABLE = False
    ORTOOLS_IMPORT_ERROR = str(exc)

import numpy as np
import openai
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
METHODS_DIR = ROOT_DIR / "03_core_algorithm" / "methods_advanced"
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))

from duplicate_checking import canonicalize_expr, dedup_candidates_advanced

DEFAULT_LLM_MODEL = "gpt-5-nano"
ORTOOLS_SOLVER_NAME = "ortools" if ORTOOLS_AVAILABLE else "ortools_fallback_greedy"
TOKEN_LOG: List[Dict] = []
_ORTOOLS_WARNING_EMITTED = False


def euclidean_distance(a, b) -> int:
    return int(round(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))))


def build_distance_matrix(coords: List[List[float]]) -> np.ndarray:
    n = len(coords)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(coords[i], coords[j])
            mat[i, j] = dist
            mat[j, i] = dist
    return mat


def get_distance_matrix(instance: Dict) -> np.ndarray:
    dist = instance.get("distance_matrix")
    if dist is not None:
        return np.asarray(dist, dtype=float)
    return build_distance_matrix(instance["coords"]).astype(float)


def route_distance(route: List[int], dist_matrix: np.ndarray, depot: int = 0) -> float:
    customer_route = [node for node in route if node != depot]
    if not customer_route:
        return 0.0
    total = float(dist_matrix[depot, customer_route[0]])
    for i in range(len(customer_route) - 1):
        total += float(dist_matrix[customer_route[i], customer_route[i + 1]])
    total += float(dist_matrix[customer_route[-1], depot])
    return total


def check_capacity_feasibility(route: List[int], demands: List[int], capacity: int):
    load = sum(demands[c] for c in route)
    return load <= capacity, load


def _normalize_objective_weights(raw: Optional[Dict]) -> Dict[str, float]:
    raw = raw or {}
    return {
        "distance": float(raw.get("distance", 1.0)),
        "late": float(raw.get("late", raw.get("lateness", 1.0))),
        "spoil": float(raw.get("spoil", raw.get("spoilage", 1.0))),
    }


def evaluate_fresh_routes(routes, instance, hard_time_window=False, hard_freshness=False):
    depot = instance.get("depot", 0)
    demands = instance["demands"]
    capacity = instance["capacity"]
    dist_matrix = get_distance_matrix(instance)

    depot_start = instance.get("depot_start_time_min", 0)
    service_time = instance.get("service_time_min", [0] * len(demands))
    ready_time = instance.get("ready_time_min", [0] * len(demands))
    due_time = instance.get("due_time_min", [10**9] * len(demands))
    max_travel_time = instance.get("max_travel_time_min", [10**9] * len(demands))
    late_penalty_per_min = instance.get("late_penalty_per_min", [0] * len(demands))
    spoilage_penalty = instance.get("spoilage_penalty", [0] * len(demands))

    obj_w = _normalize_objective_weights(instance.get("objective_weights"))
    w_d = obj_w["distance"]
    w_l = obj_w["late"]
    w_s = obj_w["spoil"]

    total_distance = 0.0
    total_late_penalty = 0.0
    total_spoil_penalty = 0.0
    total_lateness_min = 0.0
    total_spoil_min = 0.0
    late_customer_count = 0
    spoil_customer_count = 0
    capacity_violations = 0
    time_window_violations = 0
    freshness_violations = 0
    visited = set()
    feasible = True

    for route in routes:
        customer_route = [node for node in route if node != depot]
        cap_ok, _load = check_capacity_feasibility(customer_route, demands, capacity)
        if not cap_ok:
            feasible = False
            capacity_violations += 1

        total_distance += route_distance(customer_route, dist_matrix, depot=depot)
        current_time = float(depot_start)
        current_node = depot

        for cust in customer_route:
            visited.add(cust)
            travel = float(dist_matrix[current_node, cust])
            arrival = current_time + travel
            service_start = max(arrival, float(ready_time[cust]))
            lateness = max(0.0, arrival - float(due_time[cust]))
            travel_time_from_depot = arrival - float(depot_start)
            spoil = max(0.0, travel_time_from_depot - float(max_travel_time[cust]))

            late_cost = lateness * float(late_penalty_per_min[cust])
            spoil_cost = spoil * float(spoilage_penalty[cust])

            total_lateness_min += lateness
            total_spoil_min += spoil
            total_late_penalty += late_cost
            total_spoil_penalty += spoil_cost

            if lateness > 0:
                late_customer_count += 1
                time_window_violations += 1
                if hard_time_window:
                    feasible = False

            if spoil > 0:
                spoil_customer_count += 1
                freshness_violations += 1
                if hard_freshness:
                    feasible = False

            current_time = service_start + float(service_time[cust])
            current_node = cust

    n_customers = len(demands) - 1
    expected = set(range(1, n_customers + 1))
    missed_customers = sorted(expected - visited)
    duplicate_or_invalid = len(visited) != sum(len([node for node in r if node != depot]) for r in routes)
    if missed_customers or duplicate_or_invalid:
        feasible = False

    objective = w_d * total_distance + w_l * total_late_penalty + w_s * total_spoil_penalty
    return {
        "feasible": feasible,
        "objective": float(objective),
        "total_distance": float(total_distance),
        "total_late_penalty": float(total_late_penalty),
        "total_spoil_penalty": float(total_spoil_penalty),
        "total_lateness_min": float(total_lateness_min),
        "total_spoil_min": float(total_spoil_min),
        "late_customer_count": int(late_customer_count),
        "spoil_customer_count": int(spoil_customer_count),
        "capacity_violations": int(capacity_violations),
        "time_window_violations": int(time_window_violations),
        "freshness_violations": int(freshness_violations),
        "num_routes": len(routes),
        "missed_customers": missed_customers,
    }


def greedy_fresh_solver(instance: Dict) -> List[List[int]]:
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]
    dist_matrix = get_distance_matrix(instance)
    unvisited = set(range(1, instance["num_nodes"]))
    routes = []

    while unvisited:
        route = [depot]
        remaining = capacity
        current = depot
        while True:
            feasible_customers = [c for c in unvisited if demands[c] <= remaining]
            if not feasible_customers:
                break
            next_customer = sorted(
                feasible_customers,
                key=lambda c: (-demands[c], float(dist_matrix[current, c]), c),
            )[0]
            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining -= demands[next_customer]
            current = next_customer
        route.append(depot)
        routes.append(route)
    return routes


def heuristic_fresh_solver(instance: Dict, score_fn) -> List[List[int]]:
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]

    service_time = instance.get("service_time_min", [0] * len(coords))
    ready_time = instance.get("ready_time_min", [0] * len(coords))
    due_time = instance.get("due_time_min", [10**9] * len(coords))
    max_travel_time = instance.get("max_travel_time_min", [10**9] * len(coords))
    depot_start_time = instance.get("depot_start_time_min", 0)

    dist_matrix = get_distance_matrix(instance)
    unvisited = set(range(1, len(coords)))
    routes = []

    while unvisited:
        route = [depot]
        current = depot
        remaining = capacity
        route_start_time = float(depot_start_time)
        current_time = float(depot_start_time)

        while True:
            feasible_candidates = [c for c in unvisited if demands[c] <= remaining]
            if not feasible_candidates:
                break

            next_customer = min(
                feasible_candidates,
                key=lambda c: score_fn(
                    current=current,
                    c=c,
                    instance=instance,
                    remaining=remaining,
                    dist_matrix=dist_matrix,
                    current_time=current_time,
                    route_start_time=route_start_time,
                    route=route,
                    travel_to_c=float(dist_matrix[current, c]),
                    est_arrival=float(current_time + dist_matrix[current, c]),
                    est_service_start=float(max(current_time + dist_matrix[current, c], ready_time[c])),
                    est_finish_service=float(max(current_time + dist_matrix[current, c], ready_time[c]) + service_time[c]),
                    est_travel_time_from_depot=float(current_time + dist_matrix[current, c] - route_start_time),
                    est_lateness=float(max(0.0, (current_time + dist_matrix[current, c]) - due_time[c])),
                    est_spoil=float(
                        max(0.0, (current_time + dist_matrix[current, c] - route_start_time) - max_travel_time[c])
                    ),
                ),
            )

            travel = float(dist_matrix[current, next_customer])
            arrival = current_time + travel
            service_start = max(arrival, float(ready_time[next_customer]))
            finish_service = service_start + float(service_time[next_customer])

            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining -= demands[next_customer]
            current = next_customer
            current_time = finish_service

        route.append(depot)
        routes.append(route)

    return routes


def nearest_neighbor_fresh(instance: Dict) -> List[List[int]]:
    return heuristic_fresh_solver(instance, lambda **kwargs: kwargs["travel_to_c"])


def make_score_fn_from_expression_fresh(expr: str):
    def score_fn(
        current,
        c,
        instance,
        remaining,
        dist_matrix,
        current_time,
        route_start_time,
        route,
        travel_to_c,
        est_arrival,
        est_service_start,
        est_finish_service,
        est_travel_time_from_depot,
        est_lateness,
        est_spoil,
    ):
        return float(
            eval(
                expr,
                {"__builtins__": {}},
                {
                    "current": current,
                    "c": c,
                    "instance": instance,
                    "remaining": remaining,
                    "dist_matrix": dist_matrix,
                    "current_time": current_time,
                    "route_start_time": route_start_time,
                    "route": route,
                    "travel_to_c": travel_to_c,
                    "est_arrival": est_arrival,
                    "est_service_start": est_service_start,
                    "est_finish_service": est_finish_service,
                    "est_travel_time_from_depot": est_travel_time_from_depot,
                    "est_lateness": est_lateness,
                    "est_spoil": est_spoil,
                    "len": len,
                    "min": min,
                    "max": max,
                    "abs": abs,
                },
            )
        )

    return score_fn


def ortools_fresh_solver(instance: Dict, time_limit_sec: int = 3) -> List[List[int]]:
    global _ORTOOLS_WARNING_EMITTED
    if not ORTOOLS_AVAILABLE:
        if not _ORTOOLS_WARNING_EMITTED:
            print(
                f"[warn] OR-Tools unavailable for fresh, fallback to greedy solver: {ORTOOLS_IMPORT_ERROR}",
                flush=True,
            )
            _ORTOOLS_WARNING_EMITTED = True
        return greedy_fresh_solver(instance)

    dist = get_distance_matrix(instance)
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]
    n = instance["num_nodes"]
    num_vehicles = max(1, math.ceil(sum(demands) / capacity))

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist[from_node, to_node])

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,
        [int(capacity)] * num_vehicles,
        True,
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = int(time_limit_sec)
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return greedy_fresh_solver(instance)

    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = [depot]
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        route.append(depot)
        if len(route) > 2:
            routes.append(route)
    return routes or [[depot, depot]]


def evaluate_fresh_solver(solver, instance: Dict, return_routes: bool = False) -> Dict:
    start = time.time()
    routes = solver(instance)
    runtime_sec = time.time() - start
    result = evaluate_fresh_routes(routes, instance)
    result["runtime_sec"] = float(runtime_sec)
    if return_routes:
        result["routes"] = routes
    return result


def load_fresh_instance(json_path: Path) -> Dict:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    depot = data["depot"]
    customers = data["customers"]

    coords = [[depot["x"], depot["y"]]]
    demands = [0]
    service_time_min = [0]
    ready_time_min = [0]
    due_time_min = [10**9]
    freshness_class = ["depot"]
    max_travel_time_min = [10**9]
    temp_zone = ["depot"]
    late_penalty_per_min = [0]
    spoilage_penalty = [0]

    for c in customers:
        coords.append([c["x"], c["y"]])
        demands.append(c["demand"])
        service_time_min.append(c.get("service_time_min", 0))
        ready_time_min.append(c.get("ready_time_min", 0))
        due_time_min.append(c.get("due_time_min", 10**9))
        freshness_class.append(c.get("freshness_class", "low"))
        max_travel_time_min.append(c.get("max_travel_time_min", 10**9))
        temp_zone.append(c.get("temp_zone", "ambient"))
        late_penalty_per_min.append(c.get("late_penalty_per_min", 0))
        spoilage_penalty.append(c.get("spoilage_penalty", 0))

    return {
        "name": data["instance_id"],
        "problem_variant": data.get("problem_variant", "CVRP_FRESH"),
        "depot": 0,
        "coords": coords,
        "demands": demands,
        "capacity": data["vehicle_capacity"],
        "num_nodes": len(coords),
        "distance_matrix": data.get("distance_matrix"),
        "depot_start_time_min": data.get("depot_start_time_min", 0),
        "objective_weights": _normalize_objective_weights(data.get("objective_weights")),
        "service_time_min": service_time_min,
        "ready_time_min": ready_time_min,
        "due_time_min": due_time_min,
        "freshness_class": freshness_class,
        "max_travel_time_min": max_travel_time_min,
        "temp_zone": temp_zone,
        "late_penalty_per_min": late_penalty_per_min,
        "spoilage_penalty": spoilage_penalty,
        "raw": data,
    }


def load_multiple_fresh_instances(fresh_dir: str, limit=None) -> List[Dict]:
    json_files = sorted(Path(fresh_dir).glob("*.fresh.json"))
    if limit is not None:
        json_files = json_files[:limit]
    return [load_fresh_instance(p) for p in json_files]


def evaluate_expression_on_instances_fresh_dynamic(instances: List[Dict], expr: str) -> pd.DataFrame:
    score_fn = make_score_fn_from_expression_fresh(expr)
    rows = []
    for inst in instances:
        result = evaluate_fresh_solver(lambda instance: heuristic_fresh_solver(instance, score_fn), inst)
        rows.append(
            {
                "instance": inst["name"],
                "expression": expr,
                "feasible": result["feasible"],
                "objective": result["objective"],
                "cost": result["objective"],
                "distance": result["total_distance"],
                "late_penalty": result["total_late_penalty"],
                "spoil_penalty": result["total_spoil_penalty"],
                "late_customers": result["late_customer_count"],
                "spoil_customers": result["spoil_customer_count"],
                "num_routes": result["num_routes"],
                "runtime_sec": result["runtime_sec"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_expression_list_on_instances_fresh(instances: List[Dict], expressions: List[str]) -> pd.DataFrame:
    expressions = list(expressions)
    if not expressions:
        return pd.DataFrame()
    dfs = [evaluate_expression_on_instances_fresh_dynamic(instances, expr) for expr in expressions]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def summarize_expression_fresh(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()
    summary_df = (
        detail_df.groupby("expression", as_index=False)
        .agg(
            num_instances=("instance", "count"),
            feasible_rate=("feasible", "mean"),
            avg_objective=("objective", "mean"),
            avg_cost=("cost", "mean"),
            avg_distance=("distance", "mean"),
            avg_late_penalty=("late_penalty", "mean"),
            avg_spoil_penalty=("spoil_penalty", "mean"),
            avg_late_customers=("late_customers", "mean"),
            avg_spoil_customers=("spoil_customers", "mean"),
            avg_num_routes=("num_routes", "mean"),
            avg_runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(["feasible_rate", "avg_objective"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary_df


def evaluate_named_solver_on_instances_fresh(instances: List[Dict], solver_name: str, solver_fn) -> pd.DataFrame:
    rows = []
    for inst in instances:
        result = evaluate_fresh_solver(solver_fn, inst)
        rows.append(
            {
                "instance": inst["name"],
                "expression": solver_name,
                "feasible": result["feasible"],
                "objective": result["objective"],
                "cost": result["objective"],
                "distance": result["total_distance"],
                "late_penalty": result["total_late_penalty"],
                "spoil_penalty": result["total_spoil_penalty"],
                "late_customers": result["late_customer_count"],
                "spoil_customers": result["spoil_customer_count"],
                "num_routes": result["num_routes"],
                "runtime_sec": result["runtime_sec"],
            }
        )
    return pd.DataFrame(rows)


def expression_complexity(expr: str) -> int:
    expr = str(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return len(expr) + 5 * sum(expr.count(op) for op in ["+", "-", "*", "/", "%"])

    node_count = 0
    op_penalty = 0
    names = set()
    op_weights = {
        ast.Add: 1,
        ast.Sub: 1,
        ast.Mult: 2,
        ast.Div: 3,
        ast.Mod: 3,
        ast.Pow: 4,
        ast.USub: 1,
    }
    for node in ast.walk(tree):
        node_count += 1
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Subscript):
            op_penalty += 2
        elif isinstance(node, ast.Call):
            op_penalty += 8
        for op_t, weight in op_weights.items():
            if isinstance(node, op_t):
                op_penalty += weight
    return int(node_count + op_penalty + len(names))


def filter_expressions_by_complexity(expressions: List[str], max_complexity=None) -> List[str]:
    if max_complexity is None:
        return list(expressions)
    return [expr for expr in expressions if expression_complexity(expr) <= max_complexity]


def dedup_expressions(
    candidate_expressions: List[str],
    probe_instances=None,
    eval_fn=None,
    use_behavioral: bool = False,
) -> List[str]:
    if probe_instances is not None and eval_fn is not None:
        return dedup_candidates_advanced(
            candidate_expressions,
            probe_instances=probe_instances,
            eval_fn=lambda expr, instances: eval_fn(instances, [expr]),
            use_behavioral=use_behavioral,
        )

    kept = []
    seen_canonical = set()
    for expr in candidate_expressions:
        canon = canonicalize_expr(expr)
        if canon in seen_canonical:
            continue
        seen_canonical.add(canon)
        kept.append(canon)
    return kept


def make_behavior_signature_from_summary_row(
    row,
    objective_round=2,
    route_round=2,
    feas_round=3,
    late_pen_round=2,
    spoil_pen_round=2,
    complexity_bucket=5,
):
    avg_objective = row.get("avg_objective", np.inf)
    avg_num_routes = row.get("avg_num_routes", np.inf)
    feasible_rate = row.get("feasible_rate", 0.0)
    avg_late_penalty = row.get("avg_late_penalty", np.inf)
    avg_spoil_penalty = row.get("avg_spoil_penalty", np.inf)
    complexity = row.get("complexity", np.inf)
    if pd.isna(avg_objective):
        avg_objective = np.inf
    if pd.isna(avg_num_routes):
        avg_num_routes = np.inf
    if pd.isna(feasible_rate):
        feasible_rate = 0.0
    if pd.isna(avg_late_penalty):
        avg_late_penalty = np.inf
    if pd.isna(avg_spoil_penalty):
        avg_spoil_penalty = np.inf
    if pd.isna(complexity):
        complexity = np.inf
    complexity_bin = complexity if not np.isfinite(complexity) else int(complexity // complexity_bucket)
    return (
        round(float(avg_objective), objective_round) if np.isfinite(avg_objective) else np.inf,
        round(float(avg_num_routes), route_round) if np.isfinite(avg_num_routes) else np.inf,
        round(float(feasible_rate), feas_round),
        round(float(avg_late_penalty), late_pen_round) if np.isfinite(avg_late_penalty) else np.inf,
        round(float(avg_spoil_penalty), spoil_pen_round) if np.isfinite(avg_spoil_penalty) else np.inf,
        complexity_bin,
    )


def add_novelty_columns(summary_df: pd.DataFrame, archive_signatures=None) -> pd.DataFrame:
    if archive_signatures is None:
        archive_signatures = set()
    df = summary_df.copy()
    df["behavior_signature"] = df.apply(make_behavior_signature_from_summary_row, axis=1)
    df["is_novel"] = ~df["behavior_signature"].isin(archive_signatures)
    return df


def update_archive_signatures(summary_df: pd.DataFrame, archive_signatures=None, only_novel=True):
    if archive_signatures is None:
        archive_signatures = set()
    df = summary_df.copy()
    if "behavior_signature" not in df.columns:
        df["behavior_signature"] = df.apply(make_behavior_signature_from_summary_row, axis=1)
    if only_novel and "is_novel" in df.columns:
        df = df[df["is_novel"]]
    out = set(archive_signatures)
    for sig in df["behavior_signature"]:
        out.add(sig)
    return out


def _safe_minmax_norm(series, larger_is_better=False):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    lo, hi = s.min(), s.max()
    norm = pd.Series(np.zeros(len(series)), index=series.index) if hi == lo else (s - lo) / (hi - lo)
    norm = norm.fillna(1.0)
    return 1.0 - norm if larger_is_better else norm


def sort_expression_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if "complexity" not in df.columns:
        df["complexity"] = df["expression"].apply(expression_complexity)
    n_feasible = _safe_minmax_norm(df["feasible_rate"], larger_is_better=True)
    n_objective = _safe_minmax_norm(df["avg_objective"], larger_is_better=False)
    n_distance = _safe_minmax_norm(df["avg_distance"], larger_is_better=False)
    n_late = _safe_minmax_norm(df["avg_late_penalty"], larger_is_better=False)
    n_spoil = _safe_minmax_norm(df["avg_spoil_penalty"], larger_is_better=False)
    n_routes = _safe_minmax_norm(df["avg_num_routes"], larger_is_better=False)
    n_complexity = _safe_minmax_norm(df["complexity"], larger_is_better=False)
    df["mo_score"] = (
        0.40 * n_feasible
        + 0.25 * n_objective
        + 0.10 * n_distance
        + 0.08 * n_late
        + 0.08 * n_spoil
        + 0.06 * n_routes
        + 0.03 * n_complexity
    )
    return df.sort_values(
        by=["feasible_rate", "avg_objective", "avg_num_routes", "complexity", "mo_score"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def generate_mock_expression_variants(seed_expr: str, n: int = 8) -> List[str]:
    variants = set()
    demand_terms = [
        "instance['demands'][c]",
        "2 * instance['demands'][c]",
        "0.5 * instance['demands'][c]",
    ]
    spoil_terms = ["est_spoil", "2 * est_spoil", "5 * est_spoil", "10 * est_spoil", "20 * est_spoil", "0.5 * est_spoil"]
    late_terms = ["est_lateness", "2 * est_lateness", "5 * est_lateness", "10 * est_lateness", "0.5 * est_lateness"]
    base_terms = ["travel_to_c"]
    templates = [
        "{base}",
        "{base} + {spoil}",
        "{base} + {late}",
        "{base} - {demand}",
        "{base} + {demand}",
        "{base} + {spoil} + {late}",
        "{base} + {spoil} - {demand}",
        "{base} + {late} - {demand}",
        "{base} + {spoil} + {late} - {demand}",
        "{base} + {spoil} + {late} + {demand}",
    ]
    for _ in range(n * 4):
        expr = random.choice(templates).format(
            base=random.choice(base_terms),
            demand=random.choice(demand_terms),
            spoil=random.choice(spoil_terms),
            late=random.choice(late_terms),
        )
        variants.add(expr)
        if len(variants) >= n:
            break
    variants.add(seed_expr)
    return list(variants)


def generate_mock_candidates_from_top_expressions(top_expressions: List[str], variants_per_expr: int = 8) -> List[str]:
    all_candidates = []
    for expr in top_expressions:
        all_candidates.extend(generate_mock_expression_variants(expr, n=variants_per_expr))
    return all_candidates


def _build_prompt(top_expressions: List[str], max_total: int = 16) -> str:
    seed_block = "\n".join([f"- {expr}" for expr in top_expressions])
    return f"""
Generate new candidate Python score expressions for a fresh-CVRP heuristic.
Lower score is better.

Available variables only:
- travel_to_c
- est_lateness
- est_spoil
- instance['demands'][c]
- remaining

Seed expressions:
{seed_block}

Rules:
1. Return ONLY JSON.
2. JSON format must be:
{{"expressions": ["expr1", "expr2", "..."]}}
3. At most {max_total} expressions.
4. Expressions must be single-line valid Python arithmetic expressions.
5. No explanations, markdown, or extra keys.
6. Prefer local variations of seed expressions.
""".strip()


def _try_parse_json_object(text: str) -> Optional[Dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", " ", expr.strip())


def _is_safe_expression(expr: str) -> bool:
    banned = ["__", "import", "exec", "eval", "open(", "os.", "sys.", "for ", "while ", "if ", "lambda", "def "]
    low = expr.lower()
    return not any(token in low for token in banned)


def _filter_valid_expressions(expressions: List[str], max_complexity: Optional[int]) -> List[str]:
    out = []
    for expr in expressions:
        if not isinstance(expr, str):
            continue
        expr = _normalize_expression(expr)
        if not expr or not _is_safe_expression(expr):
            continue
        if max_complexity is not None and expression_complexity(expr) > max_complexity:
            continue
        out.append(expr)
    return dedup_expressions(out)


def make_openai_client() -> Optional[openai.OpenAI]:
    api_key = os.getenv("CVRP_OPENAI_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    host = os.getenv("CVRP_OPENAI_HOST", "https://api.bltcy.ai").rstrip("/")
    return openai.OpenAI(base_url=f"{host}/v1", api_key=api_key, timeout=45)


def generate_candidates_with_llm(
    client,
    top_expressions: List[str],
    n_per_expr: int = 4,
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.4,
    max_complexity: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    max_total = min(20, max(8, n_per_expr * max(1, len(top_expressions))))
    prompt = _build_prompt(top_expressions, max_total=max_total)
    raw_text = ""
    usage = {}
    for i in range(2):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
            raw_text = (resp.choices[0].message.content or "").strip()
            break
        except Exception:
            if i == 1:
                raw_text = ""
            time.sleep(i + 1)
    payload = _try_parse_json_object(raw_text) if raw_text else None
    expressions = payload.get("expressions", []) if isinstance(payload, dict) else []
    valid = _filter_valid_expressions(expressions, max_complexity=max_complexity)
    TOKEN_LOG.append(
        {
            "mode": "llm",
            "top_expr_count": len(top_expressions),
            "raw_expr_count": len(expressions) if isinstance(expressions, list) else 0,
            "valid_expr_count": len(valid),
            **usage,
        }
    )
    if verbose:
        print(
            f"[fresh-llm] raw={len(expressions) if isinstance(expressions, list) else 0}, "
            f"valid={len(valid)}, tokens={usage.get('total_tokens')}",
            flush=True,
        )
    return valid
