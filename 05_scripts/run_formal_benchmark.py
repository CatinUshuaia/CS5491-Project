import json
import math
import time
import ast
from pathlib import Path
from typing import Dict, List

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT_DIR / "03_core_algorithm" / "modules"
METHODS_DIR = ROOT_DIR / "03_core_algorithm" / "methods_advanced"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))
if str(METHODS_DIR) not in sys.path:
    sys.path.append(str(METHODS_DIR))

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
import pandas as pd

from benchmark_experiment_workflow import run_formal_experiments, stratified_sample_instances
from duplicate_checking import canonicalize_expr, dedup_candidates_advanced


ORTOOLS_SOLVER_NAME = "ortools" if ORTOOLS_AVAILABLE else "ortools_fallback_greedy"
_ORTOOLS_WARNING_EMITTED = False


def total_distance(routes: List[List[int]], dist_matrix: List[List[int]]) -> float:
    total = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total += dist_matrix[route[i]][route[i + 1]]
    return float(total)


def check_feasibility(routes: List[List[int]], demands: List[int], capacity: int, depot: int = 0) -> bool:
    for route in routes:
        if len(route) < 2 or route[0] != depot or route[-1] != depot:
            return False
        load = sum(demands[node] for node in route if node != depot)
        if load > capacity:
            return False
    return True


def evaluate_solver(instance: Dict, solver_fn, solver_name: str) -> Dict:
    start = time.time()
    routes = solver_fn(instance)
    runtime_sec = time.time() - start
    feasible = check_feasibility(routes, instance["demands"], instance["capacity"], instance["depot"])
    cost = total_distance(routes, instance["distance_matrix"])
    return {
        "instance": instance["name"],
        "expression": solver_name,
        "cost": cost,
        "runtime_sec": runtime_sec,
        "feasible": bool(feasible),
        "num_routes": len(routes),
    }


def greedy_cvrp_solver(instance: Dict) -> List[List[int]]:
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]
    n = instance["num_nodes"]
    unvisited = set(range(n))
    unvisited.remove(depot)
    routes = []
    while unvisited:
        route = [depot]
        remaining = capacity
        current = depot
        while True:
            feasible_customers = [c for c in unvisited if demands[c] <= remaining]
            if not feasible_customers:
                break
            # Greedy by max demand first, tie-break by nearest distance.
            next_customer = sorted(
                feasible_customers,
                key=lambda c: (-demands[c], instance["distance_matrix"][current][c], c),
            )[0]
            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining -= demands[next_customer]
            current = next_customer
        route.append(depot)
        routes.append(route)
    return routes


def nearest_neighbor_cvrp_solver(instance: Dict) -> List[List[int]]:
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]
    dist = instance["distance_matrix"]
    n = instance["num_nodes"]
    unvisited = set(range(n))
    unvisited.remove(depot)
    routes = []
    while unvisited:
        route = [depot]
        remaining = capacity
        current = depot
        while True:
            feasible_customers = [c for c in unvisited if demands[c] <= remaining]
            if not feasible_customers:
                break
            next_customer = min(feasible_customers, key=lambda c: (dist[current][c], -demands[c], c))
            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining -= demands[next_customer]
            current = next_customer
        route.append(depot)
        routes.append(route)
    return routes


def nearest_neighbor_v2(instance: Dict) -> List[List[int]]:
    return nearest_neighbor_cvrp_solver(instance)


def ortools_cvrp_solver(instance: Dict, time_limit_sec: int = 3) -> List[List[int]]:
    global _ORTOOLS_WARNING_EMITTED
    if not ORTOOLS_AVAILABLE:
        if not _ORTOOLS_WARNING_EMITTED:
            print(
                f"[warn] OR-Tools unavailable, fallback to greedy solver: {ORTOOLS_IMPORT_ERROR}",
                flush=True,
            )
            _ORTOOLS_WARNING_EMITTED = True
        return greedy_cvrp_solver(instance)

    dist = instance["distance_matrix"]
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
        return int(dist[from_node][to_node])

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
        return greedy_cvrp_solver(instance)

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
    if not routes:
        routes = [[depot, depot]]
    return routes


def nn_score(current, c, instance, remaining, dist_matrix):
    return dist_matrix[current][c]


def heuristic_cvrp_solver(instance: Dict, score_fn) -> List[List[int]]:
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = instance["depot"]
    dist_matrix = instance["distance_matrix"]
    n = instance["num_nodes"]
    unvisited = set(range(n))
    unvisited.remove(depot)
    routes = []
    while unvisited:
        route = [depot]
        remaining = capacity
        current = depot
        while True:
            feasible_customers = [c for c in unvisited if demands[c] <= remaining]
            if not feasible_customers:
                break
            next_customer = min(
                feasible_customers, key=lambda c: score_fn(current, c, instance, remaining, dist_matrix)
            )
            route.append(next_customer)
            unvisited.remove(next_customer)
            remaining -= demands[next_customer]
            current = next_customer
        route.append(depot)
        routes.append(route)
    return routes


def make_score_fn_from_expression(expr: str):
    allowed = {
        "__builtins__": {},
        "abs": abs,
        "min": min,
        "max": max,
    }

    def score_fn(current, c, instance, remaining, dist_matrix):
        local_vars = {
            "current": current,
            "c": c,
            "instance": instance,
            "remaining": remaining,
            "dist_matrix": dist_matrix,
        }
        return float(eval(expr, allowed, local_vars))

    return score_fn


def evaluate_expression_on_instances(instances: List[Dict], expr: str) -> pd.DataFrame:
    score_fn = make_score_fn_from_expression(expr)
    rows = []
    for inst in instances:
        result = evaluate_solver(inst, lambda x: heuristic_cvrp_solver(x, score_fn), expr)
        rows.append(result)
    return pd.DataFrame(rows)


def evaluate_expression_list_on_instances(instances: List[Dict], expressions: List[str]) -> pd.DataFrame:
    expressions = list(expressions)
    if len(expressions) == 0:
        return pd.DataFrame()
    dfs = [evaluate_expression_on_instances(instances, expr) for expr in expressions]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def summarize_expression_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return (
        df.groupby("expression", as_index=False)
        .agg(
            num_instances=("instance", "count"),
            feasible_rate=("feasible", "mean"),
            avg_cost=("cost", "mean"),
            avg_runtime_sec=("runtime_sec", "mean"),
            avg_num_routes=("num_routes", "mean"),
        )
        .sort_values(["feasible_rate", "avg_cost"], ascending=[False, True])
        .reset_index(drop=True)
    )


def evaluate_named_solver_on_instances(instances: List[Dict], solver_name: str, solver_fn) -> pd.DataFrame:
    rows = []
    for inst in instances:
        rows.append(evaluate_solver(inst, solver_fn, solver_name))
    return pd.DataFrame(rows)


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


def expression_complexity(expr: str) -> int:
    expr = str(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        op_count = sum(expr.count(op) for op in ["+", "-", "*", "/", "%"])
        return len(expr) + 5 * op_count

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
    return [e for e in expressions if expression_complexity(e) <= max_complexity]


def make_behavior_signature_from_summary_row(
    row,
    cost_round=2,
    route_round=2,
    feas_round=3,
    complexity_bucket=5,
):
    avg_cost = row.get("avg_cost", np.inf)
    avg_num_routes = row.get("avg_num_routes", np.inf)
    feasible_rate = row.get("feasible_rate", 0.0)
    complexity = row.get("complexity", np.inf)
    if pd.isna(avg_cost):
        avg_cost = np.inf
    if pd.isna(avg_num_routes):
        avg_num_routes = np.inf
    if pd.isna(feasible_rate):
        feasible_rate = 0.0
    if pd.isna(complexity):
        complexity = np.inf
    complexity_bin = complexity if not np.isfinite(complexity) else int(complexity // complexity_bucket)
    return (
        round(float(avg_cost), cost_round) if np.isfinite(avg_cost) else np.inf,
        round(float(avg_num_routes), route_round) if np.isfinite(avg_num_routes) else np.inf,
        round(float(feasible_rate), feas_round),
        complexity_bin,
    )


def add_novelty_columns(summary_df: pd.DataFrame, archive_signatures=None) -> pd.DataFrame:
    if archive_signatures is None:
        archive_signatures = set()
    out = summary_df.copy()
    out["behavior_signature"] = out.apply(make_behavior_signature_from_summary_row, axis=1)
    out["is_novel"] = ~out["behavior_signature"].isin(archive_signatures)
    return out


def update_archive_signatures(summary_df: pd.DataFrame, archive_signatures=None, only_novel=True):
    if archive_signatures is None:
        archive_signatures = set()
    out = set(archive_signatures)
    for _, row in summary_df.iterrows():
        if only_novel and ("is_novel" in row) and (not bool(row["is_novel"])):
            continue
        out.add(row["behavior_signature"])
    return out


def _safe_minmax_norm(series, larger_is_better=False):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    lo, hi = s.min(), s.max()
    if hi == lo:
        norm = pd.Series(np.zeros(len(series)), index=series.index)
    else:
        norm = (s - lo) / (hi - lo)
    norm = norm.fillna(1.0)
    return 1.0 - norm if larger_is_better else norm


def sort_expression_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if "complexity" not in df.columns:
        df["complexity"] = df["expression"].apply(expression_complexity)

    n_feasible = _safe_minmax_norm(df["feasible_rate"], larger_is_better=True)
    n_cost = _safe_minmax_norm(df["avg_cost"], larger_is_better=False)
    n_routes = _safe_minmax_norm(df["avg_num_routes"], larger_is_better=False)
    n_complexity = _safe_minmax_norm(df["complexity"], larger_is_better=False)

    df["mo_score"] = (
        0.55 * n_feasible
        + 0.25 * n_cost
        + 0.15 * n_routes
        + 0.05 * n_complexity
    )

    return df.sort_values(
        by=["feasible_rate", "avg_cost", "avg_num_routes", "complexity", "mo_score"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def generate_mock_expression_variants(seed_expr: str, n: int = 8) -> List[str]:
    base = [
        "dist_matrix[current][c]",
        "instance['demands'][c]",
        "dist_matrix[c][instance['depot']]",
        "remaining",
    ]
    variants = [
        f"{base[0]} - 0.2 * ({base[2]} + {base[1]})",
        f"{base[0]} - 0.4 * {base[1]}",
        f"{base[0]} - 0.2 * {base[2]}",
        f"{base[0]} + 0.2 * {base[2]}",
        f"{base[0]} - 0.3 * {base[1]} + 0.1 * {base[3]}",
        f"{base[0]} - 0.2 * ({base[1]} + {base[3]})",
        f"{base[0]} - 0.15 * {base[1]} - 0.05 * {base[3]}",
        f"{base[0]} + {base[1]}",
    ]
    out = [seed_expr] + variants
    return out[: max(1, n)]


def generate_mock_candidates_from_top_expressions(top_expressions: List[str], variants_per_expr=8) -> List[str]:
    all_expr = []
    for e in top_expressions:
        all_expr.extend(generate_mock_expression_variants(e, n=variants_per_expr))
    return dedup_expressions(all_expr)


def generate_candidates_with_llm(*args, **kwargs):
    # Not used in this runner (generation_mode='mock').
    return []


def load_base_instance(json_path: Path) -> Dict:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    depot_id = int(data["depot"]["id"])
    demands = [0] * int(data["dimension"])
    for c in data["customers"]:
        demands[int(c["id"]) - 1] = int(c["demand"])
    demands[depot_id - 1] = int(data["depot"]["demand"])
    instance = {
        "name": data["instance_id"],
        "depot": depot_id - 1,
        "demands": demands,
        "capacity": int(data["vehicle_capacity"]),
        "num_nodes": int(data["dimension"]),
        "distance_matrix": data["distance_matrix"],
        "raw": data,
    }
    return instance


def load_multiple_base_instances(base_dir: str, limit=None) -> List[Dict]:
    files = sorted(Path(base_dir).glob("*.base.json"))
    if limit is not None:
        files = files[:limit]
    return [load_base_instance(p) for p in files]


def main():
    start_time = time.time()
    root = Path(__file__).resolve().parents[1]
    base_dir = root / "02_processed_data" / "classic" / "base"
    output_root = "04_experiment_outputs/formal_benchmark"
    seeds = [42, 52, 62, 72]
    num_rounds = 5
    variants_per_expr = 8
    top_k_per_round = 5
    instances = load_multiple_base_instances(str(base_dir), limit=None)
    # Final-project benchmark scale: more CVRPLib instances, more seeds, and longer search.
    formal_instances = stratified_sample_instances(instances, per_bucket=15, seed=42)

    print("[start] classic formal benchmark", flush=True)
    print(f"  output_root={output_root}", flush=True)
    print(f"  loaded_instances={len(instances)}", flush=True)
    print(f"  sampled_instances={len(formal_instances)}", flush=True)
    print(f"  generation_mode=mock, seeds={seeds}", flush=True)
    print(
        f"  num_rounds={num_rounds}, variants_per_expr={variants_per_expr}, top_k_per_round={top_k_per_round}",
        flush=True,
    )
    print(f"  ortools_solver={ORTOOLS_SOLVER_NAME}", flush=True)

    seed_expressions = [
        "dist_matrix[current][c]",
        "dist_matrix[current][c] - 2 * instance['demands'][c]",
        "dist_matrix[current][c] + 0.3 * dist_matrix[c][instance['depot']]",
        "dist_matrix[current][c] - instance['demands'][c]",
        "dist_matrix[current][c] + instance['demands'][c]",
    ]

    result_paths = run_formal_experiments(
        instances=formal_instances,
        seed_expressions=seed_expressions,
        evaluate_named_solver_on_instances=evaluate_named_solver_on_instances,
        summarize_expression_results=summarize_expression_results,
        nearest_neighbor_v2=nearest_neighbor_v2,
        greedy_cvrp_solver=greedy_cvrp_solver,
        ortools_cvrp_solver=ortools_cvrp_solver,
        ortools_solver_name=ORTOOLS_SOLVER_NAME,
        evaluate_expression_list_on_instances=evaluate_expression_list_on_instances,
        dedup_expressions=dedup_expressions,
        filter_expressions_by_complexity=filter_expressions_by_complexity,
        expression_complexity=expression_complexity,
        add_novelty_columns=add_novelty_columns,
        sort_expression_summary=sort_expression_summary,
        update_archive_signatures=update_archive_signatures,
        generate_mock_candidates_from_top_expressions=generate_mock_candidates_from_top_expressions,
        generate_candidates_with_llm=generate_candidates_with_llm,
        output_root=output_root,
        run_prefix="base_cvrp",
        generation_mode="mock",
        llm_client=None,
        llm_model_name="gpt-5-nano",
        llm_temperature=0.4,
        num_rounds=num_rounds,
        variants_per_expr=variants_per_expr,
        top_k_per_round=top_k_per_round,
        seeds=seeds,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"[done] classic formal benchmark finished in {elapsed:.2f}s", flush=True)
    print(f"[output] root={result_paths['output_root']}", flush=True)
    print(f"[output] ablation_seed_summary={result_paths['ablation_seed_summary']}", flush=True)
    print(f"[output] ablation_aggregate_summary={result_paths['ablation_aggregate_summary']}", flush=True)


if __name__ == "__main__":
    main()
