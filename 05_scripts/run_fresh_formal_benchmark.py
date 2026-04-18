from pathlib import Path
import time
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT_DIR / "03_core_algorithm" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from fresh_experiment_utils import (
    ORTOOLS_SOLVER_NAME,
    add_novelty_columns,
    dedup_expressions,
    evaluate_expression_list_on_instances_fresh,
    evaluate_named_solver_on_instances_fresh,
    expression_complexity,
    filter_expressions_by_complexity,
    generate_candidates_with_llm,
    generate_mock_candidates_from_top_expressions,
    greedy_fresh_solver,
    load_multiple_fresh_instances,
    make_openai_client,
    nearest_neighbor_fresh,
    ortools_fresh_solver,
    sort_expression_summary,
    summarize_expression_fresh,
    update_archive_signatures,
)
from benchmark_experiment_workflow import run_formal_experiments, stratified_sample_instances


def main() -> None:
    start_time = time.time()
    fresh_dir = ROOT_DIR / "02_processed_data" / "fresh" / "fresh"
    output_root = "04_experiment_outputs/fresh_formal_benchmark"
    seeds = [42, 52, 62]
    num_rounds = 4
    variants_per_expr = 6
    top_k_per_round = 5
    instances = load_multiple_fresh_instances(str(fresh_dir), limit=None)
    # Final-project benchmark scale: expand fresh evaluation beyond the earlier smoke-size setup.
    formal_instances = stratified_sample_instances(instances, per_bucket=10, seed=42)

    print("[start] fresh formal benchmark", flush=True)
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
        "travel_to_c",
        "travel_to_c + 5 * est_spoil",
        "travel_to_c + 2 * est_lateness",
        "travel_to_c + 20 * est_spoil + 2 * est_lateness",
        "travel_to_c + 20 * est_spoil + 2 * est_lateness - 0.5 * instance['demands'][c]",
    ]

    client = make_openai_client()
    result_paths = run_formal_experiments(
        instances=formal_instances,
        seed_expressions=seed_expressions,
        evaluate_named_solver_on_instances=evaluate_named_solver_on_instances_fresh,
        summarize_expression_results=summarize_expression_fresh,
        nearest_neighbor_v2=nearest_neighbor_fresh,
        greedy_cvrp_solver=greedy_fresh_solver,
        ortools_cvrp_solver=ortools_fresh_solver,
        ortools_solver_name=ORTOOLS_SOLVER_NAME,
        evaluate_expression_list_on_instances=evaluate_expression_list_on_instances_fresh,
        dedup_expressions=dedup_expressions,
        filter_expressions_by_complexity=filter_expressions_by_complexity,
        expression_complexity=expression_complexity,
        add_novelty_columns=add_novelty_columns,
        sort_expression_summary=sort_expression_summary,
        update_archive_signatures=update_archive_signatures,
        generate_mock_candidates_from_top_expressions=generate_mock_candidates_from_top_expressions,
        generate_candidates_with_llm=generate_candidates_with_llm,
        output_root=output_root,
        run_prefix="fresh_cvrp",
        generation_mode="mock",
        llm_client=client,
        llm_model_name="gpt-5-nano",
        llm_temperature=0.4,
        num_rounds=num_rounds,
        variants_per_expr=variants_per_expr,
        top_k_per_round=top_k_per_round,
        seeds=seeds,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"[done] fresh formal benchmark finished in {elapsed:.2f}s", flush=True)
    print(f"[output] root={result_paths['output_root']}", flush=True)
    print(f"[output] ablation_seed_summary={result_paths['ablation_seed_summary']}", flush=True)
    print(f"[output] ablation_aggregate_summary={result_paths['ablation_aggregate_summary']}", flush=True)


if __name__ == "__main__":
    main()
