import json
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT_DIR / "03_core_algorithm" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from fresh_experiment_utils import (
    ORTOOLS_SOLVER_NAME,
    TOKEN_LOG,
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
import matplotlib.pyplot as plt
import pandas as pd

from benchmark_experiment_workflow import evaluate_baselines_table, search_outer_loop_ablation, set_global_seed, stratified_sample_instances
from benchmark_export_plot_utils import export_and_plot


def _best_row(summary_df: pd.DataFrame) -> dict:
    if summary_df is None or summary_df.empty:
        return {
            "best_avg_cost": float("nan"),
            "best_feasible_rate": float("nan"),
            "best_avg_runtime_sec": float("nan"),
            "best_avg_num_routes": float("nan"),
        }
    row = summary_df.sort_values(
        by=["feasible_rate", "avg_objective", "avg_num_routes", "complexity"],
        ascending=[False, True, True, True],
    ).iloc[0]
    return {
        "best_avg_cost": float(row["avg_objective"]),
        "best_feasible_rate": float(row["feasible_rate"]),
        "best_avg_runtime_sec": float(row.get("avg_runtime_sec", float("nan"))),
        "best_avg_num_routes": float(row.get("avg_num_routes", float("nan"))),
        "best_expression": row.get("expression", ""),
    }


def main() -> None:
    start_time = time.time()
    out_root = ROOT_DIR / "04_experiment_outputs" / "fresh_llm_vs_mock_small"
    out_root.mkdir(parents=True, exist_ok=True)

    fresh_dir = ROOT_DIR / "02_processed_data" / "fresh" / "fresh"
    instances = load_multiple_fresh_instances(str(fresh_dir), limit=None)
    # Historical script name kept for compatibility; configuration is now report-ready.
    sampled = stratified_sample_instances(instances, per_bucket=10, seed=42)

    seed_expressions = [
        "travel_to_c",
        "travel_to_c + 5 * est_spoil",
        "travel_to_c + 2 * est_lateness",
        "travel_to_c + 20 * est_spoil + 2 * est_lateness",
        "travel_to_c + 20 * est_spoil + 2 * est_lateness - 0.5 * instance['demands'][c]",
    ]

    baseline_detail, baseline_summary = evaluate_baselines_table(
        instances=sampled,
        evaluate_named_solver_on_instances=evaluate_named_solver_on_instances_fresh,
        summarize_expression_results=summarize_expression_fresh,
        nearest_neighbor_v2=nearest_neighbor_fresh,
        greedy_cvrp_solver=greedy_fresh_solver,
        ortools_cvrp_solver=ortools_fresh_solver,
        ortools_solver_name=ORTOOLS_SOLVER_NAME,
    )
    baseline_detail.to_csv(out_root / "baseline_detail.csv", index=False, encoding="utf-8")
    baseline_summary.to_csv(out_root / "baseline_summary.csv", index=False, encoding="utf-8")

    modes = ["mock", "llm"]
    seeds = [42, 52, 62]
    num_rounds = 4
    variants_per_expr = 6
    top_k = 5
    max_complexity = 90
    require_novel = False
    enable_dedup = True

    client = make_openai_client()
    if client is None:
        print("[warn] Missing API key, skip realtime fresh LLM mode and run mock-only comparison.", flush=True)
        modes = ["mock"]

    print("[start] fresh mock vs llm benchmark", flush=True)
    print(f"  output_root={out_root}", flush=True)
    print(f"  loaded_instances={len(instances)}", flush=True)
    print(f"  sampled_instances={len(sampled)}", flush=True)
    print(f"  modes={modes}, seeds={seeds}", flush=True)
    print(
        f"  num_rounds={num_rounds}, variants_per_expr={variants_per_expr}, top_k_per_round={top_k}",
        flush=True,
    )
    print(
        f"  max_complexity={max_complexity}, require_novel={require_novel}, enable_dedup={enable_dedup}",
        flush=True,
    )
    print(f"  llm_enabled={client is not None}", flush=True)

    rows = []
    for mode in modes:
        for seed in seeds:
            print(f"[start] mode={mode}, seed={seed}", flush=True)
            set_global_seed(seed)
            run_name = f"{mode}_seed{seed}"
            before_calls = len(TOKEN_LOG)
            result = search_outer_loop_ablation(
                instances=sampled,
                seed_expressions=seed_expressions,
                num_rounds=num_rounds,
                variants_per_expr=variants_per_expr,
                top_k_per_round=top_k,
                generation_mode=mode,
                verbose=True,
                enable_dedup=enable_dedup,
                max_complexity=max_complexity,
                require_novel=require_novel,
                llm_client=client,
                llm_model_name="gpt-5-nano",
                llm_temperature=0.4,
                evaluate_expression_list_on_instances=evaluate_expression_list_on_instances_fresh,
                dedup_expressions=dedup_expressions,
                filter_expressions_by_complexity=filter_expressions_by_complexity,
                summarize_expression_results=summarize_expression_fresh,
                expression_complexity=expression_complexity,
                add_novelty_columns=add_novelty_columns,
                sort_expression_summary=sort_expression_summary,
                update_archive_signatures=update_archive_signatures,
                generate_mock_candidates_from_top_expressions=generate_mock_candidates_from_top_expressions,
                generate_candidates_with_llm=generate_candidates_with_llm,
            )

            export_and_plot(
                summary_full=baseline_summary,
                all_detail_outer=result["all_detail_df"],
                all_summary_outer=result["all_round_summary_df"],
                all_top_outer=result["all_round_top_df"],
                output_root="04_experiment_outputs/fresh_llm_vs_mock_small/runs",
                run_name=run_name,
                base_dir="02_processed_data/fresh/fresh",
            )

            best = _best_row(result["all_round_summary_df"])
            llm_calls = TOKEN_LOG[before_calls:]
            total_tokens = int(sum((r.get("total_tokens") or 0) for r in llm_calls if isinstance(r.get("total_tokens"), int)))
            rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "num_instances": len(sampled),
                    "num_rounds": num_rounds,
                    "variants_per_expr": variants_per_expr,
                    "top_k_per_round": top_k,
                    "rounds_with_summary": int(result["all_round_summary_df"]["round_idx"].nunique())
                    if not result["all_round_summary_df"].empty
                    else 0,
                    "summary_rows": len(result["all_round_summary_df"]),
                    "llm_calls": len(llm_calls) if mode == "llm" else 0,
                    "llm_total_tokens": total_tokens if mode == "llm" else 0,
                    **best,
                }
            )
            print(
                f"[done] mode={mode}, seed={seed}, best_avg_cost={best['best_avg_cost']:.2f}, "
                f"best_feasible_rate={best['best_feasible_rate']:.3f}, "
                f"llm_calls={len(llm_calls) if mode == 'llm' else 0}, "
                f"llm_total_tokens={total_tokens if mode == 'llm' else 0}",
                flush=True,
            )

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(out_root / "mock_vs_llm_seed_summary.csv", index=False, encoding="utf-8")
    agg = (
        compare_df.groupby("mode", as_index=False)
        .agg(
            best_avg_cost_mean=("best_avg_cost", "mean"),
            best_avg_cost_std=("best_avg_cost", "std"),
            best_feasible_rate_mean=("best_feasible_rate", "mean"),
            best_avg_runtime_sec_mean=("best_avg_runtime_sec", "mean"),
            llm_total_tokens=("llm_total_tokens", "sum"),
            seeds=("seed", "count"),
        )
        .sort_values("best_avg_cost_mean")
    )
    agg.to_csv(out_root / "mock_vs_llm_aggregate_summary.csv", index=False, encoding="utf-8")

    token_df = pd.DataFrame(TOKEN_LOG)
    token_summary = {
        "calls": int(len(token_df)),
        "prompt_tokens_sum": int(token_df["prompt_tokens"].fillna(0).sum()) if not token_df.empty else 0,
        "completion_tokens_sum": int(token_df["completion_tokens"].fillna(0).sum()) if not token_df.empty else 0,
        "total_tokens_sum": int(token_df["total_tokens"].fillna(0).sum()) if not token_df.empty else 0,
    }
    (out_root / "llm_token_summary.json").write_text(json.dumps(token_summary, indent=2), encoding="utf-8")
    if not token_df.empty:
        token_df.to_csv(out_root / "llm_call_log.csv", index=False, encoding="utf-8")

    mode_label = "Fresh Mock vs LLM" if len(modes) > 1 else "Fresh Mock only"
    plt.figure(figsize=(7, 4))
    plt.bar(agg["mode"], agg["best_avg_cost_mean"], yerr=agg["best_avg_cost_std"].fillna(0.0))
    plt.title(f"{mode_label}: best_avg_objective")
    plt.ylabel("best_avg_objective")
    plt.tight_layout()
    plt.savefig(out_root / "mock_vs_llm_objective_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(agg["mode"], agg["best_feasible_rate_mean"])
    plt.title(f"{mode_label}: feasible_rate")
    plt.ylabel("best_feasible_rate")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_root / "mock_vs_llm_feasible_bar.png", dpi=160)
    plt.close()

    meta = {
        "num_instances_total": len(instances),
        "num_instances_sampled": len(sampled),
        "sampled_instances": [inst["name"] for inst in sampled],
        "modes": modes,
        "seeds": seeds,
        "num_rounds": num_rounds,
        "variants_per_expr": variants_per_expr,
        "top_k_per_round": top_k,
        "max_complexity": max_complexity,
        "require_novel": require_novel,
        "enable_dedup": enable_dedup,
        "llm_enabled": client is not None,
    }
    (out_root / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"[done] fresh mock vs llm benchmark finished in {elapsed:.2f}s", flush=True)
    print(f"[output] root={out_root}", flush=True)
    print(agg.to_string(index=False))
    print(f"Token summary: {token_summary}")


if __name__ == "__main__":
    main()
