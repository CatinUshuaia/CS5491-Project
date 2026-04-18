import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from benchmark_export_plot_utils import export_and_plot


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _project_root() / p


@dataclass
class AblationConfig:
    name: str
    enable_dedup: bool
    max_complexity: Optional[int]
    require_novel: bool


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _bucket_key(instance: Dict) -> str:
    n = int(instance.get("num_nodes", 0))
    if n <= 50:
        return "small"
    if n <= 100:
        return "medium"
    return "large"


def stratified_sample_instances(instances: List[Dict], per_bucket: int = 10, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    buckets = {"small": [], "medium": [], "large": []}
    for inst in instances:
        buckets[_bucket_key(inst)].append(inst)

    sampled: List[Dict] = []
    for key in ["small", "medium", "large"]:
        group = buckets[key]
        if len(group) <= per_bucket:
            sampled.extend(group)
        else:
            sampled.extend(rng.sample(group, per_bucket))
    return sampled


def evaluate_baselines_table(
    instances: List[Dict],
    evaluate_named_solver_on_instances: Callable,
    summarize_expression_results: Callable,
    nearest_neighbor_v2: Callable,
    greedy_cvrp_solver: Callable,
    ortools_cvrp_solver: Callable,
    ortools_solver_name: str = "ortools",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_nn = evaluate_named_solver_on_instances(instances, "nearest_neighbor", nearest_neighbor_v2)
    df_greedy = evaluate_named_solver_on_instances(instances, "greedy", greedy_cvrp_solver)
    df_ortools = evaluate_named_solver_on_instances(instances, ortools_solver_name, ortools_cvrp_solver)
    detail = pd.concat([df_nn, df_greedy, df_ortools], ignore_index=True)
    summary = summarize_expression_results(detail)
    return detail, summary


def _run_one_search_round_ablation(
    instances: List[Dict],
    candidate_expressions: List[str],
    archive_signatures: Optional[set],
    top_k: int,
    *,
    enable_dedup: bool,
    max_complexity: Optional[int],
    require_novel: bool,
    evaluate_expression_list_on_instances: Callable,
    dedup_expressions: Callable,
    filter_expressions_by_complexity: Callable,
    summarize_expression_results: Callable,
    expression_complexity: Callable,
    add_novelty_columns: Callable,
    sort_expression_summary: Callable,
    update_archive_signatures: Callable,
) -> Dict:
    if archive_signatures is None:
        archive_signatures = set()

    pool = list(candidate_expressions)
    if enable_dedup:
        probe_instances = list(instances[: min(5, len(instances))])
        pool = dedup_expressions(
            pool,
            probe_instances=probe_instances,
            eval_fn=evaluate_expression_list_on_instances,
            use_behavioral=True,
        )
    if max_complexity is not None:
        pool = filter_expressions_by_complexity(pool, max_complexity=max_complexity)

    if len(pool) == 0:
        return {
            "detail_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "top_df": pd.DataFrame(),
            "archive_signatures": archive_signatures,
        }

    detail_df = evaluate_expression_list_on_instances(instances, pool)
    if detail_df.empty:
        return {
            "detail_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "top_df": pd.DataFrame(),
            "archive_signatures": archive_signatures,
        }

    summary_df = summarize_expression_results(detail_df)
    if "complexity" not in summary_df.columns:
        summary_df["complexity"] = summary_df["expression"].apply(expression_complexity)
    summary_df = add_novelty_columns(summary_df, archive_signatures=archive_signatures)

    if require_novel:
        summary_df = summary_df[summary_df["is_novel"]].copy()
    if summary_df.empty:
        return {
            "detail_df": detail_df,
            "summary_df": pd.DataFrame(),
            "top_df": pd.DataFrame(),
            "archive_signatures": archive_signatures,
        }

    summary_df = sort_expression_summary(summary_df)
    top_df = summary_df.head(top_k).reset_index(drop=True)
    new_archive = update_archive_signatures(top_df, archive_signatures=archive_signatures, only_novel=False)
    return {
        "detail_df": detail_df,
        "summary_df": summary_df,
        "top_df": top_df,
        "archive_signatures": new_archive,
    }


def search_outer_loop_ablation(
    instances: List[Dict],
    seed_expressions: List[str],
    num_rounds: int,
    variants_per_expr: int,
    top_k_per_round: int,
    generation_mode: str,
    verbose: bool,
    *,
    enable_dedup: bool,
    max_complexity: Optional[int],
    require_novel: bool,
    llm_client,
    llm_model_name: str,
    llm_temperature: float,
    evaluate_expression_list_on_instances: Callable,
    dedup_expressions: Callable,
    filter_expressions_by_complexity: Callable,
    summarize_expression_results: Callable,
    expression_complexity: Callable,
    add_novelty_columns: Callable,
    sort_expression_summary: Callable,
    update_archive_signatures: Callable,
    generate_mock_candidates_from_top_expressions: Callable,
    generate_candidates_with_llm: Callable,
) -> Dict:
    archive_signatures = set()
    round_summaries = []
    round_tops = []
    all_detail_dfs = []
    current_pool = list(seed_expressions)

    for round_idx in range(num_rounds):
        result = _run_one_search_round_ablation(
            instances=instances,
            candidate_expressions=current_pool,
            archive_signatures=archive_signatures,
            top_k=top_k_per_round,
            enable_dedup=enable_dedup,
            max_complexity=max_complexity,
            require_novel=require_novel,
            evaluate_expression_list_on_instances=evaluate_expression_list_on_instances,
            dedup_expressions=dedup_expressions,
            filter_expressions_by_complexity=filter_expressions_by_complexity,
            summarize_expression_results=summarize_expression_results,
            expression_complexity=expression_complexity,
            add_novelty_columns=add_novelty_columns,
            sort_expression_summary=sort_expression_summary,
            update_archive_signatures=update_archive_signatures,
        )

        detail_df = result["detail_df"]
        summary_df = result["summary_df"]
        top_df = result["top_df"]
        archive_signatures = result["archive_signatures"]

        if not detail_df.empty:
            detail_df = detail_df.copy()
            detail_df["round_idx"] = round_idx
            all_detail_dfs.append(detail_df)
        if not summary_df.empty:
            summary_df = summary_df.copy()
            summary_df["round_idx"] = round_idx
            round_summaries.append(summary_df)
        if not top_df.empty:
            top_df = top_df.copy()
            top_df["round_idx"] = round_idx
            round_tops.append(top_df)

        if verbose:
            print(
                f"[round {round_idx}] pool={len(current_pool)}, evaluated={len(summary_df)}, top={len(top_df)}, "
                f"dedup={enable_dedup}, max_complexity={max_complexity}, require_novel={require_novel}"
            )

        if top_df.empty:
            break
        next_seed_expressions = top_df["expression"].tolist()

        if generation_mode == "mock":
            current_pool = generate_mock_candidates_from_top_expressions(
                next_seed_expressions, variants_per_expr=variants_per_expr
            )
        elif generation_mode == "llm":
            if llm_client is None:
                raise ValueError("generation_mode='llm' requires a non-empty llm_client")
            current_pool = generate_candidates_with_llm(
                client=llm_client,
                top_expressions=next_seed_expressions,
                n_per_expr=variants_per_expr,
                model_name=llm_model_name,
                temperature=llm_temperature,
                max_complexity=max_complexity,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported generation_mode: {generation_mode}")

    all_round_summary_df = pd.concat(round_summaries, ignore_index=True) if round_summaries else pd.DataFrame()
    all_round_top_df = pd.concat(round_tops, ignore_index=True) if round_tops else pd.DataFrame()
    all_detail_df = pd.concat(all_detail_dfs, ignore_index=True) if all_detail_dfs else pd.DataFrame()
    return {
        "all_detail_df": all_detail_df,
        "all_round_summary_df": all_round_summary_df,
        "all_round_top_df": all_round_top_df,
        "archive_signatures": archive_signatures,
    }


def _aggregate_ablation_rows(config_name: str, seed: int, all_summary_outer: pd.DataFrame) -> Dict:
    if all_summary_outer is None or all_summary_outer.empty:
        return {
            "config": config_name,
            "seed": seed,
            "best_avg_cost": np.nan,
            "best_feasible_rate": np.nan,
            "best_avg_runtime_sec": np.nan,
            "best_avg_num_routes": np.nan,
        }

    best_row = all_summary_outer.sort_values(
        by=["feasible_rate", "avg_cost", "avg_num_routes", "complexity"],
        ascending=[False, True, True, True],
    ).iloc[0]
    return {
        "config": config_name,
        "seed": seed,
        "best_avg_cost": float(best_row["avg_cost"]),
        "best_feasible_rate": float(best_row["feasible_rate"]),
        "best_avg_runtime_sec": float(best_row.get("avg_runtime_sec", np.nan)),
        "best_avg_num_routes": float(best_row.get("avg_num_routes", np.nan)),
    }


def _plot_ablation_summary(ablation_seed_df: pd.DataFrame, output_dir: Path) -> None:
    if ablation_seed_df.empty:
        return
    summary = (
        ablation_seed_df.groupby("config", as_index=False)
        .agg(
            best_avg_cost_mean=("best_avg_cost", "mean"),
            best_avg_cost_std=("best_avg_cost", "std"),
            best_feasible_rate_mean=("best_feasible_rate", "mean"),
        )
        .sort_values("best_avg_cost_mean")
    )

    plt.figure(figsize=(10, 5))
    plt.bar(summary["config"], summary["best_avg_cost_mean"], yerr=summary["best_avg_cost_std"].fillna(0.0))
    plt.title("Ablation Comparison (mean best_avg_cost ± std)")
    plt.ylabel("best_avg_cost")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_cost_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(summary["config"], summary["best_feasible_rate_mean"])
    plt.title("Ablation Comparison (mean best_feasible_rate)")
    plt.ylabel("best_feasible_rate")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_feasible_bar.png", dpi=160)
    plt.close()


def run_formal_experiments(
    *,
    instances: List[Dict],
    seed_expressions: List[str],
    evaluate_named_solver_on_instances: Callable,
    summarize_expression_results: Callable,
    nearest_neighbor_v2: Callable,
    greedy_cvrp_solver: Callable,
    ortools_cvrp_solver: Callable,
    ortools_solver_name: str = "ortools",
    evaluate_expression_list_on_instances: Callable,
    dedup_expressions: Callable,
    filter_expressions_by_complexity: Callable,
    expression_complexity: Callable,
    add_novelty_columns: Callable,
    sort_expression_summary: Callable,
    update_archive_signatures: Callable,
    generate_mock_candidates_from_top_expressions: Callable,
    generate_candidates_with_llm: Callable,
    output_root: str = "04_experiment_outputs/formal_benchmark",
    run_prefix: str = "formal",
    generation_mode: str = "mock",
    llm_client=None,
    llm_model_name: str = "gpt-5-nano",
    llm_temperature: float = 0.4,
    num_rounds: int = 4,
    variants_per_expr: int = 6,
    top_k_per_round: int = 5,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    if seeds is None:
        seeds = [42, 52, 62]

    root = _resolve_project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    # baseline
    baseline_detail, baseline_summary = evaluate_baselines_table(
        instances=instances,
        evaluate_named_solver_on_instances=evaluate_named_solver_on_instances,
        summarize_expression_results=summarize_expression_results,
        nearest_neighbor_v2=nearest_neighbor_v2,
        greedy_cvrp_solver=greedy_cvrp_solver,
        ortools_cvrp_solver=ortools_cvrp_solver,
        ortools_solver_name=ortools_solver_name,
    )

    ablations = [
        AblationConfig(name="base", enable_dedup=False, max_complexity=None, require_novel=False),
        AblationConfig(name="base_plus_dedup", enable_dedup=True, max_complexity=None, require_novel=False),
        AblationConfig(name="base_plus_complexity", enable_dedup=False, max_complexity=90, require_novel=False),
        AblationConfig(name="base_plus_novelty", enable_dedup=False, max_complexity=None, require_novel=True),
        AblationConfig(name="all_enabled", enable_dedup=True, max_complexity=90, require_novel=True),
    ]

    ablation_rows = []
    for cfg in ablations:
        for seed in seeds:
            if verbose:
                print(
                    f"[start] config={cfg.name}, seed={seed}, mode={generation_mode}, "
                    f"rounds={num_rounds}, variants_per_expr={variants_per_expr}, top_k={top_k_per_round}",
                    flush=True,
                )
            set_global_seed(seed)
            run_name = f"{run_prefix}_{cfg.name}_seed{seed}"
            outer_result = search_outer_loop_ablation(
                instances=instances,
                seed_expressions=seed_expressions,
                num_rounds=num_rounds,
                variants_per_expr=variants_per_expr,
                top_k_per_round=top_k_per_round,
                generation_mode=generation_mode,
                verbose=verbose,
                enable_dedup=cfg.enable_dedup,
                max_complexity=cfg.max_complexity,
                require_novel=cfg.require_novel,
                llm_client=llm_client,
                llm_model_name=llm_model_name,
                llm_temperature=llm_temperature,
                evaluate_expression_list_on_instances=evaluate_expression_list_on_instances,
                dedup_expressions=dedup_expressions,
                filter_expressions_by_complexity=filter_expressions_by_complexity,
                summarize_expression_results=summarize_expression_results,
                expression_complexity=expression_complexity,
                add_novelty_columns=add_novelty_columns,
                sort_expression_summary=sort_expression_summary,
                update_archive_signatures=update_archive_signatures,
                generate_mock_candidates_from_top_expressions=generate_mock_candidates_from_top_expressions,
                generate_candidates_with_llm=generate_candidates_with_llm,
            )

            paths = export_and_plot(
                summary_full=baseline_summary,
                all_detail_outer=outer_result["all_detail_df"],
                all_summary_outer=outer_result["all_round_summary_df"],
                all_top_outer=outer_result["all_round_top_df"],
                output_root=str(root),
                run_name=run_name,
                base_dir="02_processed_data/classic/base",
            )
            if verbose:
                print(f"[saved] {run_name}: {paths['run_dir']}")

            row = _aggregate_ablation_rows(cfg.name, seed, outer_result["all_round_summary_df"])
            ablation_rows.append(row)
            if verbose:
                print(
                    f"[done] config={cfg.name}, seed={seed}, "
                    f"best_avg_cost={row['best_avg_cost']:.2f}, "
                    f"best_feasible_rate={row['best_feasible_rate']:.3f}",
                    flush=True,
                )

    ablation_seed_df = pd.DataFrame(ablation_rows)
    ablation_seed_path = root / "ablation_seed_summary.csv"
    ablation_seed_df.to_csv(ablation_seed_path, index=False, encoding="utf-8")

    ablation_agg_df = (
        ablation_seed_df.groupby("config", as_index=False)
        .agg(
            best_avg_cost_mean=("best_avg_cost", "mean"),
            best_avg_cost_std=("best_avg_cost", "std"),
            best_feasible_rate_mean=("best_feasible_rate", "mean"),
            seeds=("seed", "count"),
        )
        .sort_values("best_avg_cost_mean")
    )
    ablation_agg_path = root / "ablation_aggregate_summary.csv"
    ablation_agg_df.to_csv(ablation_agg_path, index=False, encoding="utf-8")

    _plot_ablation_summary(ablation_seed_df, output_dir=root)

    meta = {
        "num_instances": len(instances),
        "generation_mode": generation_mode,
        "num_rounds": num_rounds,
        "variants_per_expr": variants_per_expr,
        "top_k_per_round": top_k_per_round,
        "seeds": seeds,
        "ablations": [cfg.__dict__ for cfg in ablations],
    }
    (root / "formal_experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "output_root": str(root),
        "ablation_seed_summary": str(ablation_seed_path),
        "ablation_aggregate_summary": str(ablation_agg_path),
    }
