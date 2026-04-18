import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT_DIR / "03_core_algorithm" / "modules"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from run_formal_benchmark import (  # noqa: E402
    ORTOOLS_SOLVER_NAME,
    add_novelty_columns,
    dedup_expressions,
    evaluate_expression_list_on_instances,
    evaluate_named_solver_on_instances,
    expression_complexity,
    filter_expressions_by_complexity,
    generate_mock_candidates_from_top_expressions,
    greedy_cvrp_solver,
    load_multiple_base_instances,
    nearest_neighbor_v2,
    ortools_cvrp_solver,
    sort_expression_summary,
    summarize_expression_results,
    update_archive_signatures,
)

import numpy as np
import openai
import pandas as pd

from benchmark_experiment_workflow import (  # noqa: E402
    evaluate_baselines_table,
    search_outer_loop_ablation,
    set_global_seed,
    stratified_sample_instances,
)
from benchmark_export_plot_utils import export_and_plot  # noqa: E402


TOKEN_LOG: List[Dict] = []


def _build_prompt(top_expressions: List[str], max_total: int = 16) -> str:
    seed_block = "\n".join([f"- {expr}" for expr in top_expressions])
    return f"""
Generate new candidate Python score expressions for a CVRP heuristic.
Lower score is better.

Available variables only:
- dist_matrix[current][c]
- dist_matrix[c][instance['depot']]
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
    banned = [
        "__",
        "import",
        "exec",
        "eval",
        "open(",
        "os.",
        "sys.",
        "for ",
        "while ",
        "if ",
        "lambda",
        "def ",
        "class ",
    ]
    low = expr.lower()
    return not any(b in low for b in banned)


def _filter_valid_expressions(expressions: List[str], max_complexity: Optional[int]) -> List[str]:
    out = []
    for expr in expressions:
        if not isinstance(expr, str):
            continue
        expr = _normalize_expression(expr)
        if not expr:
            continue
        if not _is_safe_expression(expr):
            continue
        if max_complexity is not None and expression_complexity(expr) > max_complexity:
            continue
        out.append(expr)
    return dedup_expressions(out)


def _is_placeholder_api_key(value: str) -> bool:
    low = value.strip().lower()
    if not low:
        return True
    return low in {
        "please use your api",
        "your_api_key_here",
        "sk-your-key",
        "replace_me",
    }


def _extract_key_from_notebook(nb_path: Path) -> str:
    if not nb_path.exists():
        return ""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            for pattern in [
                r'API_KEY\s*=\s*"([^"]+)"',
                r"API_KEY\s*=\s*'([^']+)'",
                r'OPENAI_API_KEY\s*=\s*"([^"]+)"',
                r"OPENAI_API_KEY\s*=\s*'([^']+)'",
            ]:
                m = re.search(pattern, src)
                if m:
                    api_key = m.group(1).strip()
                    if not _is_placeholder_api_key(api_key):
                        return api_key
    except Exception:
        pass
    return ""


def _extract_key_from_notebooks() -> str:
    candidates = [
        ROOT_DIR / "03_core_algorithm" / "notebooks" / "benchmark_cvrp_clean (1).ipynb",
        ROOT_DIR / "03_core_algorithm" / "notebooks" / "benchmark_cvrp_advanced (1).ipynb",
        ROOT_DIR / "benchmark_cvrp_advanced (1).ipynb",
    ]
    for nb_path in candidates:
        api_key = _extract_key_from_notebook(nb_path)
        if api_key:
            return api_key
    return ""


def _make_openai_client() -> Optional[openai.OpenAI]:
    api_key = os.getenv("CVRP_OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = _extract_key_from_notebooks()
    if _is_placeholder_api_key(api_key):
        api_key = ""
    if not api_key:
        return None
    host = os.getenv("CVRP_OPENAI_HOST", "https://api.bltcy.ai").rstrip("/")
    return openai.OpenAI(base_url=f"{host}/v1", api_key=api_key, timeout=45)


def generate_candidates_with_llm(
    client,
    top_expressions: List[str],
    n_per_expr: int = 4,
    model_name: str = "gpt-5-nano",
    temperature: float = 0.4,
    max_complexity: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    max_total = min(20, max(8, n_per_expr * max(1, len(top_expressions))))
    prompt = _build_prompt(top_expressions, max_total=max_total)
    retries = 2
    raw_text = ""
    usage = {}
    for i in range(retries):
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
            if i == retries - 1:
                raw_text = ""
            time.sleep(1.0 * (i + 1))

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
            f"[llm] raw={len(expressions) if isinstance(expressions, list) else 0}, "
            f"valid={len(valid)}, tokens={usage.get('total_tokens')}"
        , flush=True)
    return valid


def _best_row(summary_df: pd.DataFrame) -> Dict:
    if summary_df is None or summary_df.empty:
        return {
            "best_avg_cost": np.nan,
            "best_feasible_rate": np.nan,
            "best_avg_runtime_sec": np.nan,
            "best_avg_num_routes": np.nan,
        }
    row = summary_df.sort_values(
        by=["feasible_rate", "avg_cost", "avg_num_routes", "complexity"],
        ascending=[False, True, True, True],
    ).iloc[0]
    return {
        "best_avg_cost": float(row["avg_cost"]),
        "best_feasible_rate": float(row["feasible_rate"]),
        "best_avg_runtime_sec": float(row.get("avg_runtime_sec", np.nan)),
        "best_avg_num_routes": float(row.get("avg_num_routes", np.nan)),
        "best_expression": row.get("expression", ""),
    }


def main() -> None:
    start_time = time.time()
    out_root = ROOT_DIR / "04_experiment_outputs" / "llm_vs_mock_small"
    out_root.mkdir(parents=True, exist_ok=True)

    base_dir = ROOT_DIR / "02_processed_data" / "classic" / "base"
    instances = load_multiple_base_instances(str(base_dir), limit=None)
    # Historical script name kept for compatibility; configuration is now report-ready.
    sampled = stratified_sample_instances(instances, per_bucket=10, seed=42)

    seed_expressions = [
        "dist_matrix[current][c]",
        "dist_matrix[current][c] - 2 * instance['demands'][c]",
        "dist_matrix[current][c] + 0.3 * dist_matrix[c][instance['depot']]",
        "dist_matrix[current][c] - instance['demands'][c]",
        "dist_matrix[current][c] + instance['demands'][c]",
    ]

    baseline_detail, baseline_summary = evaluate_baselines_table(
        instances=sampled,
        evaluate_named_solver_on_instances=evaluate_named_solver_on_instances,
        summarize_expression_results=summarize_expression_results,
        nearest_neighbor_v2=nearest_neighbor_v2,
        greedy_cvrp_solver=greedy_cvrp_solver,
        ortools_cvrp_solver=ortools_cvrp_solver,
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

    client = _make_openai_client()
    if client is None:
        print("[warn] Missing API key, skip realtime LLM mode and run mock-only comparison.", flush=True)
        modes = ["mock"]

    print("[start] classic mock vs llm benchmark", flush=True)
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

            export_and_plot(
                summary_full=baseline_summary,
                all_detail_outer=result["all_detail_df"],
                all_summary_outer=result["all_round_summary_df"],
                all_top_outer=result["all_round_top_df"],
                output_root="04_experiment_outputs/llm_vs_mock_small/runs",
                run_name=run_name,
                base_dir="02_processed_data/classic/base",
            )

            best = _best_row(result["all_round_summary_df"])
            llm_calls = TOKEN_LOG[before_calls:]
            total_tokens = int(
                sum((r.get("total_tokens") or 0) for r in llm_calls if isinstance(r.get("total_tokens"), int))
            )
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
    if not token_df.empty:
        token_df.to_csv(out_root / "llm_call_log.csv", index=False, encoding="utf-8")
        token_summary = {
            "calls": int(len(token_df)),
            "prompt_tokens_sum": int(token_df["prompt_tokens"].fillna(0).sum()),
            "completion_tokens_sum": int(token_df["completion_tokens"].fillna(0).sum()),
            "total_tokens_sum": int(token_df["total_tokens"].fillna(0).sum()),
        }
    else:
        token_summary = {"calls": 0, "prompt_tokens_sum": 0, "completion_tokens_sum": 0, "total_tokens_sum": 0}
    (out_root / "llm_token_summary.json").write_text(json.dumps(token_summary, indent=2), encoding="utf-8")

    # quick comparison plots
    import matplotlib.pyplot as plt

    mode_label = "Mock vs LLM benchmark" if len(modes) > 1 else "Mock-only benchmark"

    plt.figure(figsize=(7, 4))
    plt.bar(agg["mode"], agg["best_avg_cost_mean"], yerr=agg["best_avg_cost_std"].fillna(0.0))
    plt.title(f"{mode_label}: best_avg_cost")
    plt.ylabel("best_avg_cost")
    plt.tight_layout()
    plt.savefig(out_root / "mock_vs_llm_cost_bar.png", dpi=160)
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
    print(f"[done] classic mock vs llm benchmark finished in {elapsed:.2f}s", flush=True)
    print(f"[output] root={out_root}", flush=True)
    print(agg.to_string(index=False))
    print(f"Token summary: {token_summary}")


if __name__ == "__main__":
    main()
