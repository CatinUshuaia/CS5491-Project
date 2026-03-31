import ast
import math
import numpy as np
import pandas as pd

# ----------------------------
# 1) expression canonicalization
# ----------------------------

class Canonicalizer(ast.NodeTransformer):
    """
    Normalize expression AST to reduce superficial syntax differences.
    Only handles safe arithmetic expressions.
    """

    COMMUTATIVE_OPS = (ast.Add, ast.Mult)

    def visit_BinOp(self, node):
        self.generic_visit(node)

        # x + 0 -> x, x * 1 -> x
        if isinstance(node.op, ast.Add):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right

        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 1:
                return node.right

        # canonical ordering for commutative ops
        if isinstance(node.op, self.COMMUTATIVE_OPS):
            left_dump = ast.dump(node.left)
            right_dump = ast.dump(node.right)
            if right_dump < left_dump:
                node.left, node.right = node.right, node.left

        return node


def canonicalize_expr(expr: str) -> str:
    """
    Convert an arithmetic expression into a normalized string form.
    If parsing fails, fall back to stripped original.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        tree = Canonicalizer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return expr.strip()


# ----------------------------
# 2) behavior fingerprint
# ----------------------------

def make_behavior_fingerprint(
    expr: str,
    probe_instances,
    eval_fn,
    rounding=2
):
    """
    Evaluate expr on a small probe set and produce a lightweight fingerprint.

    eval_fn(expr, instances) should return a detail_df or summary-like result
    containing at least:
      - instance_id
      - feasible
      - cost
      - num_routes
    """
    try:
        detail_df = eval_fn(expr, probe_instances).copy()

        # sort by instance for stable fingerprint
        if "instance_id" in detail_df.columns:
            detail_df = detail_df.sort_values("instance_id").reset_index(drop=True)

        feasible_vec = tuple(detail_df["feasible"].astype(int).tolist())

        if "cost" in detail_df.columns:
            cost_vec = tuple(np.round(detail_df["cost"].astype(float).values, rounding))
        else:
            cost_vec = ()

        if "num_routes" in detail_df.columns:
            routes_vec = tuple(np.round(detail_df["num_routes"].astype(float).values, rounding))
        else:
            routes_vec = ()

        # lightweight aggregate stats
        feasible_rate = round(float(detail_df["feasible"].mean()), 3)
        avg_cost = round(float(detail_df["cost"].mean()), rounding) if "cost" in detail_df.columns else math.inf
        avg_num_routes = round(float(detail_df["num_routes"].mean()), rounding) if "num_routes" in detail_df.columns else math.inf

        fingerprint = (
            feasible_vec,
            cost_vec,
            routes_vec,
            feasible_rate,
            avg_cost,
            avg_num_routes
        )

        return fingerprint

    except Exception:
        # invalid expr / runtime failure -> assign unique bad fingerprint
        return ("ERROR", expr)


# ----------------------------
# 3) advanced dedup
# ----------------------------

def dedup_candidates_advanced(
    candidate_expressions,
    probe_instances,
    eval_fn,
    use_behavioral=True
):
    """
    Deduplicate candidates by:
      (1) canonicalized syntax
      (2) optional lightweight behavioral fingerprint on probe set
    """
    kept = []
    seen_canonical = set()
    seen_behavior = set()

    for expr in candidate_expressions:
        canon = canonicalize_expr(expr)

        # syntax-level dedup
        if canon in seen_canonical:
            continue

        # behavior-level dedup
        if use_behavioral:
            fp = make_behavior_fingerprint(canon, probe_instances, eval_fn)
            if fp in seen_behavior:
                continue
            seen_behavior.add(fp)

        seen_canonical.add(canon)
        kept.append(canon)

    return kept