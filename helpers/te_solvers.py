# helpers/te_solvers.py

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pulp as pl
import cvxpy as cp


Demand = Tuple[int, str, str, float]  # (k_id, src, dst, volume)


def build_k_shortest_paths(
    G: nx.DiGraph,
    demands: List[Demand],
    k: int = 5,
    weight: Optional[str] = None,
) -> Dict[int, List[List[str]]]:
    """
    Build up to k shortest simple paths for each demand.

    Args:
        G: Directed graph with nodes matching the src/dst in demands.
        demands: List of (k_id, src, dst, volume).
        k: Maximum number of paths per demand.
        weight: Edge attribute to use as weight (or None for unweighted).

    Returns:
        Dict mapping k_id -> list of paths, where each path is a list of nodes.
    """
    from networkx import shortest_simple_paths

    paths: Dict[int, List[List[str]]] = {}
    for k_id, src, dst, vol in demands:
        try:
            gen = shortest_simple_paths(G, src, dst, weight=weight)
            paths[k_id] = [p for _, p in zip(range(k), gen)]
        except nx.NetworkXNoPath:
            # No path for this demand
            paths[k_id] = []

    return paths


def solve_milp_max_throughput(
    G: nx.DiGraph,
    demands: List[Demand],
    paths: Dict[int, List[List[str]]],
):
    """
    MILP baseline: maximize total served throughput.

    Variables:
        y_{k,p} in [0,1] : fraction of demand k routed on path p.

    Constraints:
        - For each demand k: sum_p y_{k,p} <= 1
        - For each edge e: sum_k sum_{p∋e} (d_k * y_{k,p}) <= capacity_e

    Objective:
        max sum_k d_k * sum_p y_{k,p}

    Args:
        G: DiGraph with edge attr 'capacity' in same units as demand volumes.
        demands: List of (k_id, src, dst, volume).
        paths: Dict k_id -> list of paths (each path is a list of nodes).

    Returns:
        x: dict k_id -> served fraction (0..1)
        y: dict (k_id, p_idx) -> PuLP variable y_{k,p}
        prob: PuLP problem object (can inspect status, objective, etc.)
    """
    # Precompute which (k, p) use each edge
    edge_to_kp = defaultdict(list)  # (u,v) -> list of (k_id, p_idx)
    for k_id, s, t, vol in demands:
        for p_idx, path in enumerate(paths[k_id]):
            edges_on_path = list(zip(path[:-1], path[1:]))
            for e in edges_on_path:
                edge_to_kp[e].append((k_id, p_idx))

    # Precompute demand volumes by k_id for easy lookup
    demand_vol = {k_id: vol for (k_id, s, t, vol) in demands}

    prob = pl.LpProblem("Abilene_TE_MILP", pl.LpMaximize)

    # Variables: y_{k,p} in [0,1]
    y: Dict[Tuple[int, int], pl.LpVariable] = {}
    for k_id, s, t, vol in demands:
        for p_idx, path in enumerate(paths[k_id]):
            y[(k_id, p_idx)] = pl.LpVariable(
                f"y_{k_id}_{p_idx}",
                lowBound=0.0,
                upBound=1.0,
            )

    # Objective: maximize total served throughput
    prob += pl.lpSum(
        demand_vol[k_id] * y[(k_id, p_idx)]
        for (k_id, s, t, vol) in demands
        for p_idx, path in enumerate(paths[k_id])
    )

    # Constraint 1: sum_p y_{k,p} <= 1  (per demand)
    for k_id, s, t, vol in demands:
        prob += (
            pl.lpSum(
                y[(k_id, p_idx)] for p_idx, _ in enumerate(paths[k_id])
            )
            <= 1
        ), f"demand_frac_{k_id}"

    # Constraint 2: capacity on each edge
    for (u, v, data) in G.edges(data=True):
        cap = data["capacity"]
        prob += (
            pl.lpSum(
                demand_vol[k_id] * y[(k_id, p_idx)]
                for (k_id, p_idx) in edge_to_kp[(u, v)]
            )
            <= cap
        ), f"cap_{u}_{v}"

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    # Extract solution: fraction per demand x_k = sum_p y_{k,p}
    x: Dict[int, float] = {}
    for k_id, s, t, vol in demands:
        total = 0.0
        for p_idx, _ in enumerate(paths[k_id]):
            v = y[(k_id, p_idx)].value()
            # treat None as 0.0
            if v is not None:
                total += float(v)
        x[k_id] = total

    return x, y, prob


def solve_pf_admm_like(
    G: nx.DiGraph,
    demands: List[Demand],
    paths: Dict[int, List[List[str]]],
    epsilon: float = 1e-6,
):
    """
    Proportional-fair (PF) convex baseline solved with CVXPY + SCS (ADMM-type).

    Variables:
        y_{k,p} >= 0 : flow of demand k on path p (absolute units).
        f_k = sum_p y_{k,p} : total flow for demand k.

    Constraints:
        - For each demand k: 0 <= f_k <= d_k
        - For each edge e: sum_k sum_{p∋e} y_{k,p} <= capacity_e

    Objective:
        max sum_k log(f_k + epsilon)

    Args:
        G: DiGraph with edge attr 'capacity' in same units as demand volumes.
        demands: List of (k_id, src, dst, volume).
        paths: Dict k_id -> list of paths (each path is a list of nodes).
        epsilon: small constant to avoid log(0).

    Returns:
        f_val: dict k_id -> served flow (0..d_k)
        y_var: CVXPY variable (vector of y_{k,p} in flat index order)
        prob: CVXPY problem object
        kp_index: dict (k_id, p_idx) -> index into y_var
    """
    # Precompute edge -> (k,p) pairs
    edge_to_kp = defaultdict(list)  # (u,v) -> list of (k_id, p_idx)
    for k_id, s, t, vol in demands:
        for p_idx, path in enumerate(paths[k_id]):
            edges_on_path = list(zip(path[:-1], path[1:]))
            for e in edges_on_path:
                edge_to_kp[e].append((k_id, p_idx))

    # Index mapping for CVXPY vars
    kp_list: List[Tuple[int, int]] = []
    kp_index: Dict[Tuple[int, int], int] = {}
    idx = 0
    for k_id, s, t, vol in demands:
        for p_idx, path in enumerate(paths[k_id]):
            kp_list.append((k_id, p_idx))
            kp_index[(k_id, p_idx)] = idx
            idx += 1
    n_kp = len(kp_list)

    # Variables: y for each (k,p)
    y = cp.Variable(n_kp, nonneg=True)

    # f_k = sum_p y_{k,p}
    f = {}
    for k_id, s, t, vol in demands:
        indices = [kp_index[(k_id, p_idx)] for p_idx, _ in enumerate(paths[k_id])]
        if indices:
            f[k_id] = cp.sum(y[indices])
        else:
            # no available path: force f_k = 0
            f[k_id] = 0.0

    constraints = []

    # Demand upper bound: f_k <= d_k
    for k_id, s, t, vol in demands:
        if isinstance(f[k_id], cp.Expression):
            constraints.append(f[k_id] <= vol)
        # if f[k_id] == 0.0 (no paths), no constraint needed

    # Capacity constraints
    for (u, v, data) in G.edges(data=True):
        cap = data["capacity"]
        indices = [kp_index[(k_id, p_idx)] for (k_id, p_idx) in edge_to_kp[(u, v)]]
        if indices:
            constraints.append(cp.sum(y[indices]) <= cap)

    # Objective: proportional fairness sum_k log(f_k + epsilon)
    # Only include k where f_k is a variable/expression (not constant 0.0)
    f_exprs = [
        f[k_id] + epsilon
        for k_id, s, t, vol in demands
        if isinstance(f[k_id], cp.Expression)
    ]
    if len(f_exprs) == 0:
        # degenerate case: no feasible flows
        obj = cp.Maximize(0.0)
    else:
        obj = cp.Maximize(cp.sum(cp.log(cp.hstack(f_exprs))))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)  # SCS uses ADMM-type operator splitting

    # Extract per-demand flow values
    f_val: Dict[int, float] = {}
    for k_id, s, t, vol in demands:
        if isinstance(f[k_id], cp.Expression):
            f_val[k_id] = float(f[k_id].value)
        else:
            f_val[k_id] = 0.0

    return f_val, y, prob, kp_index
