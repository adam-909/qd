# helpers/te_metrics.py

import numpy as np


def total_throughput(f_k: dict[int, float]) -> float:
    return float(sum(f_k.values()))


def jain_fairness(f_k: dict[int, float]) -> float:
    flows = np.array(list(f_k.values()), dtype=float)
    num = flows.sum() ** 2
    den = len(flows) * (flows ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def blocking_probability(f_k: dict[int, float], demands):
    demand_vol = {k_id: vol for (k_id, src, dst, vol) in demands}
    blocked = sum(demand_vol[k_id] - f_k.get(k_id, 0.0) for k_id in demand_vol)
    total = sum(demand_vol.values())
    return float(blocked / total) if total > 0 else 0.0


def tail_latency_proxy(effective_hops: dict[int, float], q: float = 0.95) -> float:
    vals = np.array(
        [v for v in effective_hops.values() if np.isfinite(v)],
        dtype=float,
    )
    if len(vals) == 0:
        return float("inf")
    return float(np.quantile(vals, q))

def effective_hop_lengths(paths, y_vals, kp_index, demands):
    # y_vals: solution vector (numpy) for path-based models
    eff_len = {}
    for k_id, s, t, vol in demands:
        lengths = []
        weights = []
        for p_idx, path in enumerate(paths[k_id]):
            idx = kp_index[(k_id, p_idx)]
            y_kp = y_vals[idx]
            if y_kp > 0:
                hop_len = len(path) - 1
                lengths.append(hop_len)
                weights.append(y_kp)
        if weights:
            w = np.array(weights)
            l = np.array(lengths)
            eff_len[k_id] = float((w * l).sum() / w.sum())
        else:
            eff_len[k_id] = np.inf  # completely blocked
    return eff_len
