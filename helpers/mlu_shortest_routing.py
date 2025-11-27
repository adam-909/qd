from collections import defaultdict
import networkx as nx
import pandas as pd

def compute_shortest_path_routing(G: nx.DiGraph, demands: pd.DataFrame, weight_attr="delay"):
    """
    Very simple baseline:
    - For each (src, dst, bw) demand, route all traffic on a single shortest path.
    - No ECMP splitting.
    Returns:
        edge_loads: dict[(u, v)] -> total traffic on that directed edge
    """
    edge_loads = defaultdict(float)

    for _, row in demands.iterrows():
        s = int(row["src"])
        d = int(row["dst"])
        bw = float(row["bw"])
        if s == d or bw == 0:
            continue

        try:
            path = nx.shortest_path(G, source=s, target=d, weight=weight_attr)
        except nx.NetworkXNoPath:
            # Skip if there's no path (shouldn't happen on Abilene, but just in case)
            continue

        # Add this demand's bandwidth to each edge on the path
        for u, v in zip(path[:-1], path[1:]):
            edge_loads[(u, v)] += bw

    return dict(edge_loads)


def compute_mlu(G: nx.DiGraph, edge_loads: dict) -> float:
    """
    Compute Maximum Link Utilisation (MLU):
        max_over_edges( load / capacity )
    """
    utilisations = []

    for (u, v), load in edge_loads.items():
        cap = G[u][v].get("capacity", 1.0)
        if cap <= 0:
            continue
        utilisations.append(load / cap)

    return max(utilisations) if utilisations else 0.0