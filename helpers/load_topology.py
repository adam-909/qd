import networkx as nx
from pathlib import Path

def load_repetita_topology(path: Path) -> nx.DiGraph:
    """
    Load a Repetita topology file

    Format:
    NODES 11
    label x y
    0_New_York -74.00597 40.71427
    ...
    
    EDGES 28
    label src dest weight bw delay
    edge_0 0 1 1 9953280 1913
    ...
    """
    G = nx.DiGraph()
    mode = None  # "nodes" or "edges"

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Section switches
            if line.startswith("NODES"):
                mode = "nodes"
                continue
            if line.startswith("EDGES"):
                mode = "edges"
                continue
            if line.startswith("label"):
                # header line inside a section
                continue

            parts = line.split()

            if mode == "nodes":
                # Example: 0_New_York -74.00597 40.71427
                label = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                # Node id is the prefix before the first underscore
                node_id_str = label.split("_", 1)[0]
                node_id = int(node_id_str)
                G.add_node(node_id, label=label, x=x, y=y)

            elif mode == "edges":
                # Example: edge_0 0 1 1 9953280 1913
                edge_label = parts[0]
                src = int(parts[1])
                dst = int(parts[2])
                weight = float(parts[3])
                bw = float(parts[4])      # capacity
                delay = float(parts[5])
                G.add_edge(
                    src,
                    dst,
                    label=edge_label,
                    weight=weight,
                    capacity=bw,
                    delay=delay,
                )

    return G