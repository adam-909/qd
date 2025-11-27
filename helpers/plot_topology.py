import networkx as nx
import matplotlib.pyplot as plt

def plot_repetita_topology(G: nx.DiGraph, title="Repetita Topology"):
    # Use x,y positions from the file
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}

    # Extract city names (strip the numeric prefix)
    labels = {
        n: G.nodes[n]["label"].split("_", 1)[1]    # "0_New_York" -> "New_York"
        for n in G.nodes()
    }

    plt.figure(figsize=(9, 7))
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=350,
        node_color="lightblue",
        edgecolors="k"
    )

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="-|>",
        arrowsize=12,
        width=[G[u][v]["capacity"] / 1e7 for u, v in G.edges()]
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_color="black"
    )

    plt.title(title)
    plt.axis("off")
    plt.show()
