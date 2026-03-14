"""
Ingredient co-occurrence network construction and visualization.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from networkx.algorithms import community as community_algorithms

from .data import RecipeDataset


def _coerce_dataframe(data: pd.DataFrame | RecipeDataset) -> pd.DataFrame:
    if isinstance(data, RecipeDataset):
        return data.to_dataframe(copy=True)
    return data.copy()


def build_ingredient_network(
    df: pd.DataFrame | RecipeDataset,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    min_cooccurrence: int = 1,
    max_nodes: Optional[int] = None,
    by_cuisine: Optional[str] = None,
) -> dict:
    """
    Build an ingredient co-occurrence network along with node/edge summaries.
    """
    data = _coerce_dataframe(df)
    if ingredient_col not in data.columns:
        raise KeyError(f"recipes is missing required column: {ingredient_col!r}")

    if by_cuisine:
        if cuisine_col not in data.columns:
            raise KeyError(f"recipes is missing cuisine column: {cuisine_col!r}")
        data = data[data[cuisine_col].str.lower() == by_cuisine.lower()]

    pair_counts: Counter[tuple[str, str]] = Counter()
    node_counts: Counter[str] = Counter()

    for ing_list in data[ingredient_col]:
        if not isinstance(ing_list, Iterable) or isinstance(ing_list, (str, bytes)):
            continue

        normalized = sorted(
            {
                str(ing).strip().lower()
                for ing in ing_list
                if isinstance(ing, str) and str(ing).strip()
            }
        )

        for ing in normalized:
            node_counts[ing] += 1

        for a, b in combinations(normalized, 2):
            pair_counts[tuple(sorted((a, b)))] += 1

    if max_nodes is not None and len(node_counts) > max_nodes:
        top_nodes = {n for n, _ in Counter(node_counts).most_common(max_nodes)}
    else:
        top_nodes = set(node_counts.keys())

    G = nx.Graph()

    for ing, cnt in node_counts.items():
        if ing in top_nodes:
            G.add_node(ing, frequency=cnt)

    edge_rows = []
    for (a, b), w in pair_counts.items():
        if w >= min_cooccurrence and a in G and b in G:
            G.add_edge(a, b, weight=w)
            edge_rows.append({"ingredient_1": a, "ingredient_2": b, "weight": w})

    freq_attr = nx.get_node_attributes(G, "frequency")
    node_stats = pd.DataFrame(
        [{"ingredient": node, "frequency": freq} for node, freq in freq_attr.items()]
    )
    if not node_stats.empty:
        node_stats["degree"] = node_stats["ingredient"].map(dict(G.degree()))

    edge_stats = pd.DataFrame(edge_rows)

    return {
        "graph": G,
        "node_stats": node_stats,
        "edge_stats": edge_stats,
    }


def build_ingredient_cooccurrence_network(
    df: pd.DataFrame | RecipeDataset,
    **kwargs,
) -> nx.Graph:
    """
    Backwards-compatible wrapper returning only the graph.
    """
    return build_ingredient_network(df, **kwargs)["graph"]

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Sequence

def plot_ingredient_network(
    G: nx.Graph,
    top_n: Optional[int] = 50,
    layout: str = "spring",
    node_size_attr: str = "frequency",
    node_size_scale: float = 20.0,   # still used as a global multiplier
    edge_width_scale: float = 0.2,
    figsize: tuple[float, float] = (10.0, 8.0),
    with_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    top_label_n: int = 20,           # NEW: max number of node labels
) -> plt.Axes:
    """
    Visualize an ingredient network, with node sizes rescaled so large
    networks remain readable and only top-degree nodes labelled.
    """
    # Restrict to top_n nodes by degree
    if top_n is not None and top_n < len(G):
        nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
        keep_nodes = {n for n, _ in nodes_by_degree[:top_n]}
        H = G.subgraph(keep_nodes).copy()
    else:
        H = G

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Layout
    if layout == "spring":
        # tweak k so nodes spread out a bit when there are many
        pos = nx.spring_layout(H, seed=0, k=1.0 / np.sqrt(max(len(H), 1)))
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H)
    elif layout == "circular":
        pos = nx.circular_layout(H)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    degrees = dict(H.degree)

    # --- Node sizes: rescale to a fixed visual range -----------------------
    if node_size_attr and nx.get_node_attributes(H, node_size_attr):
        attr = nx.get_node_attributes(H, node_size_attr)
        raw_sizes = np.array([attr.get(n, 1.0) for n in H.nodes], dtype=float)
    else:
        raw_sizes = np.array([degrees[n] for n in H.nodes], dtype=float)

    # Avoid all-equal list
    if raw_sizes.max() == raw_sizes.min():
        norm_sizes = np.ones_like(raw_sizes)
    else:
        norm_sizes = (raw_sizes - raw_sizes.min()) / (raw_sizes.max() - raw_sizes.min())

    # Map to a pleasant range, then multiply by node_size_scale
    min_size, max_size = 80.0, 1800.0
    node_sizes = (min_size + norm_sizes * (max_size - min_size)) * (node_size_scale / 20.0)

    # --- Edge widths --------------------------------------------------------
    edge_widths: Sequence[float] = [
        H[u][v].get("weight", 1.0) * edge_width_scale for u, v in H.edges
    ]

    if len(edge_widths) > 0:
        nx.draw_networkx_edges(
            H,
            pos,
            width=edge_widths,
            edge_color="#888888",
            alpha=0.4,        # lighter, more transparent
            ax=ax,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No edges to display (try lowering `min_cooccurrence`).",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="#333333",
        )

    # --- Nodes --------------------------------------------------------------
    nodes = nx.draw_networkx_nodes(
        H,
        pos,
        node_size=node_sizes,
        node_color=list(degrees.values()),
        cmap="viridis",
        ax=ax,
    )

    # --- Labels: only top_label_n nodes by degree --------------------------
    if with_labels and len(H) > 0:
        top_label_n = min(top_label_n, len(H))
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_label_n]
        label_dict = {n: n for n in top_nodes}
        nx.draw_networkx_labels(
            H,
            pos,
            labels=label_dict,
            font_size=8,
            font_color="black",
            verticalalignment="center",
            ax=ax,
        )

    ax.set_axis_off()
    plt.colorbar(nodes, ax=ax, label="Degree")
    ax.set_title("Ingredient Co-occurrence Network")
    return ax





def build_cuisine_ingredient_network(
    df: pd.DataFrame | RecipeDataset,
    cuisine: str,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    min_cooccurrence: int = 2,
    max_nodes: Optional[int] = None,
) -> nx.Graph:
    """
    Build an ingredient co-occurrence network for a specific cuisine.

    This is a convenience function that creates an ingredient network
    filtered to recipes from a specific cuisine, making it easier to
    analyze ingredient relationships within culinary traditions.

    Parameters
    ----------
    df : pd.DataFrame | RecipeDataset
        Recipe dataset containing ingredients and cuisine labels.
    cuisine : str
        Name of the cuisine to analyze (case-insensitive).
    ingredient_col : str, default="ingredients"
        Name of the column containing ingredient lists.
    cuisine_col : str, default="cuisine"
        Name of the column containing cuisine labels.
    min_cooccurrence : int, default=2
        Minimum number of shared recipes required for an edge.
    max_nodes : int, optional
        Maximum number of ingredients to include (keeps most frequent).

    Returns
    -------
    nx.Graph
        NetworkX graph where nodes are ingredients and edges represent
        co-occurrence relationships within the specified cuisine.

    Examples
    --------
    >>> from gastrolib import load_sample_cuisine_dataset, build_cuisine_ingredient_network
    >>> dataset, _ = load_sample_cuisine_dataset()
    >>> italian_network = build_cuisine_ingredient_network(dataset, "italian")
    >>> print(f"Italian network: {len(italian_network.nodes)} ingredients")
    """
    return build_ingredient_cooccurrence_network(
        df,
        by_cuisine=cuisine,
        ingredient_col=ingredient_col,
        cuisine_col=cuisine_col,
        min_cooccurrence=min_cooccurrence,
        max_nodes=max_nodes,
    )


