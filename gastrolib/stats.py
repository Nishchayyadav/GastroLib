"""
Core statistical utilities for recipe datasets.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import RecipeDataset


def _coerce_dataframe(data: pd.DataFrame | RecipeDataset) -> pd.DataFrame:
    if isinstance(data, RecipeDataset):
        return data.to_dataframe(copy=True)
    return data.copy()


def _compute_basic_stats(series: pd.Series) -> Dict[str, float | Dict[int, float]]:
    percentiles = {
        25: float(np.percentile(series, 25)),
        50: float(np.percentile(series, 50)),
        75: float(np.percentile(series, 75)),
    }
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "min": int(series.min()) if len(series) else 0,
        "max": int(series.max()) if len(series) else 0,
        "percentiles": percentiles,
    }


def compute_recipe_size_distribution(
    df: pd.DataFrame | RecipeDataset,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    by_cuisine: bool = False,
    plot: bool = False,
    bins: int = 20,
) -> Dict[str, object]:
    """
    Compute ingredient-count distributions globally or per cuisine.
    """
    data = _coerce_dataframe(df)
    if ingredient_col not in data.columns:
        raise KeyError(f"recipes is missing required column: {ingredient_col!r}")

    size_series = data[ingredient_col].apply(
        lambda x: len(x) if isinstance(x, (list, tuple, set)) else 0
    )

    if not by_cuisine:
        counts, bin_edges = np.histogram(size_series, bins=bins)
        figure = None
        if plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(size_series, bins=bins, color="#4c72b0", edgecolor="white")
            ax.set_xlabel("Number of ingredients per recipe")
            ax.set_ylabel("Count")
            ax.set_title("Recipe Size Distribution")
            figure = fig

        return {
            "stats": _compute_basic_stats(size_series),
            "histogram_data": {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            },
            "figure": figure,
        }

    if cuisine_col not in data.columns:
        raise KeyError(f"recipes is missing cuisine column: {cuisine_col!r}")

    stats_by_cuisine: Dict[str, Dict[str, object]] = {}
    histogram_data_by_cuisine: Dict[str, Dict[str, list]] = {}
    figures: Dict[str, plt.Figure] = {}

    for cuisine, subset in data.groupby(cuisine_col):
        series = subset[ingredient_col].apply(
            lambda x: len(x) if isinstance(x, (list, tuple, set)) else 0
        )
        stats_by_cuisine[cuisine] = _compute_basic_stats(series)
        counts, bin_edges = np.histogram(series, bins=bins)
        histogram_data_by_cuisine[cuisine] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }
        if plot:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(series, bins=bins, color="#4c72b0", edgecolor="white")
            ax.set_title(f"Recipe Size: {cuisine}")
            ax.set_xlabel("Ingredient count")
            ax.set_ylabel("Recipes")
            figures[cuisine] = fig

    return {
        "stats_by_cuisine": stats_by_cuisine,
        "histogram_data_by_cuisine": histogram_data_by_cuisine,
        "figures": figures if plot else None,
    }


def compute_ingredient_popularity(
    df: pd.DataFrame | RecipeDataset,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    by_cuisine: bool = False,
    top_k: Optional[int] = None,
    plot: bool = False,
) -> Dict[str, object]:
    """
    Compute ingredient frequencies globally and per cuisine.
    """
    data = _coerce_dataframe(df)
    if ingredient_col not in data.columns:
        raise KeyError(f"recipes is missing required column: {ingredient_col!r}")

    def count_ingredients(series: pd.Series) -> pd.Series:
        exploded = series.explode().dropna()
        exploded = exploded.astype(str).str.strip()
        exploded = exploded[exploded != ""]
        counts = exploded.value_counts().sort_values(ascending=False)
        if top_k:
            counts = counts.head(top_k)
        return counts

    global_counts = count_ingredients(data[ingredient_col])
    total = global_counts.sum() if global_counts.sum() else 1
    global_table = pd.DataFrame(
        {
            "ingredient": global_counts.index,
            "count": global_counts.values,
            "frequency": (global_counts / total).values,
        }
    )

    global_fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=global_table,
            x="ingredient",
            y="count",
            color="#55a868",
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Top Ingredients")
        global_fig = fig

    if not by_cuisine:
        return {
            "global_popularity": global_table,
            "figure": global_fig,
        }

    if cuisine_col not in data.columns:
        raise KeyError(f"recipes is missing cuisine column: {cuisine_col!r}")

    popularity_by_cuisine: Dict[str, pd.DataFrame] = {}
    figures_by_cuisine: Dict[str, plt.Figure] = {}

    for cuisine, subset in data.groupby(cuisine_col):
        counts = count_ingredients(subset[ingredient_col])
        total_counts = counts.sum() if counts.sum() else 1
        table = pd.DataFrame(
            {
                "ingredient": counts.index,
                "count": counts.values,
                "frequency": (counts / total_counts).values,
            }
        )
        popularity_by_cuisine[cuisine] = table
        if plot:
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(data=table, x="ingredient", y="count", color="#4c72b0", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"Top Ingredients: {cuisine}")
            figures_by_cuisine[cuisine] = fig

    return {
        "global_popularity": global_table,
        "popularity_by_cuisine": popularity_by_cuisine,
        "figures": {
            "global": global_fig,
            "by_cuisine": figures_by_cuisine if plot else None,
        },
    }


def analyze_category_composition(
    df: pd.DataFrame | RecipeDataset,
    group_by: str = "cuisine",
    category_col: str = "category",
    plot: bool = False,
) -> Dict[str, object]:
    """
    Compute category composition for each group (default: cuisine).
    """
    data = _coerce_dataframe(df)
    if category_col not in data.columns:
        raise KeyError(f"recipes is missing category column: {category_col!r}")
    if group_by not in data.columns:
        raise KeyError(f"recipes is missing grouping column: {group_by!r}")

    counts = (
        data.groupby([group_by, category_col]).size().reset_index(name="count")
    )
    totals = counts.groupby(group_by)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    figures: Dict[str, plt.Figure] = {}
    if plot:
        for group_value, subset in counts.groupby(group_by):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                subset["count"],
                labels=subset[category_col],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"{group_by.title()}: {group_value}")
            ax.axis("equal")
            figures[group_value] = fig

    return {
        "composition_table": counts,
        "figures": figures if plot else None,
    }


def plot_category_composition_pie(
    composition_table: pd.DataFrame,
    cuisine_col: str = "cuisine",
    count_col: str = "count",
    top_n: Optional[int] = 15,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a clean pie chart of ALL cuisines.

    Each wedge = one cuisine (or 'Other'), with a spaced-out legend on the side.
    """
    df = composition_table[composition_table[cuisine_col] != 'nan']
    counts = (
        df.groupby(cuisine_col)[count_col]
        .sum()
        .sort_values(ascending=False)
    )

    if top_n is not None and len(counts) > top_n:
        top = counts.iloc[:top_n]
        other_sum = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other_sum})])

    labels = counts.index.tolist()
    values = counts.values
    total = float(values.sum())
    percents = values / total * 100.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=90,
    )
    ax.axis("equal")
    ax.set_title("Cuisine distribution")
    legend_labels = [f"{lab} ({p:.1f}%)" for lab, p in zip(labels, percents)]
    ax.legend(
        wedges,
        legend_labels,
        title="Cuisines",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    return ax


def plot_cuisine_similarity_heatmap(similarity_dict):
    """
    similarity_dict: output of compute_cuisine_similarity
        expects keys: 'similarity_matrix', 'cuisines'
    """
    sim = np.array(similarity_dict["similarity_matrix"])
    cuisines = list(similarity_dict["cuisines"])

    # Drop any 'nan' cuisine (and its row/column)
    valid_idx = [i for i, c in enumerate(cuisines) if pd.notna(c) and str(c) != "nan"]
    sim = sim[np.ix_(valid_idx, valid_idx)]
    cuisines = [cuisines[i] for i in valid_idx]

    # Put into a DataFrame for nice labeling
    sim_df = pd.DataFrame(sim, index=cuisines, columns=cuisines)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_df,
        vmin=0, vmax=1,
        square=True,
        cmap="viridis",      # feel free to change
        cbar_kws={"label": "Similarity"},
    )
    plt.title("Cuisine–Cuisine Similarity Heatmap")
    plt.xlabel("Cuisine")
    plt.ylabel("Cuisine")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()