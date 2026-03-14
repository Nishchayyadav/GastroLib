"""
Ingredient pairing analysis utilities.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import RecipeDataset


def _coerce_dataframe(data: pd.DataFrame | RecipeDataset) -> pd.DataFrame:
    if isinstance(data, RecipeDataset):
        return data.to_dataframe(copy=True)
    return data.copy()


def _normalize_ingredients(ingredients: Iterable[str]) -> list[str]:
    result = []
    for ing in ingredients:
        if isinstance(ing, str):
            val = ing.strip().lower()
            if val:
                result.append(val)
    return sorted(set(result))


def analyze_food_pairing(
    df: pd.DataFrame | RecipeDataset,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    min_cooccurrence: int = 2,
    normalize: bool = True,
    by_cuisine: bool = False,
) -> dict:
    """
    Compute ingredient pair statistics and matrices.
    """
    data = _coerce_dataframe(df)
    if ingredient_col not in data.columns:
        raise KeyError(f"recipes is missing required column: {ingredient_col!r}")

    def _analyze_subset(subset: pd.DataFrame) -> dict:
        pair_counts: dict[tuple[str, str], int] = {}
        ingredient_counts: dict[str, int] = {}

        for ingredients in subset[ingredient_col]:
            if not isinstance(ingredients, Iterable) or isinstance(
                ingredients, (str, bytes)
            ):
                continue
            normalized = _normalize_ingredients(ingredients)
            for ing in normalized:
                ingredient_counts[ing] = ingredient_counts.get(ing, 0) + 1
            for a, b in combinations(normalized, 2):
                key = tuple(sorted((a, b)))
                pair_counts[key] = pair_counts.get(key, 0) + 1

        rows = []
        ingredients_list = sorted(ingredient_counts.keys())
        matrix = np.zeros((len(ingredients_list), len(ingredients_list)))
        index_map = {ing: idx for idx, ing in enumerate(ingredients_list)}

        for (a, b), count in pair_counts.items():
            if count < min_cooccurrence:
                continue
            if normalize:
                denom = ingredient_counts[a] + ingredient_counts[b] - count
                score = count / denom if denom else 0.0
            else:
                score = float(count)
            rows.append(
                {
                    "ingredient_1": a,
                    "ingredient_2": b,
                    "cooccurrence": count,
                    "score": score,
                }
            )
            ia, ib = index_map[a], index_map[b]
            matrix[ia, ib] = score
            matrix[ib, ia] = score

        np.fill_diagonal(matrix, 1.0)
        pair_stats = pd.DataFrame(rows).sort_values(
            ["score", "cooccurrence"], ascending=[False, False]
        )
        matrix_df = pd.DataFrame(matrix, index=ingredients_list, columns=ingredients_list)

        return {
            "pair_stats": pair_stats,
            "matrix": matrix,
            "ingredients": ingredients_list,
            "matrix_df": matrix_df,
        }

    if not by_cuisine:
        return _analyze_subset(data)

    if cuisine_col not in data.columns:
        raise KeyError(f"recipes is missing cuisine column: {cuisine_col!r}")

    stats_by_cuisine = {}
    matrices_by_cuisine = {}
    for cuisine, subset in data.groupby(cuisine_col):
        result = _analyze_subset(subset)
        stats_by_cuisine[cuisine] = result["pair_stats"]
        matrices_by_cuisine[cuisine] = {
            "matrix": result["matrix"],
            "ingredients": result["ingredients"],
            "matrix_df": result["matrix_df"],
        }
    return {
        "pair_stats_by_cuisine": stats_by_cuisine,
        "matrices_by_cuisine": matrices_by_cuisine,
    }


def compute_pairing_matrix(
    df: pd.DataFrame | RecipeDataset,
    ingredient_col: str = "ingredients",
    min_cooccurrences: int = 1,
) -> pd.DataFrame:
    """
    Backwards-compatible helper returning a pandas DataFrame matrix.
    """
    result = analyze_food_pairing(
        df,
        ingredient_col=ingredient_col,
        min_cooccurrence=min_cooccurrences,
        normalize=True,
        by_cuisine=False,
    )
    return result["matrix_df"]


def plot_pairing_heatmap(
    matrix: np.ndarray,
    ingredients: Sequence[str],
    top_n: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "mako",
) -> plt.Axes:
    """
    Plot a heatmap of ingredient pairing scores.
    """
    if top_n is not None and top_n < len(ingredients):
        indices = list(range(top_n))
    else:
        indices = list(range(len(ingredients)))

    subset = matrix[np.ix_(indices, indices)]
    labels = [ingredients[i] for i in indices]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        subset,
        ax=ax,
        cmap=cmap,
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Ingredient Pairing Heatmap")
    return ax




