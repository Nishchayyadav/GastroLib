"""
Ingredient-level statistics and cuisine profiles.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _flatten_ingredients(series: pd.Series) -> List[str]:
    """
    Flatten a Series of ingredient lists into a single list of normalized strings.
    """
    flattened: List[str] = []
    for item in series:
        if not isinstance(item, Iterable) or isinstance(item, (str, bytes)):
            continue
        for ing in item:
            if not isinstance(ing, str):
                continue
            normalized = ing.strip().lower()
            if normalized:
                flattened.append(normalized)
    return flattened


def summarize_ingredients(
    recipes: pd.DataFrame,
    ingredient_col: str = "ingredients",
    cuisine_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute basic ingredient statistics across a recipe dataset.

    Parameters
    ----------
    recipes:
        DataFrame of recipes. Must contain a column with lists of ingredients.
    ingredient_col:
        Name of the column containing ingredient lists.
    cuisine_col:
        Optional column specifying cuisine labels. If provided, per-cuisine
        usage counts and recipe coverage are computed.

    Returns
    -------
    DataFrame
        Indexed by ingredient with columns:
        - count: total number of times ingredient appears
        - recipe_count: number of recipes containing the ingredient
        - p_recipes: fraction of recipes containing the ingredient
        - (optional) additional per-cuisine columns if `cuisine_col` is given
    """
    if ingredient_col not in recipes.columns:
        raise KeyError(f"recipes is missing required column: {ingredient_col!r}")

    n_recipes = len(recipes)

    all_ings = _flatten_ingredients(recipes[ingredient_col])
    total_counts = Counter(all_ings)

    recipe_counts: Counter[str] = Counter()
    for ing_list in recipes[ingredient_col]:
        if not isinstance(ing_list, Iterable) or isinstance(ing_list, (str, bytes)):
            continue
        normalized = {
            str(ing).strip().lower()
            for ing in ing_list
            if isinstance(ing, str) and str(ing).strip()
        }
        for ing in normalized:
            recipe_counts[ing] += 1

    ingredients = sorted(total_counts.keys())
    data = {
        "ingredient": ingredients,
        "count": [total_counts[i] for i in ingredients],
        "recipe_count": [recipe_counts[i] for i in ingredients],
    }
    df = pd.DataFrame(data).set_index("ingredient")
    df["p_recipes"] = df["recipe_count"] / float(n_recipes) if n_recipes else np.nan

    if cuisine_col and cuisine_col in recipes.columns:
        for cuisine, sub in recipes.groupby(cuisine_col):
            c_counts: Counter[str] = Counter()
            for ing_list in sub[ingredient_col]:
                if not isinstance(ing_list, Iterable) or isinstance(
                    ing_list, (str, bytes)
                ):
                    continue
                normalized = {
                    str(ing).strip().lower()
                    for ing in ing_list
                    if isinstance(ing, str) and str(ing).strip()
                }
                for ing in normalized:
                    c_counts[ing] += 1
            col_name = f"recipes_in_{cuisine}"
            df[col_name] = [c_counts.get(i, 0) for i in df.index]

    return df.sort_values("count", ascending=False)




