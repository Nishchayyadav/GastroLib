"""
Nutritional pattern analysis for recipes and cuisines.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import RecipeDataset

def _coerce_dataframe(data: pd.DataFrame | RecipeDataset) -> pd.DataFrame:
    if isinstance(data, RecipeDataset):
        return data.to_dataframe(copy=True)
    return data.copy()

def aggregate_recipe_nutrition_from_ingredients(
    recipes: pd.DataFrame | RecipeDataset,
    ingredient_nutrition: pd.DataFrame,
    ingredient_col: str = "ingredients",
    recipe_id_col: Optional[str] = "recipe_id",
    nutrition_cols: Optional[Sequence[str]] = None,
    quantity_col: Optional[str] = "ingredient_quantities",
    default_quantity: float = 100.0,
) -> pd.DataFrame:
    """
    Build recipe-level nutrition by summing per-ingredient nutrition entries.

    Parameters
    ----------
    ingredient_nutrition:
        DataFrame indexed by ingredient (lowercase) with nutrition columns.
    nutrition_cols:
        If None, uses all numeric columns from ingredient_nutrition.
    """
    df = _coerce_dataframe(recipes)

    if ingredient_col not in df.columns:
        raise KeyError(f"recipes is missing ingredient column: {ingredient_col!r}")
    if recipe_id_col is not None and recipe_id_col not in df.columns:
        raise KeyError(f"recipes is missing recipe id column: {recipe_id_col!r}")

    lookup = ingredient_nutrition.copy()
    if "ingredient" in lookup.columns:
        lookup = lookup.set_index("ingredient")
    if lookup.index.name != "ingredient":
        lookup.index = lookup.index.map(lambda x: str(x).lower())
        lookup.index.name = "ingredient"
    if nutrition_cols is None:
        nutrition_cols = [
            c for c in lookup.columns if np.issubdtype(lookup[c].dtype, np.number)
        ]

    records = []
    for idx, row in df.iterrows():
        total = dict.fromkeys(nutrition_cols, 0.0)
        ingredients = row.get(ingredient_col, [])
        quantity_map = (
            row.get(quantity_col, {}) if quantity_col and quantity_col in df.columns else {}
        )
        if not isinstance(quantity_map, dict):
            quantity_map = {}
        if isinstance(ingredients, Iterable) and not isinstance(ingredients, (str, bytes)):
            for ing in ingredients:
                if not isinstance(ing, str):
                    continue
                key = ing.strip().lower()
                if key in lookup.index:
                    values = lookup.loc[key, nutrition_cols]
                    for col in nutrition_cols:
                        quantity = float(quantity_map.get(key, default_quantity))
                        scale = quantity / default_quantity if default_quantity else 1.0
                        total[col] += float(values[col]) * scale
        records.append(
            {
                (recipe_id_col or "recipe_index"): row[recipe_id_col] if recipe_id_col else idx,
                **{col: total[col] for col in nutrition_cols},
            }
        )
    result = pd.DataFrame(records)
    return result








