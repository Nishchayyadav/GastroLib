"""
Dataset utilities for GastroLib.
"""

from __future__ import annotations
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

REQUIRED_RECIPE_COLUMNS = {"recipe_id", "recipe_name", "cuisine", "ingredients"}
LIST_COLUMNS = ("ingredients", "methods")
DICT_COLUMNS = ("ingredient_quantities",)


def validate_recipes_dataframe(
    recipes: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_RECIPE_COLUMNS,
) -> None:
    missing = [col for col in required_columns if col not in recipes.columns]
    if missing:
        raise ValueError(f"Recipes dataframe missing required columns: {missing}")


class RecipeDataset:
    """
    Helper wrapper around the canonical recipe schema.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        normalized = self._normalize_dataframe(df.copy())
        self._df = normalized

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> "RecipeDataset":
        df = pd.read_csv(path, **kwargs)
        return cls(df)

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> "RecipeDataset":
        df = pd.read_parquet(path, **kwargs)
        return cls(df)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "RecipeDataset":
        return cls(df)

    def to_dataframe(self, copy: bool = True) -> pd.DataFrame:
        return self._df.copy() if copy else self._df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def filter_by_cuisine(self, cuisine_name: str) -> "RecipeDataset":
        mask = self._df["cuisine"].str.lower() == str(cuisine_name).lower()
        return RecipeDataset(self._df.loc[mask].copy())

    def copy(self) -> "RecipeDataset":
        return RecipeDataset(self._df.copy())

    def __len__(self) -> int:
        return len(self._df)

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "recipe_name" not in df.columns and "name" in df.columns:
            df = df.rename(columns={"name": "recipe_name"})

        validate_recipes_dataframe(df)

        df["recipe_id"] = df["recipe_id"].astype(str)
        df["recipe_name"] = df["recipe_name"].astype(str)
        df["cuisine"] = df["cuisine"].astype(str)

        for col in LIST_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(self._ensure_list_of_str)
            elif col == "ingredients":
                df[col] = [[] for _ in range(len(df))]

        for col in DICT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(self._ensure_quantity_mapping)

        return df

    @staticmethod
    def _ensure_list_of_str(value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                parsed = ast.literal_eval(value)
                value = parsed
            except (ValueError, SyntaxError):
                value = [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()]

    @staticmethod
    def _ensure_quantity_mapping(value: Any) -> Dict[str, float]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return {}
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
        if isinstance(value, dict):
            cleaned: Dict[str, float] = {}
            for k, v in value.items():
                try:
                    cleaned[str(k).strip()] = float(v)
                except (TypeError, ValueError):
                    continue
            return cleaned
        return {}


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_sample_cuisine_dataset() -> tuple[RecipeDataset, pd.DataFrame]:
    """
    Load a small in-memory dataset for demos and testing.
    Returns (RecipeDataset, ingredient_nutrition_table).
    """
    data = [
        {
            "recipe_id": 1,
            "recipe_name": "Spaghetti al Pomodoro",
            "cuisine": "italian",
            "category": "main",
            "ingredients": ["spaghetti", "tomato", "olive oil", "garlic", "basil", "salt"],
        },
        {
            "recipe_id": 2,
            "recipe_name": "Pesto Pasta",
            "cuisine": "italian",
            "category": "main",
            "ingredients": ["spaghetti", "basil", "pine nuts", "olive oil", "parmesan", "garlic", "salt"],
        },
        {
            "recipe_id": 3,
            "recipe_name": "Chana Masala",
            "cuisine": "indian",
            "category": "main",
            "ingredients": ["chickpeas", "tomato", "onion", "garlic", "ginger", "cumin", "coriander", "chili", "turmeric", "oil", "salt"],
        },
        {
            "recipe_id": 4,
            "recipe_name": "Guacamole",
            "cuisine": "mexican",
            "category": "starter",
            "ingredients": ["avocado", "tomato", "onion", "cilantro", "lime", "salt", "chili"],
        },
    ]
    recipes = RecipeDataset(pd.DataFrame(data))

    ingredient_nutrition = pd.DataFrame(
        [
            ("tomato", 22, 1, 0, 5),
            ("olive oil", 119, 0, 14, 0),
            ("garlic", 4, 0, 0, 1),
            ("basil", 1, 0, 0, 0),
            ("spaghetti", 200, 7, 1, 42),
            ("pine nuts", 191, 4, 19, 4),
            ("parmesan", 110, 10, 7, 1),
            ("chickpeas", 180, 10, 3, 30),
            ("onion", 44, 1, 0, 10),
            ("ginger", 5, 0, 0, 1),
            ("cumin", 8, 0, 0, 1),
            ("cilantro", 1, 0, 0, 0),
            ("lime", 11, 0, 0, 4),
            ("avocado", 160, 2, 15, 9),
        ],
        columns=["ingredient", "calories", "protein", "fat", "carbs"],
    ).set_index("ingredient")

    return recipes, ingredient_nutrition


def load_recipes(
    path: str | Path,
    ingredient_nutrition_path: Optional[str | Path] = None,
) -> Tuple[RecipeDataset, Optional[pd.DataFrame]]:
    """
    Load recipes from a CSV/TSV/Parquet file and return a `RecipeDataset`.

    If `ingredient_nutrition_path` is provided, also load that table and return
    it as a DataFrame indexed by ingredient name (if an `ingredient` column
    exists it will be used as the index).

    The function uses the same parsing/normalization rules as `RecipeDataset`.

    Returns: `(RecipeDataset, ingredient_nutrition_df_or_None)`
    """
    recipes_table = _read_table(path)
    dataset = RecipeDataset(recipes_table)

    ingredient_nutrition: Optional[pd.DataFrame] = None
    if ingredient_nutrition_path is not None:
        ing_table = _read_table(ingredient_nutrition_path)
        if "ingredient" in ing_table.columns:
            ing_table = ing_table.set_index("ingredient")
        ingredient_nutrition = ing_table

    return dataset, ingredient_nutrition