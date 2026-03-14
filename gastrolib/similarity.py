"""
Cuisine similarity computation and clustering utilities.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from .data import RecipeDataset

SimilarityMetric = Literal["cosine", "jaccard"]
EmbeddingMethod = Literal["pca", "tsne"]


def _coerce_dataframe(data: pd.DataFrame | RecipeDataset) -> pd.DataFrame:
    if isinstance(data, RecipeDataset):
        return data.to_dataframe(copy=True)
    return data.copy()


def compute_cuisine_similarity(
    df: pd.DataFrame | RecipeDataset,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    similarity_metric: SimilarityMetric = "cosine",
    min_ingredient_frequency: int = 5,
) -> dict:
    """
    Build cuisine-level ingredient vectors and similarity matrix.
    """
    data = _coerce_dataframe(df)
    if ingredient_col not in data.columns or cuisine_col not in data.columns:
        raise KeyError("recipes must contain ingredient and cuisine columns")

    exploded = data[[cuisine_col, ingredient_col]].explode(ingredient_col).dropna()
    exploded[ingredient_col] = exploded[ingredient_col].astype(str).str.strip()
    exploded = exploded[exploded[ingredient_col] != ""]

    ingredient_counts = exploded[ingredient_col].value_counts()
    keep_ingredients = ingredient_counts[
        ingredient_counts >= max(1, min_ingredient_frequency)
    ].index

    filtered = exploded[exploded[ingredient_col].isin(keep_ingredients)]
    cuisine_vectors = (
        filtered.groupby([cuisine_col, ingredient_col])
        .size()
        .unstack(fill_value=0)
        .astype(float)
    )

    if similarity_metric == "jaccard":
        binary = (cuisine_vectors > 0).astype(int)
        dist = pairwise_distances(binary.values, metric="jaccard")
    else:
        dist = pairwise_distances(cuisine_vectors.values, metric="cosine")

    similarity_matrix = 1 - dist
    np.fill_diagonal(similarity_matrix, 1.0)

    return {
        "cuisine_vectors": cuisine_vectors,
        "similarity_matrix": similarity_matrix,
        "cuisines": cuisine_vectors.index.tolist(),
    }



