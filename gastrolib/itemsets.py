"""
Frequent itemset mining and association rule utilities.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import ceil
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from .data import RecipeDataset


def _normalize_ingredient_list(ingredients: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    for ing in ingredients:
        if isinstance(ing, str):
            val = ing.strip().lower()
            if val:
                normalized.append(val)
    return tuple(sorted(set(normalized)))


def _prepare_transactions(
    recipes: pd.DataFrame,
    ingredient_col: str,
) -> List[Tuple[str, ...]]:
    if ingredient_col not in recipes.columns:
        raise KeyError(f"recipes is missing required column: {ingredient_col!r}")

    transactions: List[Tuple[str, ...]] = []
    for ing_list in recipes[ingredient_col]:
        if isinstance(ing_list, Iterable) and not isinstance(ing_list, (str, bytes)):
            normalized = _normalize_ingredient_list(ing_list)
            if normalized:
                transactions.append(normalized)
    return transactions


def mine_frequent_itemsets(
    recipes: pd.DataFrame,
    ingredient_col: str = "ingredients",
    min_support: float | int = 0.1,
    max_size: int | None = None,
) -> pd.DataFrame:
    """
    Mine frequent ingredient itemsets using a simple Apriori algorithm.

    Parameters
    ----------
    recipes:
        Recipe dataset containing an ingredient list column.
    ingredient_col:
        Column with iterable ingredients per recipe.
    min_support:
        Minimum support threshold. If >=1, interpreted as absolute count,
        otherwise as fraction of total transactions.
    max_size:
        Optional maximum size of itemsets to return.

    Returns
    -------
    DataFrame
        Columns: itemset (tuple of strings), support (float in [0, 1]),
        support_count (int).
    """
    transactions = _prepare_transactions(recipes, ingredient_col)
    n_transactions = len(transactions)
    if not n_transactions:
        return pd.DataFrame(columns=["itemset", "support", "support_count"])

    if min_support >= 1:
        min_count = int(min_support)
    else:
        min_count = ceil(min_support * n_transactions)


    item_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for trans in transactions:
        for item in trans:
            item_counts[(item,)] += 1

    frequent_itemsets: Dict[int, Dict[Tuple[str, ...], int]] = {}
    L1 = {item: cnt for item, cnt in item_counts.items() if cnt >= min_count}
    if not L1:
        return pd.DataFrame(columns=["itemset", "support", "support_count"])
    frequent_itemsets[1] = L1

    k = 2
    while True:
        if max_size and k > max_size:
            break
        prev_freq = list(frequent_itemsets[k - 1].keys())
        candidates = set()
        for i in range(len(prev_freq)):
            for j in range(i + 1, len(prev_freq)):
                union = tuple(sorted(set(prev_freq[i]).union(prev_freq[j])))
                if len(union) == k:
                    subsets = combinations(union, k - 1)
                    if all(tuple(sorted(subset)) in frequent_itemsets[k - 1] for subset in subsets):
                        candidates.add(union)

        if not candidates:
            break

        candidate_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for trans in transactions:
            trans_set = set(trans)
            for cand in candidates:
                if set(cand).issubset(trans_set):
                    candidate_counts[cand] += 1

        Lk = {cand: cnt for cand, cnt in candidate_counts.items() if cnt >= min_count}
        if not Lk:
            break
        frequent_itemsets[k] = Lk
        k += 1

    rows = []
    for size, itemsets in frequent_itemsets.items():
        for itemset, count in itemsets.items():
            rows.append(
                {
                    "itemset": itemset,
                    "size": size,
                    "support": count / n_transactions,
                    "support_count": count,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["size", "support"], ascending=[True, False]
    )
    return result.reset_index(drop=True)




