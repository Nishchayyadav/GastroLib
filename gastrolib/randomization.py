"""
Cuisine randomization and comparison utilities for GastroLib.

This module provides tools to create randomized versions of cuisines
to understand what makes real cuisines unique through comparison.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal
from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local utility function
def _coerce_dataframe(data: pd.DataFrame | RecipeDataset) -> pd.DataFrame:
    """Convert RecipeDataset to DataFrame if needed."""
    if hasattr(data, 'to_dataframe'):
        return data.to_dataframe(copy=True)
    return data.copy()


# Ingredient categorization system
INGREDIENT_CATEGORIES = {
    # Proteins
    "proteins": [
        "chicken", "beef", "pork", "fish", "salmon", "tuna", "shrimp", "eggs",
        "tofu", "tempeh", "lentils", "chickpeas", "beans", "peas"
    ],

    # Dairy
    "dairy": [
        "milk", "cheese", "parmesan", "yogurt", "butter", "ghee", "cream",
        "mozzarella", "feta", "ricotta"
    ],

    # Vegetables
    "vegetables": [
        "tomato", "onion", "garlic", "ginger", "potato", "carrot", "spinach",
        "lettuce", "cabbage", "broccoli", "cauliflower", "cucumber", "bell pepper",
        "zucchini", "eggplant", "avocado", "mushroom", "celery", "leek"
    ],

    # Fruits
    "fruits": [
        "lemon", "lime", "apple", "banana", "orange", "pineapple", "mango",
        "coconut", "raisins", "dates"
    ],

    # Grains & Carbs
    "grains": [
        "rice", "wheat", "flour", "bread", "pasta", "spaghetti", "noodles",
        "quinoa", "barley", "oats", "corn"
    ],

    # Spices & Herbs
    "spices": [
        "cumin", "coriander", "turmeric", "chili", "paprika", "cinnamon",
        "cardamom", "cloves", "nutmeg", "saffron", "fennel", "fenugreek",
        "mustard", "curry powder", "garam masala", "thyme", "rosemary",
        "oregano", "basil", "parsley", "cilantro", "mint", "dill"
    ],

    # Oils & Fats
    "oils": [
        "olive oil", "vegetable oil", "coconut oil", "canola oil", "sesame oil",
        "butter", "ghee"
    ],

    # Sweeteners
    "sweeteners": [
        "sugar", "honey", "maple syrup", "brown sugar"
    ],

    # Nuts & Seeds
    "nuts_seeds": [
        "almonds", "cashews", "peanuts", "walnuts", "pine nuts", "sesame seeds",
        "chia seeds", "flax seeds", "sunflower seeds"
    ],

    # Other
    "other": [
        "salt", "pepper", "vinegar", "soy sauce", "water", "stock", "wine"
    ]
}


def categorize_ingredient(ingredient: str) -> str:
    """Categorize a single ingredient."""
    ingredient_lower = ingredient.lower().strip()

    for category, ingredients in INGREDIENT_CATEGORIES.items():
        if ingredient_lower in ingredients:
            return category

    return "other"


def get_cuisine_ingredient_frequencies(
    df: pd.DataFrame | RecipeDataset,
    cuisine: Optional[str] = None,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine"
) -> Dict[str, int]:
    """
    Get ingredient frequency counts for a cuisine.

    Parameters
    ----------
    df : pd.DataFrame | RecipeDataset
        Recipe dataset
    cuisine : str, optional
        Specific cuisine to analyze, or None for all
    ingredient_col : str
        Column name for ingredients
    cuisine_col : str
        Column name for cuisine

    Returns
    -------
    Dict[str, int]
        Ingredient -> frequency mapping
    """
    data = _coerce_dataframe(df)

    if cuisine:
        if cuisine_col not in data.columns:
            raise ValueError(f"Cuisine column '{cuisine_col}' not found")
        data = data[data[cuisine_col].str.lower() == cuisine.lower()]

    if ingredient_col not in data.columns:
        raise ValueError(f"Ingredient column '{ingredient_col}' not found")

    ingredient_counts = Counter()
    for ingredients in data[ingredient_col]:
        if isinstance(ingredients, list):
            for ing in ingredients:
                if isinstance(ing, str):
                    ingredient_counts[ing.strip().lower()] += 1

    return dict(ingredient_counts)


def get_recipe_category_structure(
    df: pd.DataFrame | RecipeDataset,
    cuisine: str,
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine"
) -> List[Dict[str, int]]:
    """
    Analyze the category structure of recipes in a cuisine.

    For each recipe, count how many ingredients from each category it contains.

    Parameters
    ----------
    df : pd.DataFrame | RecipeDataset
        Recipe dataset
    cuisine : str
        Cuisine to analyze
    ingredient_col : str
        Column name for ingredients
    cuisine_col : str
        Column name for cuisine

    Returns
    -------
    List[Dict[str, int]]
        List of category count dictionaries, one per recipe
    """
    data = _coerce_dataframe(df)

    if cuisine_col not in data.columns:
        raise ValueError(f"Cuisine column '{cuisine_col}' not found")

    cuisine_data = data[data[cuisine_col].str.lower() == cuisine.lower()]

    if cuisine_data.empty:
        raise ValueError(f"No recipes found for cuisine '{cuisine}'")

    recipe_structures = []

    for ingredients in cuisine_data[ingredient_col]:
        if not isinstance(ingredients, list):
            continue

        category_counts = defaultdict(int)
        for ing in ingredients:
            if isinstance(ing, str):
                category = categorize_ingredient(ing.strip().lower())
                category_counts[category] += 1

        recipe_structures.append(dict(category_counts))

    return recipe_structures


def generate_random_cuisine(
    df: pd.DataFrame | RecipeDataset,
    cuisine: str,
    method: Literal["random", "frequency_preserved", "category_preserved", "freq_cat_preserved"],
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a randomized version of a cuisine.

    Parameters
    ----------
    df : pd.DataFrame | RecipeDataset
        Original recipe dataset
    cuisine : str
        Cuisine to randomize
    method : str
        Randomization method:
        - "random": Completely random ingredients
        - "frequency_preserved": Preserve ingredient frequencies but randomize assignments
        - "category_preserved": Preserve category structure but randomize within categories
        - "freq_cat_preserved": Preserve both frequencies and category structure
    ingredient_col : str
        Column name for ingredients
    cuisine_col : str
        Column name for cuisine
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Randomized recipe dataset
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    data = _coerce_dataframe(df)
    cuisine_data = data[data[cuisine_col].str.lower() == cuisine.lower()].copy()

    if cuisine_data.empty:
        raise ValueError(f"No recipes found for cuisine '{cuisine}'")

    # Get all unique ingredients in the dataset
    all_ingredients = set()
    for ingredients in data[ingredient_col]:
        if isinstance(ingredients, list):
            all_ingredients.update(ing.strip().lower() for ing in ingredients if isinstance(ing, str))

    all_ingredients = list(all_ingredients)

    if method == "random":
        return _generate_completely_random_cuisine(cuisine_data, all_ingredients, ingredient_col)

    elif method == "frequency_preserved":
        return _generate_frequency_preserved_cuisine(cuisine_data, all_ingredients, ingredient_col)

    elif method == "category_preserved":
        return _generate_category_preserved_cuisine(cuisine_data, all_ingredients, ingredient_col)

    elif method == "freq_cat_preserved":
        return _generate_freq_cat_preserved_cuisine(cuisine_data, all_ingredients, ingredient_col)

    else:
        raise ValueError(f"Unknown randomization method: {method}")


def _generate_completely_random_cuisine(
    cuisine_data: pd.DataFrame,
    all_ingredients: List[str],
    ingredient_col: str
) -> pd.DataFrame:
    """Generate completely random cuisine."""
    result_data = cuisine_data.copy()

    for idx, row in result_data.iterrows():
        original_ingredients = row[ingredient_col]
        if isinstance(original_ingredients, list):
            n_ingredients = len(original_ingredients)
            # Choose random ingredients
            random_ingredients = random.sample(all_ingredients, min(n_ingredients, len(all_ingredients)))
            result_data.at[idx, ingredient_col] = random_ingredients

    return result_data


def _generate_frequency_preserved_cuisine(
    cuisine_data: pd.DataFrame,
    all_ingredients: List[str],
    ingredient_col: str
) -> pd.DataFrame:
    """Generate frequency-preserved cuisine."""
    # Get ingredient frequencies from original cuisine
    freq_counts = get_cuisine_ingredient_frequencies(cuisine_data, ingredient_col=ingredient_col)

    # Create a pool of ingredients weighted by their frequencies
    ingredient_pool = []
    for ing, freq in freq_counts.items():
        ingredient_pool.extend([ing] * freq)

    result_data = cuisine_data.copy()

    for idx, row in result_data.iterrows():
        original_ingredients = row[ingredient_col]
        if isinstance(original_ingredients, list):
            n_ingredients = len(original_ingredients)
            # Sample from frequency-weighted pool
            if ingredient_pool:
                sampled_ingredients = random.sample(ingredient_pool, min(n_ingredients, len(ingredient_pool)))
                result_data.at[idx, ingredient_col] = list(set(sampled_ingredients))  # Remove duplicates

    return result_data


def _generate_category_preserved_cuisine(
    cuisine_data: pd.DataFrame,
    all_ingredients: List[str],
    ingredient_col: str
) -> pd.DataFrame:
    """Generate category-preserved cuisine."""
    # Group ingredients by category
    ingredients_by_category = defaultdict(list)
    for ing in all_ingredients:
        category = categorize_ingredient(ing)
        ingredients_by_category[category].append(ing)

    result_data = cuisine_data.copy()

    for idx, row in result_data.iterrows():
        original_ingredients = row[ingredient_col]
        if isinstance(original_ingredients, list):
            # Count categories in original recipe
            category_counts = defaultdict(int)
            for ing in original_ingredients:
                if isinstance(ing, str):
                    category = categorize_ingredient(ing.strip().lower())
                    category_counts[category] += 1

            # Generate new ingredients preserving category counts
            new_ingredients = []
            for category, count in category_counts.items():
                if category in ingredients_by_category and ingredients_by_category[category]:
                    category_ingredients = ingredients_by_category[category]
                    sampled = random.sample(category_ingredients, min(count, len(category_ingredients)))
                    new_ingredients.extend(sampled)

            result_data.at[idx, ingredient_col] = new_ingredients

    return result_data


def _generate_freq_cat_preserved_cuisine(
    cuisine_data: pd.DataFrame,
    all_ingredients: List[str],
    ingredient_col: str
) -> pd.DataFrame:
    """Generate frequency + category preserved cuisine."""
    # Get recipe category structures
    recipe_structures = get_recipe_category_structure(cuisine_data, cuisine_data[cuisine_data.columns[1]].iloc[0],
                                                     ingredient_col=ingredient_col,
                                                     cuisine_col=cuisine_data.columns[1])

    # Get ingredient frequencies by category
    freq_by_category = defaultdict(lambda: defaultdict(int))
    for ingredients in cuisine_data[ingredient_col]:
        if isinstance(ingredients, list):
            for ing in ingredients:
                if isinstance(ing, str):
                    category = categorize_ingredient(ing.strip().lower())
                    freq_by_category[category][ing.strip().lower()] += 1

    result_data = cuisine_data.copy()

    for idx, (row_idx, row) in enumerate(result_data.iterrows()):
        if idx < len(recipe_structures):
            structure = recipe_structures[idx]

            new_ingredients = []
            for category, count in structure.items():
                if category in freq_by_category:
                    category_freqs = freq_by_category[category]

                    # Create weighted pool based on frequencies
                    ingredient_pool = []
                    for ing, freq in category_freqs.items():
                        ingredient_pool.extend([ing] * freq)

                    if ingredient_pool:
                        sampled = random.sample(ingredient_pool, min(count, len(ingredient_pool)))
                        new_ingredients.extend(sampled)

            result_data.at[row_idx, ingredient_col] = new_ingredients

    return result_data


def compare_cuisine_randomizations(
    df: pd.DataFrame | RecipeDataset,
    cuisine: str,
    methods: List[str] = ["random", "frequency_preserved", "category_preserved", "freq_cat_preserved"],
    ingredient_col: str = "ingredients",
    cuisine_col: str = "cuisine",
    random_state: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple randomized versions of a cuisine for comparison.

    Parameters
    ----------
    df : pd.DataFrame | RecipeDataset
        Original recipe dataset
    cuisine : str
        Cuisine to randomize
    methods : List[str]
        List of randomization methods to generate
    ingredient_col : str
        Column name for ingredients
    cuisine_col : str
        Column name for cuisine
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping method names to randomized datasets
    """
    results = {}

    # Generate original cuisine data
    data = _coerce_dataframe(df)
    original_cuisine = data[data[cuisine_col].str.lower() == cuisine.lower()].copy()
    results["original"] = original_cuisine

    # Generate randomized versions
    for method in methods:
        try:
            randomized = generate_random_cuisine(
                df, cuisine, method,
                ingredient_col=ingredient_col,
                cuisine_col=cuisine_col,
                random_state=random_state
            )
            results[method] = randomized
        except Exception as e:
            print(f"Warning: Failed to generate {method} randomization: {e}")

    return results



def plot_frequency_rank_comparison(
    comparisons: Dict[str, pd.DataFrame],
    ingredient_col: str = "ingredients",
    figsize: Tuple[float, float] = (6, 5),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot ingredient frequency vs rank for different cuisine versions
    on log-log scales, in the style of the copy–mutate paper figure.
    """
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # colour cycle for different models
    colors = ['blue', 'black', 'red', 'green', 'orange', 'purple', 'brown']

    for i, (method, data) in enumerate(comparisons.items()):
        # Calculate ingredient frequencies (counts)
        freq_counts = get_cuisine_ingredient_frequencies(
            data,
            ingredient_col=ingredient_col
        )
        if not freq_counts:
            continue

        # Sort by frequency (descending)
        sorted_counts = sorted(freq_counts.values(), reverse=True)

        # Convert to relative frequency f(r)
        total = float(sum(sorted_counts))
        freqs = [c / total for c in sorted_counts]

        ranks = range(1, len(freqs) + 1)
        color = colors[i % len(colors)]

        # First entry: treat as "real cuisine" → blue dots only
        label = method.replace('_', ' ').title()
        if i == 0:
            ax.loglog(
                ranks,
                freqs,
                'o',
                markersize=4,
                linestyle='None',
                color='blue',
                label=label
            )
        else:
            ax.loglog(
                ranks,
                freqs,
                '-o',
                markersize=3,
                linewidth=1.0,
                color=color,
                label=label
            )

    ax.set_xlabel('Rank, r')
    ax.set_ylabel('Frequency, f(r)')
    ax.set_title('Ingredient Frequency vs Rank')
    ax.legend()
    ax.tick_params(which='both', direction='in')
    ax.grid(False)

    return ax
