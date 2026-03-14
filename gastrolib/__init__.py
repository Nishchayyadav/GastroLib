"""
GastroLib: tools for computational gastronomy and cuisine analysis.
"""

from .ingredients import summarize_ingredients
from .network import (
    build_ingredient_cooccurrence_network,
    build_cuisine_ingredient_network,
    plot_ingredient_network,
)
from .nutrition import (
    aggregate_recipe_nutrition_from_ingredients,
)
from .randomization import (
    compare_cuisine_randomizations,
    plot_frequency_rank_comparison,
    generate_random_cuisine,
    get_cuisine_ingredient_frequencies,
    get_recipe_category_structure,
    categorize_ingredient,
)
from .stats import (
    compute_recipe_size_distribution,
    compute_ingredient_popularity,
    analyze_category_composition,
    plot_category_composition_pie,
    plot_cuisine_similarity_heatmap
)
from .itemsets import (
    mine_frequent_itemsets,
)
from .pairing import (
    analyze_food_pairing,
    compute_pairing_matrix,
    plot_pairing_heatmap,
)
from .similarity import (
    compute_cuisine_similarity,
)
from .data import (
    RecipeDataset,
    load_sample_cuisine_dataset,
)

__all__ = [
    # Ingredient stats
    "summarize_ingredients",
    # Stats utilities
    "compute_recipe_size_distribution",
    "compute_ingredient_popularity",
    "analyze_category_composition",
    "plot_category_composition_pie",
    "plot_cuisine_similarity_heatmap",
    # Networks
    "build_ingredient_cooccurrence_network",
    "build_cuisine_ingredient_network",
    "plot_ingredient_network",
    # Nutrition
    "aggregate_recipe_nutrition_from_ingredients",
    # Itemsets
    "mine_frequent_itemsets",
    # Pairing
    "analyze_food_pairing",
    "compute_pairing_matrix",
    "plot_pairing_heatmap",
    # Similarity
    "compute_cuisine_similarity",
    # Randomization & Comparison
    "compare_cuisine_randomizations",
    "plot_frequency_rank_comparison",
    "generate_random_cuisine",
    "get_cuisine_ingredient_frequencies",
    "get_recipe_category_structure",
    "categorize_ingredient",
    # Dataset helpers
    "RecipeDataset",
    "load_sample_cuisine_dataset",
]


