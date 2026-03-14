## GastroLib

GastroLib is a Python library designed to support research in **Computational Gastronomy**.
It enables structured analysis of global cuisines through statistical, network-based, and
generative methods.

### Key Capabilities

- **Recipe statistics**: Ingredient frequencies, recipe-size distribution, category
  composition, and cuisine-specific profiles.
- **Frequent pattern mining**: Apriori-based frequent itemsets and food pairing analysis.
- **Networks & similarity**: Build co-occurrence networks, compute cuisine similarity matrices,
  and analyze ingredient relationships.
- **Nutritional analysis**: Aggregate nutrition via ingredient lookup tables.
- **Cuisine analysis**: Compare cuisines through randomization and frequency analysis.

By combining data analysis with creative modeling, GastroLib aims to uncover **hidden patterns
in culinary traditions** and help users explore cuisine scientifically.

---

### Installation

To use this project from GitHub, clone the repository and install it locally:

```bash
git clone https://github.com/Nishchayyadav/GastroLib.git
cd GastroLib
pip install -e .
```

If you only want the dependencies without installing the package in editable mode:

```bash
pip install -r requirements.txt
```

---

### Data Expectations & RecipeDataset

All modules operate on the standardized schema enforced by `RecipeDataset`. Minimum required
columns:

| Column                | Type        | Description                                                 |
|-----------------------|-------------|-------------------------------------------------------------|
| `recipe_id`           | str / int   | Unique identifier per recipe                                |
| `recipe_name`         | str         | Human-readable title                                        |
| `cuisine`             | str         | Cuisine label (`"indian"`, `"mexican"`, …)                  |
| `ingredients`         | list[str]   | Cleaned list of ingredient tokens                           |
| `category` (optional) | str         | e.g. `"veg"`, `"dessert"`, `"main"`                         |
| `methods` (optional)  | list[str]   | Cooking techniques such as `["fry", "boil"]`                |
| `ingredient_quantities` (optional) | dict[str, float] | Quantity per ingredient for nutritional analysis |

`RecipeDataset` can be constructed from CSV/Parquet or an existing DataFrame; it normalizes
the required columns (lowercases string lists, parses JSON-like strings into Python lists,
ensures consistent dtypes). Most GastroLib functions accept either a raw `DataFrame` following
this schema or a `RecipeDataset` instance.

Best practices before analysis:

- Normalize ingredient text (lowercase, singularize, strip punctuation).
- Store cooking categories/methods whenever available—they unlock richer analysis.
- If nutrition is important, provide an ingredient-level lookup table with calories/macros per unit.

---

### Library Architecture at a Glance

```
gastrolib/
├─ data.py          # RecipeDataset, loaders, schema helpers
├─ stats.py         # recipe-size distribution, ingredient popularity, category mix
├─ itemsets.py      # Apriori frequent itemsets
├─ pairing.py       # ingredient pairing scores, matrices, heatmaps
├─ network.py       # co-occurrence network builder + visualizations
├─ similarity.py    # cuisine vectors, similarity matrices
├─ nutrition.py     # ingredient lookup aggregation
├─ randomization.py # cuisine randomization and comparison utilities
└─ __init__.py      # curated, high-level API surface
```

Each module provides a single public entry point that returns structured dictionaries (stats,
tables, figures, network objects) so you can compose higher-level workflows easily.

---

### Typical Analysis Workflow

1. **Load & normalize** – wrap your raw DataFrame in `RecipeDataset`.
2. **Explore structure** – run `compute_recipe_size_distribution`, `compute_ingredient_popularity`,
   and `analyze_category_composition` to understand dataset coverage.
3. **Discover patterns** – use `mine_frequent_itemsets` and `analyze_food_pairing` to surface
   ingredient combinations; visualize via `plot_pairing_heatmap`.
4. **Build networks** – `build_ingredient_cooccurrence_network` creates co-occurrence networks;
   visualize with `plot_ingredient_network`.
5. **Compare cuisines** – compute similarity matrices with `compute_cuisine_similarity`;
   compare cuisines through randomization using `compare_cuisine_randomizations`.
6. **Assess nutrition** – use `aggregate_recipe_nutrition_from_ingredients` with a nutrition
   lookup table to quantify macros per recipe.
7. **Analyze randomization** – generate and compare randomized cuisine variants to understand
   what makes real cuisines unique.

`notebooks/demo.ipynb` implements this workflow end-to-end with comprehensive examples.

---

### High-Level API Overview

- **Dataset utilities**
  - `RecipeDataset`, `load_sample_cuisine_dataset`
- **Stats & composition**
  - `compute_recipe_size_distribution`, `compute_ingredient_popularity`,
    `analyze_category_composition`, `plot_category_composition_pie`,
    `summarize_ingredients`
- **Frequent patterns & pairing**
  - `mine_frequent_itemsets`, `analyze_food_pairing`,
    `compute_pairing_matrix`, `plot_pairing_heatmap`
- **Networks & similarity**
  - `build_ingredient_cooccurrence_network`, `build_cuisine_ingredient_network`,
    `plot_ingredient_network`, `compute_cuisine_similarity`,
    `plot_cuisine_similarity_heatmap`
- **Nutrition**
  - `aggregate_recipe_nutrition_from_ingredients`
- **Randomization & comparison**
  - `compare_cuisine_randomizations`, `plot_frequency_rank_comparison`,
    `generate_random_cuisine`, `get_cuisine_ingredient_frequencies`,
    `get_recipe_category_structure`, `categorize_ingredient`

Full docstrings are available in the source code; you can inspect them interactively with
`help()` in Python.

---

### Demos & Notebooks

- `notebooks/demo.ipynb` – comprehensive interactive walkthrough covering all
  GastroLib functions with statistics, patterns, networks, nutrition, pairing analysis,
  cuisine similarity, and randomization utilities.

---

### License

This project is licensed under the **MIT License**.