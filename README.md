# FoodTree

A family tree for foods. FoodTree applies the method biologists use to relate species (phylogenetic trees) to food products: instead of DNA sequences it compares **ingredient lists with their percentages**, and shows which products are close relatives and which only look alike on the shelf.

## How it works

1. **fetch** (`fetch.py`, `search.py`): collects ingredients and nutrition data for each product (barcode and name) from [Open Food Facts](https://world.openfoodfacts.org/) and, optionally, [USDA FoodData Central](https://fdc.nal.usda.gov/).
2. **normalise** (`normalise.py`): turns each product into ingredients with percentages by weight, harmonising names and estimating missing percentages.
3. **render** (`render.py`): computes distances between products and draws a UPGMA tree (rooted) and a Neighbor Joining tree (unrooted), plus ingredient heatmaps.

`pipeline.py` runs the three stages.

## Quick start

```bash
git clone https://github.com/drdaviddelorenzo/foodtree.git && cd foodtree
pip install -r requirements.txt            # Python 3.11+
export USDA_API_KEY=your_key               # optional; without it USDA is limited to DEMO_KEY
python pipeline.py run --input examples/products_crisps.csv --name "Crisps" --dir analyses/crisps/
```

Your input is a CSV with `barcode,product_name` (see `examples/products_crisps.csv`). `search.py "energy drinks"` helps to build one. Outputs (`raw_products.json`, `foods.csv`, tree and heatmap PNGs) are written to `--dir`.

The tree data is exported too, so you can redraw the trees in your own style: Newick (`*-tree-*.nwk`), JSON with branch lengths and depths (`*-tree-*.json`) and the distance matrix (`*-distance-*.csv`). Use `--no-export` to skip them.

## Chocolate example

`examples/chocolate_tree.py` builds a tree of ten chocolate bars from a label profile of three numbers per bar: the declared cocoa and the total sugars of the nutrition table (Open Food Facts), which can overlap slightly because cocoa carries some sugars of its own, and a remainder calculated as 100 minus both, which is not a measured share. It does not use the ingredient estimator of the main pipeline.

```bash
python examples/chocolate_tree.py examples/products_chocolate.csv
```

It prints the profile of each bar and the tree in Newick format. A row with an empty barcode (the two store brands) takes `cocoa` and `sugars` from the CSV and downloads nothing. The script needs all ten products: if one cannot be obtained, it lists which and exits with an error without drawing a tree.

## Options

`--method upgma|nj` (tree algorithm), `--distance braycurtis|aitchison|euclidean|jaccard`, `--outgroup`, `--min-pct` (ignore ingredients below this percentage).

- **Bray-Curtis** (default): compares proportions; a good general choice.
- **Aitchison**: log-ratio distance for compositional data.
- **Euclidean**: the largest ingredient dominates.
- **Jaccard**: presence or absence only.

## Caveats

Ingredient lists are what labels declare. Percentages are label values when declared; for crisps and similar snacks, oils and salt come from the nutrition table (`label+nutrition`) and the rest is shared by rank; otherwise they are estimated. Check the inputs before drawing conclusions. Open Food Facts is edited by its users. Data licences: Open Food Facts (ODbL), USDA FoodData Central (public domain).

## Licence

MIT. See `LICENSE`.
