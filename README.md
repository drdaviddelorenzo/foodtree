# FoodTree

Similarity trees for foods. FoodTree compares food products by their **ingredient lists with their percentages** and groups them with the distance-tree methods used in biology (UPGMA and Neighbor Joining), showing which products have similar recipes and which only look alike on the shelf.

The trees are phenetic: they group products by present-day similarity of composition. Products do not descend from one another, so branches do not mean shared ancestry, internal nodes are not ancestors and branch lengths are dissimilarities, not time.

## How it works

1. **fetch** (`fetch.py`, `search.py`): collects ingredients and nutrition data for each product from [Open Food Facts](https://world.openfoodfacts.org/) by barcode and, optionally, [USDA FoodData Central](https://fdc.nal.usda.gov/).
2. **normalise** (`normalise.py`, `estimate.py`): turns each product into ingredients with percentages by weight of the finished product, harmonising names and estimating the percentages the label does not declare (see *How percentages are estimated*).
3. **render** (`render.py`): computes distances between products and draws a UPGMA tree (rooted) or a Neighbor Joining tree (unrooted), plus ingredient heatmaps.

`pipeline.py` runs the three stages.

## Quick start

```bash
git clone https://github.com/drdaviddelorenzo/foodtree.git && cd foodtree
pip install -r requirements.txt            # Python 3.11+
export USDA_API_KEY=your_key               # optional; without it USDA is skipped
python pipeline.py run --input examples/crisps/products.csv --name "Crisps" --dir analyses/crisps/ --moisture 1-4
```

Your input is a CSV with `barcode,product_name` (see `examples/crisps/`, which also holds a frozen snapshot of the data to reproduce the example offline). `search.py "energy drinks"` helps to build one. Outputs (`raw_products.json`, `foods.csv`, tree and heatmap PNGs) are written to `--dir`.

The tree data is exported too, so you can redraw the trees in your own style: Newick (`*-tree-*.nwk`), JSON with branch lengths and depths (`*-tree-*.json`) and the distance matrix (`*-distance-*.csv`). Use `--no-export` to skip them.

## Options

`--method upgma|nj|both` (tree algorithm, default `upgma`), `--distance braycurtis|aitchison|euclidean|jaccard`, `--outgroup` (a reference product: highlighted in UPGMA, used as root in NJ), `--min-pct` (ignore ingredients below this percentage), `--name-search` (look up products without a valid barcode by name; the match is reported and recorded), `--manual` (a CSV of products entered by hand: `FoodItem,Ingredient,Percentage[,Category,Source]`), `--allow-missing` (build the tree without products that have no usable data; by default the run stops, and with this option they are listed in `foods-omitted.txt` and `run-config.txt`), `--moisture LO-HI` (water of the finished products, g per 100 g; by default it is left to the nutrition table), `--allow-inconsistent` (keep products whose label no composition satisfies, using the least-violating one; by default the run stops and says which constraints conflict).

`--dir` is the analysis's working folder: a render overwrites files there that have its output names, and says so when the previous render did not list them. Output names carry the distance (for example `crisps-heatmap-braycurtis.png`), so renders with different distances sit side by side. Each render publishes its files only once all of them have been produced (if publishing fails, the previous files are put back; if some cannot be, their copies are kept in a `.foodtree-recovery-<time>` folder and the error says where), lists them in `<name>-<distance>-outputs.txt`, and removes the files listed by the previous render of the same analysis and distance that it did not write again; no other file is deleted.

- **Bray-Curtis** (default): the share of the composition two products do not have in common.
- **Aitchison**: log-ratio distance for compositional data.
- **Euclidean**: the largest ingredient dominates.
- **Jaccard**: presence or absence only.

## How percentages are estimated

A label gives three kinds of evidence, on two different bases. The order of the ingredient list and any declared percentage refer to the weight of each ingredient as used, including its water (Regulation (EU) No 1169/2011, Article 18 and Annex VIII); ingredients under 2% of the finished product may be listed in any order after the others (Annex VII, Part A, point 6). The nutrition table refers to the finished product, and its values may differ from the true ones within the tolerances of the European Commission guidance on nutrient values declared on a label (December 2012).

`estimate.py` finds every composition compatible with all three: grams of each ingredient used per 100 g of product, the dry matter it leaves in the finished product, and the residual water. Each ingredient's water, fat, carbohydrate, protein and salt come from ranges in `data/ingredient_composition.csv`, built from USDA FoodData Central (SR Legacy, referenced by FDC id) or, where there is no entry, from a stated assumption. `foods.csv` gives, for each ingredient on a finished-product basis, `PctLow` and `PctHigh` (outer bounds: every compatible composition lies within them, though they can be somewhat wider than the compatible set) and `Percentage`, a weighted mean of sampled recipes. Sampling has two levels: each ingredient's composition is drawn uniformly within its range, and for each composition 5 recipes are drawn uniformly among those the label allows, every allowed ingredient ordering included. A composition is kept only if those 5 valid recipes turn up among its first 200 candidates; this bounds run time but under-represents compositions whose valid recipes are a small part of the candidates. This is a stated prior, not a uniform distribution over all compatible compositions and recipes taken together; where a label says little, `Percentage` depends on it and the bounds are the robust result. Each recipe is weighted by how close its nutrients are to the declared values, with the tolerance band as two standard deviations on each side. Residual water appears as its own row. For the under-2% exception, each ingredient's share of the finished product counts its dry matter plus a share of the residual water proportional to the water it brought; this allocation is an assumption, since a label does not say which ingredient keeps the water. Sampling uses a fixed seed, so results are reproducible; a product with few effective samples is reported.

`Source` says whether the quantity used was `declared` on the label or `label-constrained`. A label that no composition satisfies stops the run with the label constraints in conflict (declared percentages, order, nutrition values or moisture), unless `--allow-inconsistent` is given (`Source` is then `inconsistent-label`). If the bounds admit a label but no compatible recipe is sampled, the run stops with a different message: the label may be fine and the ingredient classes should be checked. Products entered with `--manual` keep their values (`Source` `manual`).

## Caveats

Ingredient lists are what labels declare, and Open Food Facts is edited by its users: check the inputs before drawing conclusions. Composition ranges are generic and an ingredient whose class is unknown is left unconstrained, so bounds can be wide. Data licences: Open Food Facts (ODbL), USDA FoodData Central (public domain).

## Licence

Code: MIT, see `LICENSE`. Data in `examples/`: see the README in each example folder.
