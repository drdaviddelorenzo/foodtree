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

`--method upgma|nj|both` (tree algorithm, default `upgma`), `--distance braycurtis|unifrac|aitchison|euclidean|jaccard`, `--level ingredient|source|group` (compare ingredients as listed or collapsed in the ingredient tree; output names then carry the level), `--outgroup` (a reference product: highlighted in UPGMA, used as root in NJ), `--min-pct` (ignore ingredients below this percentage), `--name-search` (look up products without a valid barcode by name; the match is reported and recorded), `--manual` (a CSV of products entered by hand: `FoodItem,Ingredient,Percentage[,Category,Source]`), `--allow-missing` (build the tree without products that have no usable data; by default the run stops, and with this option they are listed in `foods-omitted.txt` and `run-config.txt`), `--moisture LO-HI` (water of the finished products, g per 100 g; by default it is left to the nutrition table), `--allow-inconsistent` (keep products whose label no composition satisfies, using the least-violating one; by default the run stops and says which constraints conflict).

`--dir` is the analysis's working folder: a render overwrites files there that have its output names, and says so when the previous render did not list them. Output names carry the distance (for example `crisps-heatmap-braycurtis.png`), so renders with different distances sit side by side. Each render publishes its files only once all of them have been produced (if publishing fails, the previous files are put back; if some cannot be, their copies are kept in a `.foodtree-recovery-<time>` folder and the error says where), lists them in `<name>-<distance>-outputs.txt`, and removes the files listed by the previous render of the same analysis and distance that it did not write again; no other file is deleted.

- **Bray-Curtis** (default): the share of the composition two products do not have in common.
- **UniFrac** (weighted, normalised): like Bray-Curtis, but over the ingredient tree, so related ingredients (sunflower oil and an unspecified vegetable oil) count as close and unrelated ones (an oil and salt) as far.
- **Aitchison**: log-ratio distance for compositional data.
- **Euclidean**: the largest ingredient dominates.
- **Jaccard**: presence or absence only.

## The ingredient tree

`data/ingredient_tree.csv` relates ingredients to one another. Each ingredient sits under the plant or material it comes from (potato flour under potato products, as in the Open Food Facts taxonomy, which gives it the parents *flour* and *potato*), and these under broad groups by function: starchy bases, fats, salts, seasonings, additives, water, and groups for sweeteners, dairy, protein foods and other categories. An ingredient that is not in the file is placed under the node for its composition class or, failing that, its food category; what is left goes under "other", and a render warns when more than 5% of a product ends up there, since all of it is compared as one undifferentiated group. The ingredient tree is not a phylogeny: it is a curated classification of ingredients, with every branch given the same length, and its grouping choices (and the residual water, which every product shares) shape the distances. The tree is a choice, not a fact: grouping by form instead (all flours and starches together) is just as defensible and gives a different picture, and the file can be edited to try it.

`--level source` or `--level group` collapses ingredients to those nodes before computing distances (`--min-pct` then applies to the collapsed sources or groups), for example to compare products by how much potato, maize, oil and salt they contain whatever the form. UniFrac (Lozupone et al. 2007, *Applied and Environmental Microbiology* 73:1576-1585) is computed with every branch of length 1. Processing is kept apart where it changes composition: dehydrated potato is a different leaf from fresh potato.

## How percentages are estimated

A label gives three kinds of evidence, on two different bases. The order of the ingredient list and any declared percentage refer to the weight of each ingredient as used, including its water (Regulation (EU) No 1169/2011, Article 18 and Annex VIII); ingredients under 2% of the finished product may be listed in any order after the others (Annex VII, Part A, point 6). The nutrition table refers to the finished product, and its values may differ from the true ones within the tolerances of the European Commission guidance on nutrient values declared on a label (December 2012).

`estimate.py` finds every composition compatible with all three: grams of each ingredient used per 100 g of product, the dry matter it leaves in the finished product, and the residual water. Each ingredient's water, fat, carbohydrate, protein and salt come from ranges in `data/ingredient_composition.csv`, built from USDA FoodData Central (SR Legacy, referenced by FDC id) or, where there is no entry, from a stated assumption. `foods.csv` gives, for each ingredient on a finished-product basis, `PctLow` and `PctHigh` (outer bounds: every compatible composition lies within them, though they can be somewhat wider than the compatible set) and `Percentage`, a weighted mean of sampled recipes. Sampling has two levels: each ingredient's composition is drawn uniformly within its range, and for each composition 5 recipes are drawn uniformly among those the label allows, every allowed ingredient ordering included. A composition is kept only if those 5 valid recipes turn up among its first 200 candidates; this bounds run time but under-represents compositions whose valid recipes are a small part of the candidates. This is a stated prior, not a uniform distribution over all compatible compositions and recipes taken together; where a label says little, `Percentage` depends on it and the bounds are the robust result. Each recipe is weighted by how close its nutrients are to the declared values, with the tolerance band as two standard deviations on each side. Residual water appears as its own row. For the under-2% exception, each ingredient's share of the finished product counts its dry matter plus a share of the residual water proportional to the water it brought; this allocation is an assumption, since a label does not say which ingredient keeps the water. Sampling uses a fixed seed, so results are reproducible; a product with few effective samples is reported.

`Source` says whether the quantity used was `declared` on the label or `label-constrained`. A label that no composition satisfies stops the run with the label constraints in conflict (declared percentages, order, nutrition values or moisture), unless `--allow-inconsistent` is given (`Source` is then `inconsistent-label`). If the bounds admit a label but no compatible recipe is sampled, the run stops with a different message: the label may be fine and the ingredient classes should be checked. Products entered with `--manual` keep their values (`Source` `manual`).

## How far a tree can be trusted

`normalise` also writes `foods-samples.csv`: for each product, 200 compositions resampled from those compatible with its label (products entered by hand keep their values). `render` rebuilds the tree from 100 of them (`--support N`, 0 to skip) and labels each branch with the percentage of replicate trees that contain the same group of products; the numbers appear on the plots and as internal labels in the Newick and JSON files. This is in the spirit of the phylogenetic bootstrap, but it resamples what the labels leave undetermined, not characters: high support means a grouping does not depend on the unknown percentages, not that the ingredient tree, the distance or the tree method are right. Two products can be grouped with full support for a poor reason (for example, sharing an ingredient name under Bray-Curtis).

Each render also writes `<name>-diagnostics-<distance>.txt` with the cophenetic correlation of the UPGMA tree (how well the tree preserves the distances; well below about 0.8 it distorts them), and a principal coordinates analysis (`<name>-pcoa-<distance>.png` and `.csv`), which places the products in a plane without forcing a tree, with the share of variation on each axis.

## Caveats

Ingredient lists are what labels declare, and Open Food Facts is edited by its users: check the inputs before drawing conclusions. Composition ranges are generic and an ingredient whose class is unknown is left unconstrained, so bounds can be wide. Data licences: Open Food Facts (ODbL), USDA FoodData Central (public domain).

## Reproducing and testing

- **Without a connection.** `python pipeline.py run --snapshot examples/crisps/raw_products.json --name "Crisps" --dir analyses/crisps/ --moisture 1-4` reuses the saved records instead of fetching them, so a whole class can run the example without calling Open Food Facts.
- **Same versions.** `requirements.txt` gives minimum versions; `requirements-lock.txt` pins the full environment the code was tested in (Python 3.12.9 on macOS, every package including dependencies). Use it for a course; on other platforms, run the tests to check that the results hold.
- **Determinism.** Estimation and branch support use fixed seeds: the same inputs and versions give the same outputs.
- **Tests.** `pip install pytest` and run `pytest` from the repository folder. `tests/test_example.py` re-runs the crisps example (about a minute) and checks its bounds and that the UniFrac tree recovers the maize and crisp families with high support.

## Licence

Code: MIT, see `LICENSE`. Data in `examples/`: see the README in each example folder.
