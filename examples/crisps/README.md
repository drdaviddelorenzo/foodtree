# Crisps example

Eleven snacks sold in Spain, the United Kingdom and France: five fried potato crisps, three products made from a potato-based dough, and three maize snacks. "Marca blanca" is a supermarket own-brand product with three ingredients (potato, sunflower oil, salt), used as the simplest reference.

- `products.csv`: barcodes and the names used in the trees.
- `raw_products.json`: the Open Food Facts records as retrieved on 2026-11-06 (the `retrieved_at` field of each product). Labels on Open Food Facts are entered by its users and can change; this snapshot keeps the example reproducible. The brand and product-name fields of the own-brand reference are left empty; its barcode identifies the record, and no other field was changed.

Run the analysis from the snapshot, without contacting Open Food Facts:

```bash
python pipeline.py run --snapshot examples/crisps/raw_products.json --name "Crisps" \
    --dir analyses/crisps/ --moisture 1-4 --min-pct 0 --distance unifrac
```

Use `--distance braycurtis` to compare with the published post, and `--method both` to add the Neighbor Joining tree.

Or fetch the current records instead: `python pipeline.py run --input examples/crisps/products.csv --name "Crisps" --dir analyses/crisps/ --moisture 1-4`.

`--moisture 1-4` sets the water of the finished snacks to 1-4 g per 100 g, around the values USDA FoodData Central gives for salted potato chips (1.86 g, FDC 169677) and salted tortilla chips (2.64 g, FDC 167558). Without it, water is left to the nutrition table and the bounds are wider.

## Data licence

`raw_products.json` contains data from Open Food Facts (https://world.openfoodfacts.org), made available under the Open Database License (ODbL) 1.0: https://opendatacommons.org/licenses/odbl/1-0/. Individual contents are under the Database Contents License: https://opendatacommons.org/licenses/dbcl/1-0/. The MIT licence of this repository covers the code only, not this file.
