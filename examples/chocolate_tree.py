"""Chocolate example: a tree from the label profile of each bar.

The profile has three numbers per bar: the declared cocoa and the total sugars of the
nutrition table (both label figures, which can overlap slightly because cocoa carries
some sugars of its own) and a remainder calculated as 100 minus both, never below zero.
The remainder is not a measured share of the bar.

Each row of the CSV needs barcode and product_name. A row with an empty barcode must give
cocoa and sugars in the CSV itself (label figures typed by hand); nothing is downloaded for it.
The script needs all the products of the file: if one cannot be obtained, it says which and
stops without drawing a tree.

    python examples/chocolate_tree.py examples/products_chocolate.csv
"""

import csv
import sys
import time

import numpy as np
import requests
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

URL = "https://world.openfoodfacts.org/api/v2/product/{}"
HEADERS = {"User-Agent": "FoodTree/2.0 (foodtree-project; open-source)"}


def fetch(barcode):
    """Product record from Open Food Facts, or None after three failed attempts."""
    for attempt in range(3):
        try:
            r = requests.get(URL.format(barcode), params={"fields": "ingredients,nutriments"},
                             headers=HEADERS, timeout=30)
            if r.ok:
                return r.json().get("product") or {}
        except (requests.RequestException, ValueError):
            pass  # timeout, connection error or invalid JSON: try again
        time.sleep(3 * (attempt + 1))
    return None


def declared_cocoa(product):
    """Cocoa percentage recorded on the product page, or the one declared in the ingredient list."""
    value = product.get("nutriments", {}).get("cocoa_100g")
    if value is not None:
        return float(value)
    for ing in product.get("ingredients", []):
        if ing.get("id") in ("en:cocoa", "en:cocoa-solids") and ing.get("percent") is not None:
            return float(ing["percent"])


def quote(label):
    """Newick label in single quotes, with any single quote inside doubled."""
    return "'" + label.replace("'", "''") + "'"


def newick(node, names, parent_height):
    height = node.dist / 2
    if node.is_leaf():
        return f"{quote(names[node.id])}:{parent_height - height:.4f}"
    left, right = (newick(c, names, height) for c in (node.left, node.right))
    return f"({left},{right}):{parent_height - height:.4f}"


def read_products(path):
    """Return (names, rows, failed) with rows of cocoa, sugars and remainder per product."""
    names, rows, failed = [], [], []
    for item in csv.DictReader(open(path, encoding="utf-8")):
        name, barcode = item["product_name"], (item.get("barcode") or "").strip()
        if barcode:
            product = fetch(barcode)
            if product is None:
                failed.append(f"{name} ({barcode}): could not be downloaded")
                continue
            cocoa = declared_cocoa(product)
            sugars = product.get("nutriments", {}).get("sugars_100g")
        else:
            cocoa, sugars = (item.get("cocoa") or None), (item.get("sugars") or None)
        if cocoa is None or sugars is None:
            failed.append(f"{name} ({barcode or 'no barcode'}): no declared cocoa or sugars")
            continue
        cocoa, sugars = float(cocoa), float(sugars)
        names.append(name)
        rows.append([cocoa, sugars, max(100 - cocoa - sugars, 0)])
        print(f"{name:28s} cocoa {cocoa:5.1f}  sugars {sugars:5.1f}  remainder {rows[-1][2]:5.1f}")
        if barcode:
            time.sleep(0.7)
    return names, rows, failed


if __name__ == "__main__":
    names, rows, failed = read_products(sys.argv[1])
    if failed:
        print("The tree needs every product in the file. Missing:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    m = np.array(rows)
    dist = np.array([[np.abs(a - b).sum() / (a + b).sum() for b in m] for a in m])
    root = to_tree(linkage(squareform(dist, checks=False), method="average"))
    print(newick(root, names, root.dist / 2).rsplit(":", 1)[0] + ";")
