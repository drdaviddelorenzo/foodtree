"""Stage 1: fetch ingredient and nutrient data from Open Food Facts (and USDA, optionally).

    python fetch.py --input products.csv --output raw_products.json [--name-search]

products.csv has two columns: barcode, product_name. Products are looked up by barcode;
a search by name is only made with --name-search, and its match is recorded and reported.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

import requests


OFF_BARCODE_URL  = "https://world.openfoodfacts.org/api/v2/product/{barcode}"
OFF_SEARCH_URL   = "https://world.openfoodfacts.org/cgi/search.pl"
USDA_SEARCH_URL  = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_DETAIL_URL  = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

OFF_FIELDS = (
    "code,product_name,brands,ingredients,ingredients_tags,ingredients_text,"
    "ingredients_text_en,nutriments,quantity"
)

HEADERS = {"User-Agent": "FoodTree/2.0 (foodtree-project; open-source)"}

USDA_API_KEY = os.environ.get("USDA_API_KEY", "DEMO_KEY")

USDA_NUTRIENT_MAP = {
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbohydrates_g",
    1079: "fiber_g",
    2000: "sugar_g",
    1093: "sodium_g",
    1087: "calcium_mg",
    1162: "vitamin_c_mg",
    1008: "energy_kcal",
}


def _off_extract(product: dict, product_name: str, matched_by: str) -> dict:
    """Extract the fields we need from an OFF product dict."""
    n = product.get("nutriments", {})
    return {
        "product_name": product_name,
        "matched_by": matched_by,
        "off_code": product.get("code", ""),
        "off_product_name": product.get("product_name", ""),
        "off_brands": product.get("brands", ""),
        "ingredients_raw_text": (
            product.get("ingredients_text_en")
            or product.get("ingredients_text", "")
        ),
        "ingredients_tags": product.get("ingredients_tags", []),
        "ingredients": product.get("ingredients", []),
        "nutrients_per_100g": {
            "energy_kcal":      n.get("energy-kcal_100g"),
            "protein_g":        n.get("proteins_100g"),
            "fat_g":            n.get("fat_100g"),
            "saturated_fat_g":  n.get("saturated-fat_100g"),
            "carbohydrates_g":  n.get("carbohydrates_100g"),
            "sugar_g":          n.get("sugars_100g"),
            "fiber_g":          n.get("fiber_100g"),
            "sodium_g":         n.get("sodium_100g"),
            "salt_g":           n.get("salt_100g"),
        },
        "source": "openfoodfacts",
    }


def _off_get(url: str, params: dict, retries: int = 2) -> requests.Response | None:
    """GET with silent 404 handling and one retry on 5xx errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2)
            continue
    return None


def fetch_off_by_barcode(barcode: str, product_name: str) -> dict | None:
    url = OFF_BARCODE_URL.format(barcode=barcode.strip())
    resp = _off_get(url, {"fields": OFF_FIELDS})
    if resp is None:
        return None
    data = resp.json()
    if data.get("status") == 1 and "product" in data:
        p = data["product"]
        if p.get("ingredients") or p.get("ingredients_tags") or p.get("ingredients_text"):
            return _off_extract(p, product_name, "barcode")
    return None


def fetch_off_by_name(product_name: str) -> dict | None:
    params = {
        "search_terms": product_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 5,
        "fields": OFF_FIELDS,
    }
    resp = _off_get(OFF_SEARCH_URL, params)
    if resp is None:
        return None
    products = resp.json().get("products", [])
    for p in products:
        if p.get("ingredients") or p.get("ingredients_tags"):
            return _off_extract(p, product_name, "name_search")
    return None


def _usda_extract(food: dict, product_name: str) -> dict:
    """Extract ingredients text and nutrient data from a USDA food dict."""
    nutrients = {}
    for n in food.get("foodNutrients", []):
        nid = n.get("nutrientId")
        if nid in USDA_NUTRIENT_MAP:
            nutrients[USDA_NUTRIENT_MAP[nid]] = n.get("value")

    return {
        "product_name": product_name,
        "usda_description": food.get("description", ""),
        "usda_brand": food.get("brandOwner", "") or food.get("brandName", ""),
        "ingredients_raw_text": food.get("ingredients", ""),
        "nutrients_per_100g": nutrients,
        "source": "usda",
    }


def _clean_usda_query(name: str) -> str:
    """Strip characters that cause USDA API 400 errors."""
    import re
    name = re.sub(r"\s*\(.*?\)", "", name)
    if "," in name:
        parts = [p.strip() for p in name.split(",")]
        name = max(parts, key=len)
    name = re.sub(r"[^\w\s\-&]", " ", name)
    return " ".join(name.split())


def _usda_search(query: str, api_key: str, barcode: str | None = None) -> dict | None:
    """Search USDA by barcode (preferred) or product name."""
    params: dict = {
        "api_key": api_key,
        "dataType": ["Branded"],
        "pageSize": 5,
    }

    if barcode:
        params["query"] = barcode
        try:
            resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            foods = resp.json().get("foods", [])
            for f in foods:
                gtin = f.get("gtinUpc", "")
                if gtin and barcode.lstrip("0") in gtin:
                    return f
        except requests.RequestException as e:
            print(f"    [USDA barcode] {e}", file=sys.stderr)

    clean_query = _clean_usda_query(query)
    params["query"] = clean_query
    try:
        resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        foods = resp.json().get("foods", [])
        if foods:
            return foods[0]
    except requests.RequestException as e:
        print(f"    [USDA name] {e}", file=sys.stderr)

    return None


def _usda_relevant(food: dict, product_name: str) -> bool:
    """Basic relevance check: at least one meaningful word from the product name"""
    STOPWORDS = {"a", "an", "the", "of", "with", "and", "or", "for",
                 "in", "on", "at", "to", "by", "fish", "food"}
    query_words = {w.lower() for w in product_name.split()
                   if len(w) > 2 and w.lower() not in STOPWORDS}
    if not query_words:
        return True
    description = (food.get("description", "") + " " +
                   food.get("brandOwner", "")).lower()
    return any(w in description for w in query_words)


def fetch_usda(barcode: str, product_name: str, api_key: str) -> dict | None:
    if api_key == "DEMO_KEY":
        return None
    food = _usda_search(product_name, api_key, barcode=barcode or None)
    if food and _usda_relevant(food, product_name):
        return _usda_extract(food, product_name)
    if food:
        print(f"    [USDA] rejected irrelevant match: {food.get('description', '')[:50]}")
    return None


def fetch_product(barcode: str, product_name: str, usda_key: str,
                  name_search: bool = False) -> dict:
    result = {
        "product_name": product_name,
        "barcode": barcode,
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "off_data": None,
        "usda_data": None,
    }

    off_data = None
    if barcode:
        print(f"  [OFF] barcode {barcode}...", end=" ", flush=True)
        off_data = fetch_off_by_barcode(barcode, product_name)
        if off_data is None:
            print("✗ not found")

    if off_data is None and name_search:
        print(f"  [OFF] name search '{product_name}'...", end=" ", flush=True)
        off_data = fetch_off_by_name(product_name)
        if off_data:
            print(f"matched {off_data['off_code'] or '(no code)'} "
                  f"'{off_data['off_brands']} {off_data['off_product_name']}'. "
                  "Check that this is the intended product.", end=" ")
    elif off_data is None:
        reason = "barcode not found" if barcode else "no barcode"
        print(f"  [OFF] {reason}: skipped (use --name-search, or add it with --manual "
              "at the normalise stage)")

    if off_data:
        n_ings = len(off_data.get("ingredients", []))
        n_tags = len(off_data.get("ingredients_tags", []))
        print(f"✓  ({n_ings} ingredients, {n_tags} tags)")
        result["off_data"] = off_data
    elif name_search:
        print("✗ not found")

    if usda_key and usda_key != "DEMO_KEY":
        print(f"  [USDA] searching '{product_name}'...", end=" ", flush=True)
        usda_data = fetch_usda(barcode, product_name, usda_key)
        if usda_data:
            print(f"✓  (brand: {usda_data['usda_brand'] or 'unknown'})")
            result["usda_data"] = usda_data
        else:
            print(f"✗ not found")
    else:
        result["usda_data"] = None

    return result


def fetch_all(products_csv: Path, output_json: Path, usda_key: str,
              name_search: bool = False) -> list[dict]:
    products = []
    with open(products_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append({
                "barcode": row.get("barcode", "").strip(),
                "product_name": row.get("product_name", "").strip(),
            })

    if usda_key == "DEMO_KEY":
        print("Note: USDA_KEY not set. USDA lookups skipped to avoid rate limits.")
        print("      Get a free key at https://fdc.nal.usda.gov/api-guide.html\n")

    print(f"Fetching {len(products)} products...\n")
    results = []
    for i, p in enumerate(products, 1):
        print(f"[{i}/{len(products)}] {p['product_name']}")
        data = fetch_product(p["barcode"], p["product_name"], usda_key, name_search)
        results.append(data)
        time.sleep(0.5)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    off_found  = sum(1 for r in results if r["off_data"])
    by_name = [r["product_name"] for r in results
               if r["off_data"] and r["off_data"].get("matched_by") == "name_search"]
    usda_found = sum(1 for r in results if r["usda_data"])
    print(f"\n✓ Raw data saved to {output_json}")
    print(f"  OFF data:  {off_found}/{len(results)}")
    if by_name:
        print(f"  Matched by name search, not barcode (check them): {', '.join(by_name)}")
    print(f"  USDA data: {usda_found}/{len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="FoodTree Stage 1: Fetch product data")
    parser.add_argument("--input", required=True,
                        help="products.csv (columns: barcode, product_name)")
    parser.add_argument("--output", required=True,
                        help="Output path for raw_products.json")
    parser.add_argument("--usda-key", default=USDA_API_KEY,
                        help="USDA FoodData Central API key")
    parser.add_argument("--name-search", action="store_true",
                        help="Search Open Food Facts by name when a barcode is missing or not found")
    args = parser.parse_args()

    fetch_all(Path(args.input), Path(args.output), args.usda_key, args.name_search)


if __name__ == "__main__":
    main()
