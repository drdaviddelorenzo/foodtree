"""Search Open Food Facts and save a products.csv for the pipeline.

    python search.py "energy drinks"
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

OFF_SEARCH_URL = "https://search.openfoodfacts.org/search"
HEADERS = {"User-Agent": "FoodTree/2.0 (foodtree-project; open-source)"}

COUNTRY_ALIASES = {
    "uk": "united-kingdom",
    "gb": "united-kingdom",
    "us": "united-states",
    "usa": "united-states",
    "es": "spain",
    "fr": "france",
    "de": "germany",
    "it": "italy",
    "pt": "portugal",
    "nl": "netherlands",
    "be": "belgium",
    "ch": "switzerland",
}


def search_off(query: str | None, category: str | None,
               country: str | None, limit: int) -> list[dict]:
    """Search Open Food Facts and return a list of product dicts."""
    fetch_size = min(limit * 3 if country else limit, 200)

    params = {
        "fields": "code,product_name,brands,categories_tags,countries_tags,quantity",
        "page_size": fetch_size,
        "page": 1,
    }

    if query:
        params["q"] = query
    if category:
        params["categories_tags"] = "en:" + category.lower().replace(" ", "-")

    country_tag = None
    if country:
        c = country.lower().strip()
        c = COUNTRY_ALIASES.get(c, c)
        country_tag = f"en:{c}"

    try:
        resp = requests.get(OFF_SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"Error contacting OpenFoodFacts: {e}", file=sys.stderr)
        sys.exit(1)

    products = data.get("hits", [])
    if not products:
        print("No products found. Try a different search term or remove the country filter.")
        sys.exit(0)

    results = []
    for p in products:
        barcode = (p.get("code") or "").strip()
        name = (p.get("product_name") or "").strip()
        brands_raw = p.get("brands") or []
        if isinstance(brands_raw, list):
            brand = ", ".join(brands_raw)
        else:
            brand = str(brands_raw)
        brand = brand.strip()
        qty = (p.get("quantity") or "").strip()

        if not name and not brand:
            continue

        if country_tag:
            tags = p.get("countries_tags") or []
            if country_tag not in tags:
                continue

        display = name or brand
        if brand and brand.lower() not in display.lower():
            display = f"{brand} {display}"
        if qty:
            display = f"{display} ({qty})"
        display = display.strip()

        results.append({
            "barcode": barcode,
            "product_name": display,
            "raw_name": name,
            "brand": brand,
        })

        if len(results) >= limit:
            break

    if not results:
        print("No products found. Try a different search term or remove the country filter.")
        sys.exit(0)

    return results


def print_results(results: list[dict]) -> None:
    print(f"\nFound {len(results)} products:\n")
    print(f"  {'#':>3}  {'Barcode':<16}  Product")
    print(f"  {'─'*3}  {'─'*16}  {'─'*50}")
    for i, p in enumerate(results, 1):
        barcode = p["barcode"] or "(no barcode)"
        name = p["product_name"][:60]
        print(f"  {i:>3}  {barcode:<16}  {name}")
    print()


def select_products(results: list[dict]) -> list[dict]:
    """Let the user pick which products to save."""
    print("Which products do you want to include in your analysis?")
    print("  Enter numbers separated by commas (e.g.  1,3,5,7)")
    print("  Or press Enter to save ALL of them")
    print("  Or type 'q' to quit without saving\n")

    raw = input("  Your selection: ").strip()

    if raw.lower() == "q":
        print("Cancelled.")
        sys.exit(0)

    if not raw:
        return results

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        selected = []
        for i in indices:
            if 0 <= i < len(results):
                selected.append(results[i])
            else:
                print(f"  Warning: {i+1} is out of range, skipping")
        return selected
    except ValueError:
        print("Couldn't parse your selection. Saving all products.")
        return results


def save_csv(products: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        ans = input(f"\n  {output_path} already exists. Overwrite? (y/n): ").strip().lower()
        if ans != "y":
            existing = []
            with open(output_path) as f:
                reader = csv.DictReader(f)
                existing = list(reader)
            products = existing + products
            print(f"  Appending to existing file ({len(existing)} + {len(products) - len(existing)} products)")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["barcode", "product_name"])
        writer.writeheader()
        for p in products:
            writer.writerow({
                "barcode": p["barcode"],
                "product_name": p["product_name"],
            })

    print(f"\n✓ Saved {len(products)} products to {output_path}")
    print(f"\nNext step:")
    print(f"  python pipeline.py run \\")
    print(f"      --input {output_path} \\")
    print(f"      --name \"Your Analysis Name\" \\")
    print(f"      --dir {output_path.parent}/")
    print(f"")
    print(f"  You will be asked for a min % threshold after ingredients are fetched.")
    print(f"  The default (2%) works for most foods. Use a lower value if your")
    print(f"  products contain small but important ingredients (e.g. 0.1% for")
    print(f"  energy drinks, to keep caffeine and taurine). To skip the prompt:")
    print(f"      --min-pct 2      (standard foods)")
    print(f"      --min-pct 0.1    (energy drinks, supplements, etc.)")


def main():
    parser = argparse.ArgumentParser(
        description="Search OpenFoodFacts and build a products.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?",
                        help="Keyword search term (e.g. 'energy drinks', 'dark chocolate')")
    parser.add_argument("--category", "-c",
                        help="OpenFoodFacts category tag (e.g. 'energy-drinks', 'potato-chips')")
    parser.add_argument("--country",
                        help="Filter by country (e.g. 'uk', 'spain', 'france')")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: <query>/products.csv)")
    parser.add_argument("--limit", "-n", type=int, default=40,
                        help="Max number of results to fetch (default: 40)")
    args = parser.parse_args()

    if not args.query and not args.category:
        parser.print_help()
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        folder_name = (args.query or args.category).lower().replace(" ", "-")
        output_path = Path("analyses") / folder_name / "products.csv"

    term = args.query or args.category
    country_label = f" [{args.country}]" if args.country else ""
    print(f"\nSearching OpenFoodFacts for: \"{term}\"{country_label} ...")

    results = search_off(
        query=args.query,
        category=args.category,
        country=args.country,
        limit=args.limit,
    )

    print_results(results)
    selected = select_products(results)

    if not selected:
        print("Nothing selected.")
        sys.exit(0)

    save_csv(selected, output_path)


if __name__ == "__main__":
    main()
