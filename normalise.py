"""Stage 2: turn raw product data into foods.csv
(FoodItem, Ingredient, Percentage, PctLow, PctHigh, Category, Class, Source).

    python normalise.py --input raw_products.json --output foods.csv [--moisture 1-4]

Ingredient names are harmonised across languages. Percentages are estimated from the label
(declared percentages, ingredient order and nutrition table) as described in estimate.py.
"""

import csv
import json
import math
import os
import re
import sys
import argparse
import unicodedata
from pathlib import Path

import hashlib
import zlib

import numpy as np

import estimate


SYNONYMS: dict[str, str] = {
    "cocoa_mass": "cocoa_mass",
    "cacao_mass": "cocoa_mass",
    "cocoa_liquor": "cocoa_mass",
    "cocoa_solids": "cocoa_mass",
    "chocolate_liquor": "cocoa_mass",
    "masa_de_cacao": "cocoa_mass",
    "cacao_paste": "cocoa_mass",
    "cocoa_paste": "cocoa_mass",
    "cocoa_butter": "cocoa_butter",
    "cacao_butter": "cocoa_butter",
    "manteca_de_cacao": "cocoa_butter",
    "beurre_de_cacao": "cocoa_butter",
    "cocoa_powder": "cocoa_powder",
    "cacao_powder": "cocoa_powder",
    "cacao_en_polvo": "cocoa_powder",
    "defatted_cocoa_powder": "cocoa_powder",
    "fat_reduced_cocoa_powder": "cocoa_powder",
    "sugar": "sugar",
    "sucrose": "sugar",
    "cane_sugar": "sugar",
    "beet_sugar": "sugar",
    "azucar": "sugar",
    "sucre": "sugar",
    "zucker": "sugar",
    "brown_sugar": "brown_sugar",
    "raw_cane_sugar": "brown_sugar",
    "glucose_syrup": "glucose_syrup",
    "corn_syrup": "glucose_syrup",
    "high_fructose_corn_syrup": "glucose_syrup",
    "fructose_syrup": "glucose_syrup",
    "glucose-fructose_syrup": "glucose_syrup",
    "glucose": "glucose",
    "fructose": "fructose",
    "honey": "honey",
    "miel": "honey",
    "lactose": "lactose",
    "maltose": "maltose",
    "maltodextrin": "maltodextrin",
    "vegetable_oil": "vegetable_oil",
    "aceite_vegetal": "vegetable_oil",
    "huile_végétale": "vegetable_oil",
    "sunflower_oil": "sunflower_oil",
    "aceite_de_girasol": "sunflower_oil",
    "palm_oil": "palm_oil",
    "aceite_de_palma": "palm_oil",
    "rapeseed_oil": "rapeseed_oil",
    "canola_oil": "rapeseed_oil",
    "olive_oil": "olive_oil",
    "aceite_de_oliva": "olive_oil",
    "coconut_oil": "coconut_oil",
    "whole_milk": "whole_milk",
    "milk": "whole_milk",
    "leche": "whole_milk",
    "lait": "whole_milk",
    "skimmed_milk": "skimmed_milk",
    "skim_milk": "skimmed_milk",
    "nonfat_milk": "skimmed_milk",
    "milk_powder": "milk_powder",
    "whole_milk_powder": "milk_powder",
    "skimmed_milk_powder": "skimmed_milk_powder",
    "nonfat_milk_powder": "skimmed_milk_powder",
    "cream": "cream",
    "butter": "butter",
    "mantequilla": "butter",
    "butterfat": "butter",
    "whey": "whey",
    "whey_powder": "whey",
    "suero_de_leche": "whey",
    "lactoserum": "whey",
    "wheat_flour": "wheat_flour",
    "harina_de_trigo": "wheat_flour",
    "farine_de_blé": "wheat_flour",
    "corn_flour": "corn_flour",
    "maize_flour": "corn_flour",
    "harina_de_maiz": "corn_flour",
    "corn_starch": "corn_starch",
    "maize_starch": "corn_starch",
    "almidon_de_maiz": "corn_starch",
    "potato_starch": "potato_starch",
    "rice_flour": "rice_flour",
    "oat_flour": "oat_flour",
    "wheat_starch": "wheat_starch",
    "soy_protein": "soy_protein",
    "soya_protein": "soy_protein",
    "pea_protein": "pea_protein",
    "whey_protein": "whey_protein",
    "egg_white": "egg_white",
    "albumin": "egg_white",
    "salt": "salt",
    "sal": "salt",
    "sodium_chloride": "salt",
    "sea_salt": "salt",
    "iodized_salt": "salt",
    "water": "water",
    "agua": "water",
    "eau": "water",
    "wasser": "water",
    "carbonated_water": "water",
    "sparkling_water": "water",
    "mineral_water": "water",
    "soy_lecithin": "soy_lecithin",
    "sunflower_lecithin": "sunflower_lecithin",
    "lecithin": "soy_lecithin",
    "lecitina_de_soja": "soy_lecithin",
    "emulsifier": "emulsifier",
    "vanilla": "vanilla",
    "vainilla": "vanilla",
    "vanillin": "vanillin",
    "natural_flavour": "natural_flavouring",
    "natural_flavor": "natural_flavouring",
    "natural_flavoring": "natural_flavouring",
    "flavouring": "natural_flavouring",
    "flavoring": "natural_flavouring",
    "aroma": "natural_flavouring",
    "caffeine": "caffeine",
    "cafeina": "caffeine",
    "taurine": "taurine",
    "taurina": "taurine",
    "b3": "niacin",
    "niacin": "niacin",
    "b6": "vitamin_b6",
    "b12": "vitamin_b12",
    "citric_acid": "citric_acid",
    "acido_citrico": "citric_acid",
    "carbon_dioxide": "carbon_dioxide",
    "co2": "carbon_dioxide",
    "potato": "potato",
    "patata": "potato",
    "dehydrated_potato": "dehydrated_potato",
    "tomato": "tomato",
    "tomate": "tomato",
    "corn": "corn",
    "maize": "corn",
    "maiz": "corn",
    "potatoes": "potato",
    "select_potatoes": "potato",
    "dehydrated_potatoes": "dehydrated_potato",
    "dried_potato": "dehydrated_potato",
    "potato_flakes": "dehydrated_potato",
    "cornmeal": "cornmeal",
    "corn_meal": "cornmeal",
    "maize_meal": "cornmeal",
}

CATEGORIES: dict[str, str] = {
    "cocoa_mass": "base",        "cocoa_butter": "fat",
    "cocoa_powder": "base",      "sugar": "sweetener",
    "brown_sugar": "sweetener",  "glucose_syrup": "sweetener",
    "glucose": "sweetener",      "fructose": "sweetener",
    "honey": "sweetener",        "lactose": "sweetener",
    "maltose": "sweetener",      "maltodextrin": "sweetener",
    "vegetable_oil": "fat",      "sunflower_oil": "fat",
    "palm_oil": "fat",           "rapeseed_oil": "fat",
    "olive_oil": "fat",          "coconut_oil": "fat",
    "butter": "fat",
    "whole_milk": "dairy",       "skimmed_milk": "dairy",
    "milk_powder": "dairy",      "skimmed_milk_powder": "dairy",
    "cream": "dairy",            "whey": "dairy",
    "wheat_flour": "base",       "corn_flour": "base",
    "corn_starch": "base",       "potato_starch": "base",
    "rice_flour": "base",        "oat_flour": "base",
    "wheat_starch": "base",      "potato": "base",
    "dehydrated_potato": "base", "potato_flour": "base",
    "cornmeal": "base",          "corn_oil": "fat",
    "residual_water": "water",
    "corn": "base",              "tomato": "base",
    "cod": "protein",            "alaska_pollock": "protein",
    "salmon": "protein",         "tuna": "protein",
    "haddock": "protein",        "tilapia": "protein",
    "fish": "protein",           "seafood": "protein",
    "chicken": "protein",        "chicken_breast": "protein",
    "beef": "protein",           "pork": "protein",
    "lamb": "protein",           "turkey": "protein",
    "egg": "protein",            "whole_egg": "protein",
    "tofu": "protein",           "soy_protein": "protein",
    "pea_protein": "protein",    "whey_protein": "protein",
    "egg_white": "protein",      "chickpea": "protein",
    "lentil": "protein",         "kidney_bean": "protein",
    "salt": "mineral",           "water": "water",
    "caffeine": "stimulant",     "taurine": "stimulant",
    "niacin": "vitamin",         "vitamin_b6": "vitamin",
    "vitamin_b12": "vitamin",    "citric_acid": "additive",
    "carbon_dioxide": "additive","soy_lecithin": "additive",
    "sunflower_lecithin": "additive", "emulsifier": "additive",
    "vanilla": "flavouring",     "vanillin": "flavouring",
    "natural_flavouring": "flavouring",
}


def slugify(text: str) -> str:
    """Convert raw ingredient text to a clean snake_case slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\d+[.,]?\d*\s*%", "", text)
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def canonical_name(raw: str) -> str:
    """Return the canonical name for an ingredient string."""
    slug = slugify(raw)
    if slug in SYNONYMS:
        return SYNONYMS[slug]
    for key, value in SYNONYMS.items():
        if slug.startswith(key) or key.startswith(slug[:6]) and len(slug) >= 6:
            return value
    return slug


def parse_off_tag(tag: str) -> str:
    """Convert an OFF ingredients_tag like 'en:cocoa-mass' to a canonical name."""
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    slug = tag.replace("-", "_").replace(" ", "_").lower().strip("_")
    return SYNONYMS.get(slug, slug)


_LABEL_TEXT_SIGNALS = {
    "caution", "storage", "keep_frozen", "keep_refrigerated", "although",
    "may_remain", "important", "if_food", "thawed", "cooler", "defrost",
    "allergen", "warning", "best_before", "use_by", "prepared", "packed",
    "manufactured", "contains_fish", "contains_milk", "contains_wheat",
    "including_cereals", "cereals_containing", "this_product_contains",
    "some_may", "every_care", "bones_removed", "remove_bones",
}


def _is_label_text(name: str) -> bool:
    """Return True if `name` looks like label text rather than an ingredient."""
    if name[0].isdigit():
        return True
    if name.count("_") > 6:
        return True
    parts = set(name.split("_"))
    return bool(parts & _LABEL_TEXT_SIGNALS)


class InconsistentLabel(Exception):
    """The label of a product cannot be satisfied by any composition."""

    def __init__(self, product: str, problems: list[str]):
        super().__init__(f"{product}: " + "; ".join(problems))
        self.product, self.problems = product, problems


class SamplingFailure(InconsistentLabel):
    """The outer bounds admit the label but no compatible composition was sampled."""


def _off_name(ing: dict) -> str:
    return parse_off_tag(ing.get("id", "")) if ing.get("id") else canonical_name(ing.get("text", ""))


def _items_from_text(text: str) -> list[estimate.Item]:
    """Top-level ingredients from a plain ingredient list (USDA, or OFF without structure)."""
    parts, _ = estimate.split_top_level(text)
    items = []
    for part in parts:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", part.split("(")[0] + " " +
                      (part.split("(")[1] if "(" in part else ""))
        name = canonical_name(part)
        items.append(estimate.Item(
            name=name,
            off_id="",
            cls=estimate.ingredient_class("en:" + name.replace("_", "-"), part),
            declared=float(m.group(1).replace(",", ".")) if m else None,
        ))
    return items


REPLICATES = 200


def samples_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}-samples.csv")


def samples_meta_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}-samples.json")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_together(pairs: list[tuple[Path, Path]]) -> None:
    """Move several temporary files into place as one unit: files being replaced are moved
    aside first and put back if any move fails.

    This covers errors raised during publication, not the process being killed half way. In
    that case the folder may hold files from two runs; the fingerprints in
    foods-samples.json then no longer match, render.py refuses those replicates and asks for
    normalise to be rerun, so no support is ever computed from mismatched files."""
    moved, placed = [], []
    try:
        for tmp, dest in pairs:
            if dest.exists():
                backup = dest.with_name(f".{dest.name}.previous")
                os.replace(dest, backup)
                moved.append((backup, dest))
            os.replace(tmp, dest)
            placed.append(dest)
    except BaseException:
        for dest in placed:
            dest.unlink(missing_ok=True)
        for backup, dest in moved:
            os.replace(backup, dest)
        raise
    for backup, _ in moved:
        backup.unlink(missing_ok=True)


def normalise_product(raw: dict, comp: dict, moisture: tuple[float, float] = (0.0, 100.0),
                      allow_inconsistent: bool = False) -> tuple[list[dict], list[dict]] | None:
    """Rows of foods.csv for one product and its replicate compositions, or None if it has no
    usable ingredient data.

    Replicates are REPLICATES compositions resampled from the estimator's weighted samples;
    render.py uses them to measure how well supported each branch of a tree is.

    Percentages are on a finished-product basis: the dry matter each ingredient leaves in
    100 g of product, plus residual water as its own row. Each is the weighted mean of the
    compositions compatible with the label (see estimate.py), with PctLow and PctHigh its
    bounds. Raises InconsistentLabel when no composition satisfies the label, unless
    allow_inconsistent is set (the least-violating composition is then used).
    """
    product_name = raw["product_name"]
    off = raw.get("off_data") or {}
    usda = raw.get("usda_data") or {}

    if off.get("ingredients"):
        items, notes = estimate.top_level_items(off, _off_name)
    else:
        text = off.get("ingredients_raw_text", "")
        if not text and usda:
            desc = (usda.get("usda_description") or "").lower()
            words = {w.lower() for w in product_name.split() if len(w) > 2}
            if words and any(w in desc for w in words):
                text = usda.get("ingredients_raw_text", "")
            else:
                print(f"  [normalise] USDA match rejected (brand mismatch: '{desc[:40]}')",
                      file=sys.stderr)
        if not text:
            print(f"  [normalise] No reliable ingredient data for '{product_name}' "
                  f"— add it with --manual", file=sys.stderr)
            return None
        items, notes = _items_from_text(text), []
    items = [it for it in items if it.name and not _is_label_text(it.name)]
    for it in items:
        if it.cls == "potato_dehydrated" and it.name == "potato":
            it.name = "dehydrated_potato"
    if not items:
        print(f"  [normalise] Could not extract ingredient names for '{product_name}'",
              file=sys.stderr)
        return None
    for note in notes:
        print(f"  [normalise] {product_name}: {note}", file=sys.stderr)

    nutrients = {**(usda.get("nutrients_per_100g") or {}), **(off.get("nutrients_per_100g") or {})}
    nutrients = {k: v for k, v in nutrients.items() if v is not None}
    est = estimate.estimate_product(items, nutrients, comp, moisture=moisture,
                                    allow_inconsistent=allow_inconsistent)
    for warning in est.warnings:
        print(f"  [normalise] {product_name}: {warning}", file=sys.stderr)
    if est.sampling_failed:
        raise SamplingFailure(product_name, est.problems)
    if not est.feasible and not allow_inconsistent:
        raise InconsistentLabel(product_name, est.problems)
    if not est.feasible:
        print(f"  [normalise] {product_name}: inconsistent label, least-violating composition "
              f"used ({'; '.join(est.problems)})", file=sys.stderr)
    if len(est.samples) == 0:
        raise InconsistentLabel(product_name, est.problems or ["no compatible composition found"])

    centre = est.weights @ est.samples
    rows = []
    for i, it in enumerate(items):
        source = ("inconsistent-label" if not est.feasible
                  else "declared" if it.declared is not None else "label-constrained")
        rows.append({
            "FoodItem": product_name,
            "Ingredient": it.name,
            "Percentage": round(float(centre[i]), 3),
            "PctLow": round(float(est.dry_low[i]), 3) if est.feasible else "",
            "PctHigh": round(float(est.dry_high[i]), 3) if est.feasible else "",
            "Category": CATEGORIES.get(it.name, "other"),
            "Class": it.cls,
            "Source": source,
        })
    rows.append({
        "FoodItem": product_name,
        "Ingredient": "residual_water",
        "Percentage": round(float(centre[-1]), 3),
        "PctLow": round(float(est.water_low), 3) if est.feasible else "",
        "PctHigh": round(float(est.water_high), 3) if est.feasible else "",
        "Category": "water",
        "Class": "",
        "Source": "inconsistent-label" if not est.feasible else "label-constrained",
    })
    rng = np.random.default_rng(zlib.crc32(product_name.encode("utf-8")))
    picks = rng.choice(len(est.samples), size=REPLICATES, p=est.weights)
    names = [it.name for it in items] + ["residual_water"]
    replicates = [{"FoodItem": product_name, "Replicate": r, "Ingredient": name,
                   "Percentage": round(float(v), 3)}
                  for r, k in enumerate(picks) for name, v in zip(names, est.samples[k])]
    return rows, replicates


def load_manual(manual_csv: Path) -> dict[str, list[dict]]:
    """Read products entered by hand: FoodItem, Ingredient, Percentage[, Category, Source].

    Percentages of each product must add up to 100 (within 1).
    """
    products: dict[str, list[dict]] = {}
    with open(manual_csv, newline="", encoding="utf-8") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            item = (row.get("FoodItem") or "").strip()
            ing = (row.get("Ingredient") or "").strip()
            try:
                pct = float(row.get("Percentage") or "")
            except ValueError:
                pct = -1.0
            if not item or not ing or not math.isfinite(pct) or pct < 0:
                sys.exit(f"Error: {manual_csv} line {line}: FoodItem, Ingredient and a "
                         "finite, non-negative Percentage are required.")
            name = canonical_name(ing)
            products.setdefault(item, []).append({
                "FoodItem": item,
                "Ingredient": name,
                "Percentage": pct,
                "Category": (row.get("Category") or "").strip() or CATEGORIES.get(name, "other"),
                "Source": (row.get("Source") or "").strip() or "manual",
            })
    for item, rows in products.items():
        total = sum(r["Percentage"] for r in rows)
        if not math.isfinite(total) or abs(total - 100) > 1:
            sys.exit(f"Error: {manual_csv}: percentages of '{item}' add up to {total:.2f}, not 100.")
    return products


def omitted_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}-omitted.txt")


def parse_moisture(text: str | None) -> tuple[float, float]:
    """'1-4' -> (1.0, 4.0): water of the finished product, g per 100 g."""
    if not text:
        return (0.0, 100.0)
    try:
        lo, hi = (float(x) for x in text.split("-"))
    except ValueError:
        sys.exit(f"Error: --moisture must look like 1-4 (g water per 100 g), not '{text}'.")
    if not 0 <= lo <= hi <= 100:
        sys.exit(f"Error: --moisture {text} is not a range within 0-100.")
    return (lo, hi)


def normalise_all(input_json: Path, output_csv: Path, manual_csv: Path | None = None,
                  allow_missing: bool = False, moisture: tuple[float, float] = (0.0, 100.0),
                  allow_inconsistent: bool = False) -> list[str]:
    """Write foods.csv and foods-samples.csv (replicate compositions) and return the
    products left out.

    A product without usable data stops the run unless allow_missing is set; the products
    left out are then listed in <output>-omitted.txt. A product whose label no composition
    can satisfy stops the run unless allow_inconsistent is set.
    """
    with open(input_json) as f:
        raw_products = json.load(f)
    manual = load_manual(manual_csv) if manual_csv else {}
    comp = estimate.load_composition()

    all_rows = []
    failed = []
    all_replicates: list[dict] = []
    inconsistent: list[InconsistentLabel] = []
    unsampled: list[SamplingFailure] = []

    print(f"\nNormalising {len(raw_products)} products...\n")

    for i, raw in enumerate(raw_products, 1):
        name = raw["product_name"]
        print(f"[{i}/{len(raw_products)}] {name}")
        if name in manual:
            if raw.get("off_data") or raw.get("usda_data"):
                sys.exit(f"Error: '{name}' was fetched and is also in {manual_csv}; "
                         "rename one of them.")
            print(f"  → taken from {manual_csv.name}")
            continue
        try:
            result = normalise_product(raw, comp, moisture, allow_inconsistent)
        except SamplingFailure as exc:
            unsampled.append(exc)
            print("  ✗ No composition sampled")
            continue
        except InconsistentLabel as exc:
            inconsistent.append(exc)
            print("  ✗ Label inconsistent")
            continue
        if result:
            rows, replicates = result
            all_rows.extend(rows)
            all_replicates.extend(replicates)
            print(f"  ✓ {len(rows) - 1} ingredients + residual water")
        else:
            failed.append(name)
            print(f"  ✗ Failed")

    kept = (f"\n{output_csv.name} was left unchanged and belongs to an earlier run."
            if output_csv.exists() else "")
    messages = []
    if inconsistent:
        details = "\n".join(f"  {e.product}: {'; '.join(e.problems)}" for e in inconsistent)
        messages.append("No composition satisfies the label of these products (constraints "
                        f"that would have to be relaxed):\n{details}\nCheck their data, or "
                        "rerun with --allow-inconsistent to use the least-violating composition.")
    if unsampled:
        details = "\n".join(f"  {e.product}" for e in unsampled)
        messages.append("For these products the label is possible within the outer bounds, but "
                        f"no compatible composition was sampled:\n{details}\nThe label data may "
                        "be fine: check the composition classes of their ingredients "
                        "(data/ingredient_composition.csv and ingredient_class in estimate.py).")
    if messages:
        sys.exit("Error: " + "\n\n".join(messages) + kept)

    for name, rows in manual.items():
        all_rows.extend(rows)
        print(f"[manual] {name}\n  ✓ {len(rows)} ingredients  (from {manual_csv.name})")

    omitted_file = omitted_path(output_csv)
    if failed and not allow_missing:
        kept = (f" {output_csv.name} was left unchanged and belongs to an earlier run."
                if output_csv.exists() else "")
        sys.exit(f"Error: no usable data for {len(failed)} product(s): {', '.join(failed)}. "
                 "Fix their barcodes, add them with --manual, or rerun with --allow-missing "
                 f"to build the tree without them.{kept}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["FoodItem", "Ingredient", "Percentage", "PctLow", "PctHigh", "Category", "Class",
                  "Source"]
    tmp_csv = output_csv.with_name(f".{output_csv.name}.tmp")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    tmp_samples = output_csv.with_name(f".{samples_path(output_csv).name}.tmp")
    with open(tmp_samples, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["FoodItem", "Replicate", "Ingredient", "Percentage"])
        writer.writeheader()
        writer.writerows(all_replicates)
    tmp_meta = output_csv.with_name(f".{samples_meta_path(output_csv).name}.tmp")
    tmp_meta.write_text(json.dumps({
        "foods_sha256": file_sha256(tmp_csv),
        "samples_sha256": file_sha256(tmp_samples),
        "replicates": REPLICATES,
        "products_with_replicates": sorted({r["FoodItem"] for r in all_replicates}),
    }, indent=2), encoding="utf-8")
    try:
        publish_together([(tmp_csv, output_csv), (tmp_samples, samples_path(output_csv)),
                          (tmp_meta, samples_meta_path(output_csv))])
    finally:
        for tmp in (tmp_csv, tmp_samples, tmp_meta):
            tmp.unlink(missing_ok=True)
    if failed:
        omitted_file.write_text("\n".join(failed) + "\n", encoding="utf-8")
    else:
        omitted_file.unlink(missing_ok=True)

    written = len({r["FoodItem"] for r in all_rows})
    print(f"\n✓ Saved {len(all_rows)} rows to {output_csv}")
    print(f"  Products written: {written} ({len(manual)} manual), omitted: {len(failed)}")
    if failed:
        print(f"  Omitted (--allow-missing): {', '.join(failed)} → {omitted_file.name}",
              file=sys.stderr)
    return failed


def main():
    parser = argparse.ArgumentParser(description="FoodTree Stage 2: Normalise ingredients")
    parser.add_argument("--input", required=True, help="raw_products.json from fetch.py")
    parser.add_argument("--output", required=True, help="Output foods.csv path")
    parser.add_argument("--manual", default=None,
                        help="CSV of products entered by hand (FoodItem, Ingredient, Percentage"
                             "[, Category, Source])")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Leave out products without usable data instead of stopping")
    parser.add_argument("--moisture", default=None,
                        help="Water of the finished products, g per 100 g, as LO-HI (e.g. 1-4 "
                             "for crisps); by default it is left to the nutrition table")
    parser.add_argument("--allow-inconsistent", action="store_true",
                        help="Keep products whose label no composition satisfies, using the "
                             "least-violating composition")
    args = parser.parse_args()
    normalise_all(Path(args.input), Path(args.output),
                  Path(args.manual) if args.manual else None, args.allow_missing,
                  parse_moisture(args.moisture), args.allow_inconsistent)


if __name__ == "__main__":
    main()
