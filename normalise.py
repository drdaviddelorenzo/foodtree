"""Stage 2: turn raw product data into foods.csv (FoodItem, Ingredient, Percentage, Category, Source).

    python normalise.py --input raw_products.json --output foods.csv

Ingredient names are harmonised across languages and missing percentages are estimated
from nutrition data. No LLM is used.
"""

import csv
import json
import re
import sys
import warnings
import argparse
import unicodedata
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


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
    "tomato": "tomato",
    "tomate": "tomato",
    "corn": "corn",
    "maize": "corn",
    "maiz": "corn",
    "potatoes": "potato",
    "select_potatoes": "potato",
    "dehydrated_potato": "potato",
    "dehydrated_potatoes": "potato",
    "dried_potato": "potato",
    "cornmeal": "corn",
    "corn_meal": "corn",
    "maize_meal": "corn",
    "maize_flour": "corn_flour",
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
    slug = tag.replace("-", "_").lower().strip("_")
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


def extract_percentages_from_off(ingredients: list[dict]) -> dict[str, float | None]:
    """Extract percent_estimate values from OFF ingredient list."""
    result = {}
    for ing in ingredients:
        name = canonical_name(ing.get("text", ""))
        pct = ing.get("percent_estimate")
        if pct is not None:
            try:
                result[name] = float(pct)
            except (ValueError, TypeError):
                result[name] = None
        else:
            result[name] = None
    return result


def estimate_percentages_from_rank(
    names: list[str],
    nutrients: dict[str, float | None],
    decay: float = 0.65,
) -> dict[str, float]:
    """Estimate ingredient percentages using geometric rank decay, then calibrate"""
    n = len(names)
    if n == 0:
        return {}

    weights = np.array([decay ** i for i in range(n)])
    weights = weights / weights.sum() * 100.0

    MACRO_PROFILES: dict[str, dict[str, float]] = {
        "sugar":          {"carb": 1.00, "fat": 0.00, "protein": 0.00},
        "brown_sugar":    {"carb": 0.97, "fat": 0.00, "protein": 0.00},
        "glucose_syrup":  {"carb": 0.80, "fat": 0.00, "protein": 0.00},
        "cocoa_mass":     {"carb": 0.25, "fat": 0.50, "protein": 0.12},
        "cocoa_butter":   {"carb": 0.00, "fat": 1.00, "protein": 0.00},
        "cocoa_powder":   {"carb": 0.35, "fat": 0.12, "protein": 0.20},
        "whole_milk":     {"carb": 0.05, "fat": 0.04, "protein": 0.03},
        "milk_powder":    {"carb": 0.38, "fat": 0.27, "protein": 0.26},
        "skimmed_milk_powder": {"carb": 0.52, "fat": 0.01, "protein": 0.36},
        "whey":           {"carb": 0.73, "fat": 0.01, "protein": 0.13},
        "vegetable_oil":  {"carb": 0.00, "fat": 1.00, "protein": 0.00},
        "sunflower_oil":  {"carb": 0.00, "fat": 1.00, "protein": 0.00},
        "palm_oil":       {"carb": 0.00, "fat": 1.00, "protein": 0.00},
        "butter":         {"carb": 0.01, "fat": 0.81, "protein": 0.01},
        "wheat_flour":    {"carb": 0.72, "fat": 0.01, "protein": 0.10},
        "corn_flour":     {"carb": 0.73, "fat": 0.04, "protein": 0.07},
        "potato":         {"carb": 0.17, "fat": 0.00, "protein": 0.02},
        "water":          {"carb": 0.00, "fat": 0.00, "protein": 0.00},
        "salt":           {"carb": 0.00, "fat": 0.00, "protein": 0.00},
        "caffeine":       {"carb": 0.00, "fat": 0.00, "protein": 0.00},
        "taurine":        {"carb": 0.00, "fat": 0.00, "protein": 0.00},
    }

    fat_target = nutrients.get("fat_g") or nutrients.get("fat")
    prot_target = nutrients.get("protein_g") or nutrients.get("protein")
    carb_target = nutrients.get("carbohydrates_g") or nutrients.get("carbohydrates")

    known_macros = {
        k: v for k, v in {"fat": fat_target, "protein": prot_target, "carb": carb_target}.items()
        if v is not None
    }

    if not known_macros:
        return dict(zip(names, np.round(weights, 2)))

    macro_keys = list(known_macros.keys())
    A = np.zeros((len(macro_keys), n))
    for j, name in enumerate(names):
        profile = MACRO_PROFILES.get(name, {})
        for m_idx, macro in enumerate(macro_keys):
            A[m_idx, j] = profile.get(macro, 0.0)

    targets = np.array([known_macros[m] for m in macro_keys])

    def objective(x):
        """Minimise deviation from macro targets + deviation from geometric prior."""
        macro_error = np.sum((A @ x - targets) ** 2)
        prior_error = np.sum((x - weights) ** 2) * 0.1
        return macro_error + prior_error

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 100}]
    bounds = [(0.01, 99) for _ in range(n)]

    try:
        result = minimize(
            objective,
            x0=weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 500},
        )
        if result.success:
            return dict(zip(names, np.round(result.x, 2)))
    except Exception:
        pass

    return dict(zip(names, np.round(weights, 2)))


LOW_FAT_WORDS = ("potato", "corn", "maize", "wheat", "rice", "flour", "starch", "meal",
                 "semolina", "cereal", "maltodextrin", "tapioca")
MINOR_WORDS = ("flavour", "flavor", "aroma", "season", "condiment", "prepar", "whey", "cheese", "milk",
               "cream", "buttermilk", "tomato", "onion", "garlic", "paprika", "pepper", "yeast", "sugar",
               "dextrose", "acid", "extract", "spice", "lactose", "vinegar", "sweetener", "colour",
               "color", "protein", "enhancer", "antioxidant", "emulsifier")
SALT_NAMES = {"salt", "sea_salt"}


def _is_additive(name: str) -> bool:
    """E-numbers, flavourings and seasoning ingredients: present in small amounts."""
    return bool(re.match(r"^e\d{3}[a-z]*(_|$)", name)) or any(w in name for w in MINOR_WORDS)


def estimate_fried_snack(names: list[str], declared: dict[str, float],
                         nutrients: dict[str, float | None], decay: float = 0.65):
    """Crisps and similar snacks, on a finished-product basis.

    Declared percentages are kept. Oils get the fat of the nutrition table, salt gets the salt
    of the table, additives get a trace, and the rest is shared by rank. Returns None when the
    product is not of this kind (the general estimator is used instead).
    """
    fat, salt = nutrients.get("fat_g"), nutrients.get("salt_g")
    fats = [n for n in names if n.endswith(("_oil", "_fat"))]
    salts = [n for n in names if n in SALT_NAMES]
    additives = [n for n in names if _is_additive(n)]
    base = [n for n in names if n not in fats and n not in salts and n not in additives]
    if fat is None or not fats or not base or not all(any(w in n for w in LOW_FAT_WORDS) for n in base):
        return None
    pct = {n: declared[n] for n in names if n in declared}
    left = lambda: max(100.0 - sum(pct.values()), 0.0)
    for n in salts:
        if n not in pct and salt is not None:
            pct[n] = min(salt, left())
    free_fats = [n for n in fats if n not in pct]
    if free_fats:
        total = max(fat - sum(pct[n] for n in fats if n in pct), 0.0)
        for n in free_fats:
            pct[n] = min(total / len(free_fats), left())
    for n in additives:
        if n not in pct:
            pct[n] = min(0.1, left())
    free_base = [n for n in base if n not in pct]
    if free_base:
        w = np.array([decay ** i for i in range(len(free_base))])
        for n, share in zip(free_base, w / w.sum() * left()):
            pct[n] = float(share)
    return {n: round(pct.get(n, 0.0), 3) for n in names}


def normalise_product(raw: dict) -> list[dict] | None:
    """Normalise a single product. Returns list of row dicts or None if no data."""
    product_name = raw["product_name"]
    off = raw.get("off_data") or {}
    usda = raw.get("usda_data") or {}


    raw_ingredients = off.get("ingredients", [])
    names: list[str] = []
    pct_map: dict[str, float | None] = {}

    if raw_ingredients:
        seen: set[str] = set()
        for ing in raw_ingredients:
            raw_id = ing.get("id", "")
            raw_text = ing.get("text", "")
            name = parse_off_tag(raw_id) if raw_id else canonical_name(raw_text)
            if not name or name in seen:
                continue
            if _is_label_text(name):
                continue
            seen.add(name)
            names.append(name)
            pct = ing.get("percent_estimate")
            pct_map[name] = float(pct) if pct is not None else None

    else:
        off_text = off.get("ingredients_raw_text", "")
        usda_text = ""
        if usda:
            usda_desc = (usda.get("usda_description") or "").lower()
            stopwords = {"a", "an", "the", "of", "with", "and", "or", "for",
                         "in", "on", "at", "to", "by", "fish", "food", "s"}
            query_words = {w.lower() for w in product_name.split()
                           if len(w) > 2 and w.lower() not in stopwords}
            if query_words and any(w in usda_desc for w in query_words):
                usda_text = usda.get("ingredients_raw_text", "")
            elif query_words:
                print(f"  [normalise] USDA match rejected (brand mismatch: '{usda_desc[:40]}')",
                      file=sys.stderr)

        raw_text = off_text or usda_text
        if not raw_text:
            print(f"  [normalise] No reliable ingredient data for '{product_name}' "
                  f"— add it manually to foods.csv", file=sys.stderr)
            return None
        seen = set()
        for part in re.split(r"[,;]", raw_text):
            name = canonical_name(part)
            if name and name not in seen and not _is_label_text(name):
                seen.add(name)
                names.append(name)
                pct_map[name] = None

    if not names:
        print(f"  [normalise] Could not extract ingredient names for '{product_name}'", file=sys.stderr)
        return None

    nutrients = {
        **(usda.get("nutrients_per_100g") or {}),
        **(off.get("nutrients_per_100g") or {}),
    }
    nutrients = {k: v for k, v in nutrients.items() if v is not None}
    declared = {}
    for ing in raw_ingredients:
        name = parse_off_tag(ing.get("id", "")) if ing.get("id") else canonical_name(ing.get("text", ""))
        if ing.get("percent") is not None and name in names:
            declared[name] = float(ing["percent"])
    snack = estimate_fried_snack(names, declared, nutrients)

    available = [v for v in pct_map.values() if v is not None]
    coverage = len(available) / len(names) if names else 0

    if snack is not None:
        pct_map = snack
        source_note = "label+nutrition"
    elif coverage < 0.5:
        nutrients = {
            **(usda.get("nutrients_per_100g") or {}),
            **(off.get("nutrients_per_100g") or {}),
        }
        nutrients = {k: v for k, v in nutrients.items() if v is not None}
        estimated = estimate_percentages_from_rank(names, nutrients)
        for name in names:
            if pct_map.get(name) is None:
                pct_map[name] = estimated.get(name, 0.0)
        source_note = "estimated"
    else:
        nutrients = {
            **(usda.get("nutrients_per_100g") or {}),
            **(off.get("nutrients_per_100g") or {}),
        }
        nutrients = {k: v for k, v in nutrients.items() if v is not None}
        estimated = estimate_percentages_from_rank(names, nutrients)
        for name in names:
            if pct_map.get(name) is None:
                pct_map[name] = estimated.get(name, 0.0)
        source_note = "off_estimate"

    source = off.get("source", "openfoodfacts")
    if usda.get("source"):
        source = f"{source}+usda"

    rows = []
    total = sum(pct_map.get(n, 0) or 0 for n in names)

    if total > 0 and abs(total - 100) > 5:
        scale = 100.0 / total
    else:
        scale = 1.0

    for name in names:
        pct = pct_map.get(name) or 0.0
        pct = round(pct * scale, 3)
        if pct <= 0:
            continue
        rows.append({
            "FoodItem": product_name,
            "Ingredient": name,
            "Percentage": pct,
            "Category": CATEGORIES.get(name, "other"),
            "Source": source_note,
        })

    return rows if rows else None


def normalise_all(input_json: Path, output_csv: Path) -> None:
    with open(input_json) as f:
        raw_products = json.load(f)

    all_rows = []
    failed = []

    print(f"\nNormalising {len(raw_products)} products...\n")

    for i, raw in enumerate(raw_products, 1):
        name = raw["product_name"]
        print(f"[{i}/{len(raw_products)}] {name}")
        rows = normalise_product(raw)
        if rows:
            all_rows.extend(rows)
            print(f"  ✓ {len(rows)} ingredients  (percentages: {rows[0]['Source']})")
        else:
            failed.append(name)
            print(f"  ✗ Failed")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["FoodItem", "Ingredient", "Percentage", "Category", "Source"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ Saved {len(all_rows)} rows to {output_csv}")
    print(f"  Products written: {len(raw_products) - len(failed)}/{len(raw_products)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="FoodTree Stage 2: Normalise ingredients")
    parser.add_argument("--input", required=True, help="raw_products.json from fetch.py")
    parser.add_argument("--output", required=True, help="Output foods.csv path")
    args = parser.parse_args()
    normalise_all(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
