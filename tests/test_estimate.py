import numpy as np
import pytest

import estimate

COMP = estimate.load_composition()


def items(*specs):
    return [estimate.Item(name, f"en:{name.replace('_', '-')}", cls, declared)
            for name, cls, declared in specs]


CRISPS = items(("potato", "potato_raw", None), ("sunflower_oil", "oil", 35.0), ("salt", "salt", None))
CRISPS_LABEL = {"fat_g": 35, "carbohydrates_g": 50, "protein_g": 6, "salt_g": 1.5}


@pytest.mark.parametrize("nutrient,value,expected", [
    ("fat", 5, (3.5, 6.5)), ("fat", 20, (16.0, 24.0)), ("fat", 50, (42.0, 58.0)),
    ("carb", 56, (48.0, 64.0)), ("protein", 6, (4.0, 8.0)),
    ("salt", 1.0, (0.625, 1.375)), ("salt", 2.0, (1.6, 2.4)),
])
def test_tolerance_bands_follow_eu_guidance(nutrient, value, expected):
    assert estimate.tolerance_band(nutrient, value) == pytest.approx(expected)


def test_unclosed_parenthesis_is_reported():
    parts, open_at = estimate.split_top_level("maize (80%), oils (maize), seasoning (whey, salt")
    assert parts[:2] == ["maize (80%)", "oils (maize)"]
    assert open_at == 2


def test_bounds_contain_declared_oil_and_sum_to_100():
    est = estimate.estimate_product(CRISPS, CRISPS_LABEL, COMP, moisture=(1, 4))
    assert est.feasible
    assert 34.4 <= est.dry_low[1] <= est.dry_high[1] <= 35.5
    assert np.allclose(est.samples.sum(axis=1), 100, atol=1e-6)
    centre = est.weights @ est.samples
    assert np.all(centre >= np.append(est.dry_low, est.water_low) - 1e-6)
    assert np.all(centre <= np.append(est.dry_high, est.water_high) + 1e-6)


def test_results_are_deterministic():
    a = estimate.estimate_product(CRISPS, CRISPS_LABEL, COMP, moisture=(1, 4))
    b = estimate.estimate_product(CRISPS, CRISPS_LABEL, COMP, moisture=(1, 4))
    assert np.array_equal(a.samples, b.samples) and np.array_equal(a.weights, b.weights)


def test_inconsistent_label_names_the_label_constraint():
    est = estimate.estimate_product(CRISPS, dict(CRISPS_LABEL, fat_g=10), COMP, moisture=(1, 4))
    assert not est.feasible
    assert any(p.startswith("fat (label") for p in est.problems)


def test_conflicting_declared_percentages_can_be_relaxed():
    oils = items(("sunflower_oil", "oil", 60.0), ("rapeseed_oil", "oil", 60.0))
    est = estimate.estimate_product(oils, {"fat_g": 99}, COMP)
    assert not est.feasible and any(p.startswith("declared") for p in est.problems)
    est = estimate.estimate_product(oils, {"fat_g": 99}, COMP, allow_inconsistent=True)
    assert len(est.samples) == 1


def test_impossible_nutrient_totals_are_infeasible():
    seasoning = items(("seasoning", "cheese_seasoning", None))
    est = estimate.estimate_product(seasoning, {"fat_g": 25, "carbohydrates_g": 75,
                                                "protein_g": 25, "salt_g": 20}, COMP)
    assert not est.feasible


def test_drawn_compositions_never_exceed_100_percent():
    rng = np.random.default_rng(0)
    for cp in COMP.values():
        for _ in range(500):
            d = estimate._draw_composition(cp, rng)
            assert d is None or sum(d.values()) <= 1 + 1e-12


def test_fixed_moisture_still_samples_distinct_recipes():
    est = estimate.estimate_product(CRISPS, CRISPS_LABEL, COMP, moisture=(2, 2))
    assert len(np.unique(est.samples.round(6), axis=0)) > 100


def test_order_rule_allows_small_ingredients_out_of_order():
    u, mass = np.array([60.0, 30.0, 1.0, 1.5]), np.array([60.0, 30.0, 1.0, 1.5])
    assert estimate._order_ok(u, mass)
    assert not estimate._order_ok(np.array([30.0, 60.0]), np.array([30.0, 60.0]))


def test_non_finite_values_count_as_not_declared():
    assert estimate.nutrient_value({"fat_g": float("nan")}, "fat") is None
    assert estimate.nutrient_value({"salt_g": None, "sodium_g": float("inf")}, "salt") is None
    assert estimate.nutrient_value({"salt_g": None, "sodium_g": 1e308}, "salt") is None
    assert estimate.nutrient_value({"sodium_g": 0.4}, "salt") == pytest.approx(1.0)
    off = {"ingredients": [{"id": "en:salt", "text": "salt", "percent": "NaN"}],
           "ingredients_raw_text": "salt"}
    items, _ = estimate.top_level_items(off, lambda ing: "salt")
    assert items[0].declared is None
