"""Regression on the crisps example: the snapshot must keep producing the published families.

Slow (about a minute): it normalises the 11 products and renders a tree with branch support.
"""

import csv
import shutil
from pathlib import Path

import pytest

import normalise
import render

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "crisps" / "raw_products.json"


@pytest.fixture(scope="module")
def crisps(tmp_path_factory):
    d = tmp_path_factory.mktemp("crisps")
    shutil.copy(EXAMPLE, d / "raw_products.json")
    normalise.normalise_all(d / "raw_products.json", d / "foods.csv", moisture=(1.0, 4.0))
    return d


def rows(d):
    return {(r["FoodItem"], r["Ingredient"]): r for r in csv.DictReader(open(d / "foods.csv"))}


def test_every_product_sums_to_100(crisps):
    totals = {}
    for (item, _), r in rows(crisps).items():
        totals[item] = totals.get(item, 0) + float(r["Percentage"])
    assert len(totals) == 11
    assert all(abs(t - 100) < 0.05 for t in totals.values())


@pytest.mark.parametrize("item,ingredient,low,high", [
    ("Marca blanca", "sunflower_oil", 34.4, 35.5),
    ("Doritos", "corn", 69.9, 73.2),
    ("Cheetos Gusanitos", "cornmeal", 75.1, 78.8),
    ("Pringles Original", "dehydrated_potato", 12.7, 74.9),
])
def test_bounds(crisps, item, ingredient, low, high):
    r = rows(crisps)[(item, ingredient)]
    assert float(r["PctLow"]) == pytest.approx(low, abs=0.2)
    assert float(r["PctHigh"]) == pytest.approx(high, abs=0.2)
    assert low - 0.2 <= float(r["Percentage"]) <= high + 0.2


def test_unifrac_tree_recovers_the_families_with_support(crisps):
    render.render_all(crisps / "foods.csv", crisps, "Crisps", metric="unifrac", min_pct=0,
                      support=50)
    text = (crisps / "crisps-diagnostics-unifrac.txt").read_text()
    maize = "Cheetos Gusanitos, Doritos, Doritos Tex Mex"
    crisps_family = "Kettle, Lay's Gourmet, Marca blanca, Tyrrell's, Walkers"
    for family in (maize, crisps_family):
        line = next(ln for ln in text.splitlines() if ln.strip().endswith(family))
        assert float(line.split()[0]) >= 90


def test_replicates_vary_where_the_label_is_uninformative(crisps):
    import pandas as pd
    reps = pd.read_csv(crisps / "foods-samples.csv")
    pringles = reps[(reps["FoodItem"] == "Pringles Original")
                    & (reps["Ingredient"] == "dehydrated_potato")]["Percentage"]
    oil = reps[(reps["FoodItem"] == "Marca blanca")
               & (reps["Ingredient"] == "sunflower_oil")]["Percentage"]
    assert reps["Replicate"].nunique() == normalise.REPLICATES
    assert pringles.std() > 5
    assert oil.max() - oil.min() < 1.2


def test_support_is_refused_with_an_inconsistent_label(tmp_path):
    import copy
    import json
    data = json.loads(EXAMPLE.read_text())
    bad = copy.deepcopy(data[0])
    bad["product_name"] = "Bad label"
    bad["off_data"]["nutrients_per_100g"]["fat_g"] = 10
    (tmp_path / "raw_products.json").write_text(json.dumps(data[:3] + [bad]))
    normalise.normalise_all(tmp_path / "raw_products.json", tmp_path / "foods.csv",
                            moisture=(1.0, 4.0), allow_inconsistent=True)
    render.render_all(tmp_path / "foods.csv", tmp_path, "T", metric="unifrac", min_pct=0,
                      support=20)
    text = (tmp_path / "t-diagnostics-unifrac.txt").read_text()
    assert "not computed" in text and "Bad label" in text


def test_snapshot_is_checked_before_replacing(tmp_path):
    import subprocess
    import sys
    repo = EXAMPLE.parent.parent.parent
    (tmp_path / "raw_products.json").write_text("[]")
    for k, content in enumerate(["{not json", '[{"product_name": "X", "off_data": "bad"}]',
                                 '[{"product_name": "X", "off_data": {"ingredients": "bad"}}]',
                                 '[{"product_name": "X", "off_data": {"ingredients": '
                                 '[{"id": "en:salt", "ingredients": ["bad"]}]}}]',
                                 '[{"product_name": "X", "usda_data": {"usda_description": 3}}]',
                                 '[{"product_name": "X", "off_data": {"ingredients": '
                                 '[{"id": "en:salt", "percent": "NaN"}]}}]',
                                 '[{"product_name": "X", "off_data": {"nutrients_per_100g": '
                                 '{"fat_g": "Infinity"}}}]']):
        broken = tmp_path / f"broken{k}.json"
        broken.write_text(content)
        out = subprocess.run([sys.executable, "pipeline.py", "run", "--snapshot", str(broken),
                              "--name", "X", "--dir", str(tmp_path), "--min-pct", "0",
                              "--distance", "unifrac"], cwd=repo, capture_output=True, text=True)
        assert out.returncode == 1
        assert (tmp_path / "raw_products.json").read_text() == "[]"
        assert not list(tmp_path.glob(".raw_products.json.*.tmp"))


def test_same_path_snapshot_is_still_checked(tmp_path):
    import subprocess
    import sys
    repo = EXAMPLE.parent.parent.parent
    target = tmp_path / "raw_products.json"
    target.write_text('[{"product_name": "X", "off_data": "bad"}]')
    out = subprocess.run([sys.executable, "pipeline.py", "run", "--snapshot", str(target),
                          "--name", "X", "--dir", str(tmp_path)], cwd=repo,
                         capture_output=True, text=True)
    assert out.returncode == 1 and "not a usable" in out.stderr
