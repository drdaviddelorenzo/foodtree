import io

import numpy as np
import pandas as pd
import pytest
from Bio import Phylo

import normalise
import ontology
import render


def unifrac(rows, level=None, categories=None):
    tree = ontology.IngredientTree()
    table = pd.DataFrame(rows, index=[f"P{i}" for i in range(len(rows))]).fillna(0)
    if level:
        table = ontology.collapse(table, tree, ontology.LEVELS[level], {}, categories)
        categories = None
    return np.round(ontology.weighted_unifrac(table, tree, {}, categories), 4).tolist()


def test_unifrac_hand_computed_cases():
    assert unifrac([{"potato": 70, "sunflower_oil": 30}] * 2) == [0.0]
    assert unifrac([{"sunflower_oil": 100}, {"salt": 100}]) == [1.0]
    assert unifrac([{"sunflower_oil": 100}, {"vegetable_oil": 100}]) == [0.3333]
    assert unifrac([{"potato": 100}, {"corn": 100}]) == [0.6667]


def test_collapsed_unifrac_follows_the_tree():
    assert unifrac([{"potato": 100}, {"corn": 100}, {"sunflower_oil": 100}], "source") == [0.5, 1.0, 1.0]


def test_unknown_ingredients_attach_by_category():
    cats = {"sugar": "sweetener", "cod": "protein"}
    assert unifrac([{"sugar": 100}, {"cod": 100}], "group", cats) == [1.0]


def test_collapse_with_product_names_equal_to_ingredients():
    tree = ontology.IngredientTree()
    table = pd.DataFrame({"sugar": [100, 0], "honey": [0, 100]}, index=["sugar", "honey"])
    out = ontology.collapse(table, tree, 1, {}, {"sugar": "sweetener", "honey": "sweetener"})
    assert out.loc["sugar"].sum() == 100 and out.loc["honey"].sum() == 100


def test_synonyms_keep_processing_apart():
    assert normalise.canonical_name("cornmeal") == "cornmeal"
    assert normalise.parse_off_tag("en:cornmeal") == "cornmeal"
    assert normalise.canonical_name("dried potato") == "dehydrated_potato"


def test_newick_labels_with_apostrophes_are_valid():
    node = {"name": None, "branch_length": 0.0, "support": 87.0, "children": [
        {"name": "Lay's Gourmet", "branch_length": 0.1},
        {"name": "Tyrrell's", "branch_length": 0.2}]}
    tree = Phylo.read(io.StringIO(render._newick(node) + ";"), "newick")
    assert sorted(t.name for t in tree.get_terminals()) == ["Lay's_Gourmet", "Tyrrell's"]
    assert tree.root.confidence == 87


def test_undefined_distances_raise():
    table = pd.DataFrame({"a": [0.0, 0.0, 5.0], "b": [0.0, 0.0, 5.0]}, index=["X", "Y", "Z"])
    with pytest.raises(ValueError):
        render.build_distance_matrix(table, "braycurtis")
