"""Ingredient tree that relates ingredients to one another (data/ingredient_tree.csv).

Each ingredient sits under the plant or material it comes from (potato flour under potato
products, as in the Open Food Facts taxonomy), and these under broad groups by function
(starchy bases, fats, salts, seasonings, additives, water). An ingredient that is not a node
of the tree is attached under the node listed for its composition class, as a leaf of its
own; failing that, under the node for its food category (sweeteners, dairy, protein foods
...), and otherwise under "other". The tree is used to collapse ingredients to a coarser level and to compute weighted
UniFrac distances (Lozupone et al. 2007, Applied and Environmental Microbiology 73:1576-1585)
with every branch of length 1.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

TREE_CSV = Path(__file__).with_name("data") / "ingredient_tree.csv"
LEVELS = {"group": 1, "source": 2}


class IngredientTree:
    def __init__(self, path: Path = TREE_CSV):
        self.parent: dict[str, str | None] = {}
        self.attach: dict[str, str] = {}
        self.attach_category: dict[str, str] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                node = row["node"].strip()
                if node in self.parent:
                    raise ValueError(f"{path.name}: node '{node}' is listed twice")
                self.parent[node] = row["parent"].strip() or None
                for cls in filter(None, (c.strip() for c in row["classes"].split(";"))):
                    self.attach[cls] = node
                for cat in filter(None, (c.strip() for c in (row.get("categories") or "").split(";"))):
                    self.attach_category[cat] = node
        roots = [n for n, p in self.parent.items() if p is None]
        if len(roots) != 1:
            raise ValueError(f"{path.name}: the tree needs exactly one root, found {roots}")
        self.root = roots[0]
        for node, par in self.parent.items():
            if par is not None and par not in self.parent:
                raise ValueError(f"{path.name}: parent '{par}' of '{node}' is not a node")
        for node in self.parent:
            self.path(node)
        if "other" not in self.parent:
            raise ValueError(f"{path.name}: the tree needs an 'other' node for unknown classes")

    def children(self, node: str) -> list[str]:
        return [n for n, p in self.parent.items() if p == node]

    def path(self, node: str) -> list[str]:
        """Nodes from the root down to node."""
        out, seen = [], set()
        while node is not None:
            if node in seen:
                raise ValueError(f"cycle in the ingredient tree at '{node}'")
            seen.add(node)
            out.append(node)
            node = self.parent[node]
        return out[::-1]

    def node_for(self, name: str, cls: str = "", category: str = "") -> str:
        """The node that holds an ingredient's mass.

        A name that is already a node (a leaf, or an internal node after collapsing) is used
        as it is, so its mass sits at that node. Any other name becomes a new leaf under the
        node for its composition class, else for its food category, else under "other".
        """
        if name in self.parent:
            return name
        by_class = self.attach.get(cls)
        if by_class == "other":
            by_class = None
        self.parent[name] = by_class or self.attach_category.get(category) or "other"
        return name

    def is_under(self, node: str, ancestor: str) -> bool:
        return ancestor in self.path(node)

    def ancestor_at(self, node: str, level: int) -> str:
        path = self.path(node)
        return path[min(level, len(path) - 1)]


def nodes_of(pivot: pd.DataFrame, tree: IngredientTree, classes: dict[str, str] | None = None,
             categories: dict[str, str] | None = None) -> list[str]:
    classes, categories = classes or {}, categories or {}
    return [tree.node_for(ing, classes.get(ing, ""), categories.get(ing, ""))
            for ing in pivot.columns]


def collapse(pivot: pd.DataFrame, tree: IngredientTree, level: int,
             classes: dict[str, str] | None = None,
             categories: dict[str, str] | None = None) -> pd.DataFrame:
    """Sum ingredient columns into their ancestors at the given depth (root = 0)."""
    nodes = nodes_of(pivot, tree, classes, categories)
    groups = [tree.ancestor_at(node, level) for node in nodes]
    return pivot.T.groupby(groups, sort=False).sum().T


def share_under(pivot: pd.DataFrame, tree: IngredientTree, ancestor: str,
                classes: dict[str, str] | None = None,
                categories: dict[str, str] | None = None) -> pd.Series:
    """Percentage of each product's composition placed under one node (e.g. "other")."""
    nodes = nodes_of(pivot, tree, classes, categories)
    cols = [c for c, node in zip(pivot.columns, nodes) if tree.is_under(node, ancestor)]
    return pivot[cols].sum(axis=1) / pivot.sum(axis=1) * 100


def weighted_unifrac(pivot: pd.DataFrame, tree: IngredientTree,
                     classes: dict[str, str] | None = None,
                     categories: dict[str, str] | None = None) -> np.ndarray:
    """Normalised weighted UniFrac (condensed form), every branch of length 1.

    For each branch, the share of a product's composition below it is compared between two
    products; the sum of absolute differences is divided by the sum of the shares, so the
    distance lies between 0 (same composition) and 1 (nothing below any common branch).
    Mass on an internal node (after collapsing) counts on the branches above that node only.
    """
    leaves = nodes_of(pivot, tree, classes, categories)
    props = pivot.values / pivot.values.sum(axis=1, keepdims=True)
    branches: dict[str, np.ndarray] = {}
    for j, leaf in enumerate(leaves):
        for node in tree.path(leaf)[1:]:
            branches.setdefault(node, np.zeros(len(pivot)))
            branches[node] = branches[node] + props[:, j]
    mass = np.array(list(branches.values())).T
    n = len(pivot)
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            num = np.abs(mass[a] - mass[b]).sum()
            den = (mass[a] + mass[b]).sum()
            out.append(num / den if den > 0 else np.nan)
    return np.array(out)
