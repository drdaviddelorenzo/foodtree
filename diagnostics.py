"""How far a tree can be trusted: branch support, cophenetic correlation and PCoA.

Branch support. normalise.py writes, for each product, compositions resampled from all those
compatible with its label (foods-samples.csv). A tree is rebuilt from each replicate, and a
branch's support is the percentage of replicate trees that contain the same group of
products. It measures how robust a grouping is to what the label leaves undetermined, in
the spirit of the phylogenetic bootstrap; it does not resample characters, and it says
nothing about errors in the label itself or in the choice of distance and tree method.

Cophenetic correlation. The correlation between the distances in the matrix and the
distances implied by the UPGMA tree (the height at which two products join). Close to 1,
the tree is a faithful summary; well below about 0.8, it distorts the distances.

PCoA. Principal coordinates of the distance matrix: products drawn in a plane without
forcing a tree. If the groups of the tree also separate here, they do not depend on the tree.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage, to_tree
from scipy.spatial.distance import squareform


def load_replicates(foods_csv) -> tuple[pd.DataFrame | None, str]:
    """Replicate compositions written with foods_csv, or None and the reason they cannot be used.

    They are used only if the fingerprints recorded by normalise.py match the current
    foods.csv and samples file, so support always describes the table being drawn.
    """
    import hashlib
    import json
    from pathlib import Path
    from normalise import samples_meta_path, samples_path
    foods_csv = Path(foods_csv)
    samples, meta = samples_path(foods_csv), samples_meta_path(foods_csv)
    if not samples.exists() or not meta.exists():
        return None, "no replicate compositions (run normalise first)"
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        matches = (isinstance(info, dict) and info.get("foods_sha256") == sha(foods_csv)
                   and info.get("samples_sha256") == sha(samples))
    except (OSError, ValueError) as exc:
        return None, f"the replicate fingerprint file cannot be read ({exc}); rerun normalise"
    if not matches:
        return None, ("the replicate compositions do not belong to this foods.csv (it was "
                      "edited or replaced, or a run was interrupted); rerun normalise")
    try:
        df = pd.read_csv(samples)
    except (OSError, ValueError) as exc:
        return None, f"the replicate file cannot be read ({exc}); rerun normalise"
    return (df, "") if not df.empty else (None, "the replicate file is empty")


def replicate_tables(replicates: pd.DataFrame, point: pd.DataFrame, n: int):
    """Yield n tables (products x ingredients, as in foods.csv) built from the replicates;
    products without replicates (entered by hand) keep their point values."""
    ids = sorted(replicates["Replicate"].unique())[:n]
    have = set(replicates["FoodItem"])
    fixed = point[~point.index.isin(have)]
    by_rep = {r: g for r, g in replicates.groupby("Replicate")}
    for r in ids:
        table = by_rep[r].pivot_table(index="FoodItem", columns="Ingredient",
                                      values="Percentage", aggfunc="sum")
        yield pd.concat([table, fixed]).fillna(0)


def upgma_clades(dist: np.ndarray, labels: list[str]) -> set[frozenset]:
    """Groups of products joined by the internal nodes of a UPGMA tree (root excluded)."""
    root = to_tree(linkage(dist, method="average"))
    clades = set()

    def walk(node):
        if node.is_leaf():
            return frozenset([labels[node.id]])
        leaves = walk(node.left) | walk(node.right)
        clades.add(leaves)
        return leaves

    everything = walk(root)
    clades.discard(everything)
    return clades


def splits(tree, labels: list[str]) -> set[frozenset]:
    """Non-trivial bipartitions of an unrooted tree, each given as the side without labels[0]."""
    out = set()
    everything = frozenset(labels)
    for clade in tree.find_clades():
        side = frozenset(t.name for t in clade.get_terminals())
        if 1 < len(side) < len(labels) - 1:
            out.add(side if labels[0] not in side else everything - side)
    return out


def split_key(leaves: frozenset, labels: list[str]) -> frozenset:
    return leaves if labels[0] not in leaves else frozenset(labels) - leaves


def cophenetic_correlation(dist: np.ndarray) -> float:
    r, _ = cophenet(linkage(dist, method="average"), dist)
    return float(r)


def pcoa(dist: np.ndarray, labels: list[str]) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Classical principal coordinates: coordinates on the axes with positive eigenvalues,
    the share of variation on each, and the share carried by negative eigenvalues (a sign
    that the distance is not Euclidean)."""
    d2 = squareform(dist) ** 2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = vals > 1e-10
    coords = vecs[:, pos] * np.sqrt(vals[pos])
    total = np.abs(vals).sum()
    explained = vals[pos] / vals[pos].sum() if pos.any() else np.zeros(0)
    negative = float(np.abs(vals[vals < -1e-10]).sum() / total) if total > 0 else 0.0
    cols = [f"PCo{i + 1}" for i in range(coords.shape[1])]
    return pd.DataFrame(coords, index=labels, columns=cols), explained, negative
