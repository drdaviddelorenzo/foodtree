"""Stage 3: build trees (UPGMA or Neighbor Joining) and ingredient heatmaps from foods.csv.

    python render.py --input foods.csv --output analyses/crisps/ --name "Crisps" [--method upgma|nj|both] [--distance braycurtis]
"""

import argparse
import datetime
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

import diagnostics
import ontology

warnings.filterwarnings("ignore")


STYLE = {
    "bg": "#0d0d0d",
    "fg": "#f0f0f0",
    "accent": "#e8c547",
    "branch": "#a0a0a0",
    "leaf": "#ffffff",
    "grid": "#2a2a2a",
    "font_family": "DejaVu Sans",
    "dpi": 150,
    "fig_portrait": (7.2, 9.6),
    "fig_square": (7.2, 7.2),
    "fig_wide": (12.0, 7.0),
}

HEATMAP_CMAP = "YlOrRd"
KEY_CATEGORIES = {"protein", "base", "fat", "sweetener", "dairy", "stimulant"}

PSEUDOCOUNT = 0.01


def apply_dark_style(fig, ax):
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["bg"])
    ax.tick_params(colors=STYLE["fg"])
    ax.xaxis.label.set_color(STYLE["fg"])
    ax.yaxis.label.set_color(STYLE["fg"])
    ax.title.set_color(STYLE["accent"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["grid"])


def apply_min_pct(pivot: pd.DataFrame, min_pct: float, unit: str = "ingredient") -> pd.DataFrame:
    """Drop columns whose maximum across products is below min_pct."""
    if min_pct <= 0:
        return pivot
    before = len(pivot.columns)
    pivot = pivot.loc[:, pivot.max(axis=0) >= min_pct]
    dropped = before - len(pivot.columns)
    if dropped:
        print(f"  Dropped {dropped} {unit}(s) with max < {min_pct}% "
              f"({len(pivot.columns)} remaining)")
    empty = pivot.index[pivot.sum(axis=1) <= 0].tolist()
    if empty:
        raise ValueError(f"Nothing left after --min-pct {min_pct} for: "
                         f"{', '.join(empty)}. Lower --min-pct.")
    return pivot


def load_foods(csv_path: Path,
               min_pct: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load foods.csv into a pivot table, drop minor ingredients, and return the
    category table and each ingredient's composition class."""
    df = pd.read_csv(csv_path)
    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce").fillna(0)

    pivot = df.pivot_table(
        index="FoodItem", columns="Ingredient", values="Percentage", aggfunc="sum"
    ).fillna(0)

    row_sums = pivot.sum(axis=1)
    empty = row_sums[row_sums <= 0].index.tolist()
    if empty:
        raise ValueError(f"No ingredient percentages for: {', '.join(empty)}")
    pivot = pivot.div(row_sums, axis=0) * 100

    pivot = apply_min_pct(pivot, min_pct)

    meta = pd.DataFrame()
    if "Category" in df.columns:
        meta = df[["Ingredient", "Category"]].drop_duplicates().set_index("Ingredient")
        meta = meta[meta.index.isin(pivot.columns)]

    classes: dict[str, str] = {}
    if "Class" in df.columns:
        for ing, cls in zip(df["Ingredient"], df["Class"].fillna("")):
            if cls and ing not in classes:
                classes[ing] = cls

    return pivot, meta, classes


def bray_curtis(pivot: pd.DataFrame) -> np.ndarray:
    """Bray-Curtis dissimilarity (condensed form)."""
    return pdist(pivot.values, metric="braycurtis")


def aitchison(pivot: pd.DataFrame) -> np.ndarray:
    """Aitchison distance: Euclidean distance on CLR-transformed compositions."""
    X = pivot.values + PSEUDOCOUNT
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    clr = log_X - log_X.mean(axis=1, keepdims=True)
    return pdist(clr, metric="euclidean")


def euclidean(pivot: pd.DataFrame) -> np.ndarray:
    """Euclidean distance on raw percentage vectors."""
    return pdist(pivot.values, metric="euclidean")


def jaccard(pivot: pd.DataFrame) -> np.ndarray:
    """Jaccard dissimilarity on binarised presence/absence."""
    binary = (pivot.values > 0).astype(float)
    return pdist(binary, metric="jaccard")


DISTANCE_FUNCTIONS = {
    "braycurtis": bray_curtis,
    "aitchison":  aitchison,
    "euclidean":  euclidean,
    "jaccard":    jaccard,
}


_CONTEXT: dict = {"classes": {}, "categories": {}, "tree": None}


def unifrac(pivot: pd.DataFrame) -> np.ndarray:
    """Weighted UniFrac over the ingredient tree (see ontology.py)."""
    tree = _CONTEXT["tree"] or ontology.IngredientTree()
    return ontology.weighted_unifrac(pivot, tree, _CONTEXT["classes"], _CONTEXT["categories"])


DISTANCE_FUNCTIONS["unifrac"] = unifrac


def build_distance_matrix(pivot: pd.DataFrame, metric: str) -> np.ndarray:
    fn = DISTANCE_FUNCTIONS.get(metric)
    if fn is None:
        raise ValueError(f"Unknown distance metric '{metric}'. "
                         f"Choose from: {list(DISTANCE_FUNCTIONS)}")
    dist = fn(pivot)
    if not np.all(np.isfinite(dist)):
        labels = pivot.index.tolist()
        full = squareform(dist, checks=False)
        bad = [f"{labels[i]} / {labels[j]}" for i in range(len(labels))
               for j in range(i + 1, len(labels)) if not np.isfinite(full[i, j])]
        raise ValueError(f"Undefined {metric} distance for: {'; '.join(bad)}")
    return dist


def get_upgma_leaf_order(pivot: pd.DataFrame, metric: str) -> list[str]:
    dist = build_distance_matrix(pivot, metric)
    linked = linkage(dist, method="average")
    labels = pivot.index.tolist()
    result = dendrogram(linked, labels=labels, no_plot=True)
    return [labels[i] for i in result["leaves"]]


def render_upgma(pivot: pd.DataFrame, output_path: Path, title: str,
                 metric: str, outgroup: str | None = None,
                 support: dict | None = None) -> None:
    dist = build_distance_matrix(pivot, metric)
    labels = pivot.index.tolist()
    linked = linkage(dist, method="average")

    fig, ax = plt.subplots(figsize=STYLE["fig_portrait"])
    apply_dark_style(fig, ax)

    drawn = dendrogram(
        linked,
        labels=labels,
        orientation="left",
        ax=ax,
        color_threshold=0,
        above_threshold_color=STYLE["branch"],
    )
    if support:
        leaf_y = {leaf: 5 + 10 * i for i, leaf in enumerate(drawn["leaves"])}

        def place(node):
            if node.is_leaf():
                return leaf_y[node.id], frozenset([labels[node.id]])
            (yl, sl), (yr, sr) = place(node.left), place(node.right)
            y, leaves = (yl + yr) / 2, sl | sr
            if leaves in support:
                ax.text(node.dist, y + 1.5, f"{support[leaves]:.0f}", color=STYLE["accent"],
                        fontsize=8, ha="right", va="bottom")
            return y, leaves

        place(to_tree(linked))

    for lbl in ax.get_yticklabels():
        is_og = outgroup and lbl.get_text() == outgroup
        lbl.set_color(STYLE["accent"] if is_og else STYLE["leaf"])
        lbl.set_fontsize(11)
        lbl.set_fontfamily(STYLE["font_family"])
        lbl.set_fontweight("bold" if is_og else "normal")

    ax.set_xlabel(f"Distance ({metric})" + ("  ·  numbers: % of replicate trees with that group"
                                              if support else ""), color=STYLE["fg"], fontsize=10)
    ax.set_title(f"{title}\nIngredient Similarity Tree (UPGMA · {metric})",
                 color=STYLE["accent"], fontsize=13, pad=14, fontweight="bold")
    if outgroup:
        fig.text(0.02, 0.01, f"◆ reference (highlighted; UPGMA is not rooted on it): {outgroup}",
                 color=STYLE["accent"], fontsize=8, va="bottom")
    fig.text(0.98, 0.02, "FoodTree", color=STYLE["grid"], fontsize=8,
             ha="right", va="bottom", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  ✓ UPGMA tree ({metric}) → {output_path.name}")


def _nj_leaf_order(clade) -> list[str]:
    """DFS traversal to get leaf names in visual top-to-bottom order."""
    if clade.is_terminal():
        return [clade.name]
    order = []
    for child in clade.clades:
        order.extend(_nj_leaf_order(child))
    return order


def _draw_nj_clade(ax, clade, parent_x: float,
                   leaf_y: dict, max_depth: float,
                   outgroup: str | None = None, support: dict | None = None,
                   labels: list[str] | None = None, labelled: set | None = None) -> float:
    """Recursively draw NJ tree branches as a horizontal dendrogram."""
    branch_len = clade.branch_length or 0.0
    x = parent_x + branch_len

    if clade.is_terminal():
        y = leaf_y[clade.name]
        is_og = outgroup and clade.name == outgroup
        ax.plot([parent_x, x], [y, y],
                color=STYLE["accent"] if is_og else STYLE["branch"],
                lw=2.0 if is_og else 1.5, solid_capstyle="butt")
        label = f"◆ {clade.name}" if is_og else clade.name
        ax.text(x + max_depth * 0.015, y, label,
                va="center", ha="left",
                color=STYLE["accent"] if is_og else STYLE["leaf"],
                fontsize=11, fontweight="bold" if is_og else "normal",
                fontfamily=STYLE["font_family"])
        return y

    labelled = set() if labelled is None else labelled
    child_ys = [_draw_nj_clade(ax, child, x, leaf_y, max_depth, outgroup, support, labels,
                               labelled) for child in clade.clades]

    node_y = (min(child_ys) + max(child_ys)) / 2
    if support and labels:
        key = diagnostics.split_key(frozenset(t.name for t in clade.get_terminals()), labels)
        if key in support and key not in labelled:
            labelled.add(key)
            ax.text(x, node_y + 0.15, f"{support[key]:.0f}", color=STYLE["accent"],
                    fontsize=8, ha="right", va="bottom")

    ax.plot([x, x], [min(child_ys), max(child_ys)],
            color=STYLE["branch"], lw=1.5, solid_capstyle="butt")
    ax.plot([parent_x, x], [node_y, node_y],
            color=STYLE["branch"], lw=1.5, solid_capstyle="butt")

    return node_y


def build_nj_tree(pivot: pd.DataFrame, metric: str, outgroup: str | None = None):
    """Neighbor Joining tree, rooted on the outgroup if given, otherwise at the midpoint."""
    labels = pivot.index.tolist()
    full = squareform(build_distance_matrix(pivot, metric))
    full = (full + full.T) / 2
    np.fill_diagonal(full, 0.0)
    dm = DistanceMatrix(names=labels)
    dm.matrix = [full[i, :i + 1].tolist() for i in range(len(labels))]
    tree = DistanceTreeConstructor().nj(dm)
    if outgroup and outgroup in labels:
        tree.root_with_outgroup(outgroup)
    else:
        tree.root_at_midpoint()
    return tree


def render_nj(pivot: pd.DataFrame, output_path: Path, title: str,
              metric: str, outgroup: str | None = None, support: dict | None = None) -> None:
    labels = pivot.index.tolist()
    tree = build_nj_tree(pivot, metric, outgroup)

    leaf_order = _nj_leaf_order(tree.root)
    leaf_y = {name: i for i, name in enumerate(leaf_order)}
    n = len(leaf_order)

    def _max_depth(clade, depth=0.0) -> float:
        d = depth + (clade.branch_length or 0.0)
        if clade.is_terminal():
            return d
        return max(_max_depth(c, d) for c in clade.clades)

    max_depth = _max_depth(tree.root) or 1.0

    fig, ax = plt.subplots(figsize=STYLE["fig_portrait"])
    apply_dark_style(fig, ax)

    root_branch = tree.root.branch_length or 0.0
    _draw_nj_clade(ax, tree.root, root_branch, leaf_y, max_depth, outgroup, support, labels)

    ax.set_xlim(-max_depth * 0.02, max_depth * 1.30)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(f"Distance ({metric})" + ("  ·  numbers: % of replicate trees with that split"
                                              if support else ""), color=STYLE["fg"], fontsize=10)
    ax.set_yticks([])
    rooting = "outgroup rooted" if (outgroup and outgroup in labels) else "midpoint rooted"
    ax.set_title(
        f"{title}\nIngredient Similarity Tree (NJ · {metric} · {rooting})",
        color=STYLE["accent"], fontsize=13, pad=14, fontweight="bold",
    )
    if outgroup and outgroup in labels:
        fig.text(0.02, 0.01, f"◆ outgroup (reference): {outgroup}",
                 color=STYLE["accent"], fontsize=8, va="bottom")
    fig.text(0.98, 0.02, "FoodTree", color=STYLE["grid"], fontsize=8,
             ha="right", va="bottom", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  ✓ NJ tree ({metric}) → {output_path.name}")


def _node(name, branch_length: float, parent_depth: float) -> dict:
    return {"name": name, "branch_length": round(branch_length, 6),
            "depth": round(parent_depth + branch_length, 6)}


def _upgma_dict(pivot: pd.DataFrame, metric: str) -> dict:
    labels = pivot.index.tolist()
    root = to_tree(linkage(build_distance_matrix(pivot, metric), method="average"))

    def walk(n, parent_height: float, parent_depth: float) -> dict:
        d = _node(labels[n.id] if n.is_leaf() else None,
                  max(parent_height - n.dist, 0.0), parent_depth)
        if not n.is_leaf():
            d["children"] = [walk(n.left, n.dist, d["depth"]), walk(n.right, n.dist, d["depth"])]
        return d

    return walk(root, root.dist, 0.0)


def _clade_dict(clade, parent_depth: float = 0.0) -> dict:
    d = _node(clade.name if clade.is_terminal() else None,
              clade.branch_length or 0.0, parent_depth)
    if not clade.is_terminal():
        d["children"] = [_clade_dict(c, d["depth"]) for c in clade.clades]
    return d


def _leaf_names(node: dict) -> list[str]:
    if "children" not in node:
        return [node["name"]]
    return [name for c in node["children"] for name in _leaf_names(c)]


def _newick_label(name: str) -> str:
    """A Newick label: spaces become underscores; labels with other special characters are
    quoted, doubling any single quote (Newick standard)."""
    label = name.replace(" ", "_")
    if any(ch in label for ch in "'(),:;[]"):
        label = "'" + label.replace("'", "''") + "'"
    return label


def _annotate_support(node: dict, support: dict | None, key=lambda leaves: leaves,
                      done: set | None = None) -> frozenset:
    """Add a 'support' field (percentage) to internal nodes found in support, once per
    group (a midpoint root splits one unrooted edge into two, which share a split)."""
    done = set() if done is None else done
    if "children" not in node:
        return frozenset([node["name"]])
    leaves = frozenset().union(*(_annotate_support(c, support, key, done)
                                 for c in node["children"]))
    if support and key(leaves) in support and key(leaves) not in done:
        done.add(key(leaves))
        node["support"] = round(support[key(leaves)], 1)
    return leaves


def _newick(node: dict) -> str:
    name = _newick_label(node["name"] or (f"{node['support']:g}" if "support" in node else ""))
    inner = "(" + ",".join(_newick(c) for c in node["children"]) + ")" if "children" in node else ""
    return f"{inner}{name}:{node['branch_length']}"


def export_tree_data(pivot: pd.DataFrame, metric: str, outgroup: str | None,
                     output_dir: Path, slug: str, analysis_name: str,
                     methods: tuple[str, ...] = ("upgma", "nj"),
                     supports: dict | None = None) -> None:
    """Write the trees as Newick and JSON, and the distance matrix as CSV, to redraw them
    elsewhere. Internal nodes carry their support (% of replicate trees) when computed."""
    supports = supports or {}
    labels = pivot.index.tolist()
    builders = {"upgma": lambda: _upgma_dict(pivot, metric),
                "nj": lambda: _clade_dict(build_nj_tree(pivot, metric, outgroup).root)}
    trees = {m: builders[m]() for m in methods}
    if "upgma" in trees:
        _annotate_support(trees["upgma"], supports.get("upgma"))
    if "nj" in trees:
        _annotate_support(trees["nj"], supports.get("nj"),
                          key=lambda leaves: diagnostics.split_key(leaves, labels))
    for method, tree in trees.items():
        base = output_dir / f"{slug}-tree-{method}-{metric}"
        info = {"analysis": analysis_name, "method": method, "distance": metric,
                "outgroup": outgroup, "leaf_order": _leaf_names(tree), "tree": tree}
        base.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        base.with_suffix(".nwk").write_text(_newick(tree) + ";\n")
    labels = pivot.index.tolist()
    matrix = pd.DataFrame(squareform(build_distance_matrix(pivot, metric)), index=labels, columns=labels)
    matrix.round(6).to_csv(output_dir / f"{slug}-distance-{metric}.csv")
    print(f"  ✓ Tree data (Newick + JSON) and distance matrix (CSV) → {slug}-tree-*.nwk/.json, {slug}-distance-{metric}.csv")


def get_key_ingredients(pivot: pd.DataFrame, meta: pd.DataFrame,
                        max_ingredients: int = 8) -> list[str]:
    """Select the most informative ingredients for the heatmap."""
    if meta.empty:
        variances = pivot.var().sort_values(ascending=False)
        return variances.index.tolist()[:max_ingredients]

    protein_ings = meta[meta["Category"] == "protein"].index.tolist()
    other_key_ings = meta[meta["Category"].isin(KEY_CATEGORIES - {"protein"})].index.tolist()

    protein_ings = [i for i in protein_ings if i in pivot.columns]
    other_key_ings = [i for i in other_key_ings if i in pivot.columns]

    variances = pivot.var()
    protein_ings.sort(key=lambda x: -variances.get(x, 0))
    other_key_ings.sort(key=lambda x: -variances.get(x, 0))

    candidates = protein_ings + other_key_ings
    for ing in variances.sort_values(ascending=False).index:
        if ing not in candidates:
            candidates.append(ing)
        if len(candidates) >= max_ingredients:
            break

    return candidates[:max_ingredients]


def render_heatmap_combined(pivot: pd.DataFrame, ingredients: list[str],
                            output_path: Path, title: str,
                            leaf_order: list[str]) -> None:
    """Render a combined ingredient overview: products (rows) × key ingredients (columns)."""
    data = pivot.reindex(leaf_order)[ingredients].fillna(0)

    fig, ax = plt.subplots(figsize=(max(7.2, len(ingredients) * 1.2), 7.2))
    apply_dark_style(fig, ax)

    normed = data.copy()
    for col in data.columns:
        col_max = data[col].max()
        normed[col] = data[col] / col_max if col_max > 0 else 0

    im = ax.imshow(normed.values, aspect="auto", cmap=HEATMAP_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_yticks(range(len(leaf_order)))
    ax.set_yticklabels(leaf_order, color=STYLE["leaf"], fontsize=10)
    ax.set_xticks(range(len(ingredients)))
    ax.set_xticklabels(
        [i.replace("_", " ").title() for i in ingredients],
        color=STYLE["fg"], fontsize=9, rotation=35, ha="right",
    )

    for i, row_name in enumerate(leaf_order):
        for j, col_name in enumerate(ingredients):
            val = data.loc[row_name, col_name]
            if val > 0:
                text_color = "white" if normed.loc[row_name, col_name] > 0.5 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color=text_color, fontsize=8, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Relative content")
    cb.ax.yaxis.label.set_color(STYLE["fg"])
    cb.ax.tick_params(colors=STYLE["fg"])

    ax.set_title(f"{title} — Ingredient Overview",
                 color=STYLE["accent"], fontsize=13, pad=12, fontweight="bold")
    fig.text(0.98, 0.02, "FoodTree", color=STYLE["grid"], fontsize=8,
             ha="right", va="bottom", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  ✓ Ingredient overview → {output_path.name}")


def render_table(pivot: pd.DataFrame, ingredients: list[str],
                 output_path: Path, title: str, leaf_order: list[str]) -> None:
    """Save ingredient percentages as a plain CSV."""
    data = pivot.reindex(leaf_order)[ingredients].fillna(0)
    csv_path = output_path.with_suffix(".csv")
    data[ingredients].round(1).to_csv(csv_path)
    print(f"  ✓ Ingredient table (CSV) → {csv_path.name}")


def render_distance_matrix(pivot: pd.DataFrame, output_path: Path,
                           title: str, metric: str,
                           leaf_order: list[str] | None = None) -> None:
    """Render the pairwise distance matrix between all products as a heatmap."""
    dist_condensed = build_distance_matrix(pivot, metric)
    full = squareform(dist_condensed)
    labels = pivot.index.tolist()
    n = len(labels)

    if leaf_order is None:
        linked = linkage(dist_condensed, method="average")
        dend = dendrogram(linked, labels=labels, no_plot=True)
        leaf_order = [labels[i] for i in dend["leaves"]]

    ordered_labels = leaf_order
    order = [labels.index(name) for name in leaf_order]
    full_ordered = full[np.ix_(order, order)]

    fig_size = max(5.0, n * 0.8 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    apply_dark_style(fig, ax)

    im = ax.imshow(full_ordered, cmap="YlOrRd", vmin=0,
                   vmax=full_ordered.max(), interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered_labels, rotation=40, ha="right",
                       color=STYLE["fg"], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered_labels, color=STYLE["fg"], fontsize=10)

    for i in range(n):
        for j in range(n):
            val = full_ordered[i, j]
            text_color = "white" if val > full_ordered.max() * 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_color)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      label=f"{metric} distance")
    cb.ax.yaxis.label.set_color(STYLE["fg"])
    cb.ax.tick_params(colors=STYLE["fg"])

    ax.set_title(f"{title} — Pairwise Distance Matrix ({metric})",
                 color=STYLE["accent"], fontsize=13, pad=12, fontweight="bold")
    fig.text(0.98, 0.01, "FoodTree", color=STYLE["grid"], fontsize=8,
             ha="right", va="bottom", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  ✓ Distance matrix → {output_path.name}")


MANIFEST_HEADER = "Files written by this run:"


def output_files(slug: str, metric: str, methods: tuple[str, ...], export_data: bool) -> list[str]:
    """Names of the files one render writes for this analysis and metric."""
    files = [f"{slug}-tree-{m}-{metric}.png" for m in methods]
    if export_data:
        files += [f"{slug}-tree-{m}-{metric}{ext}" for m in methods for ext in (".json", ".nwk")]
        files.append(f"{slug}-distance-{metric}.csv")
    files += [f"{slug}-distance-{metric}.png", f"{slug}-heatmap-{metric}.png",
              f"{slug}-table-{metric}.csv", f"{slug}-pcoa-{metric}.png",
              f"{slug}-pcoa-{metric}.csv", f"{slug}-diagnostics-{metric}.txt"]
    return files


def manifest_path(output_dir: Path, slug: str, metric: str) -> Path:
    return output_dir / f"{slug}-{metric}-outputs.txt"


def read_manifest(path: Path) -> list[str]:
    """Files listed by an earlier render's manifest (empty if there is none)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if MANIFEST_HEADER not in lines:
        return []
    return [line.strip() for line in lines[lines.index(MANIFEST_HEADER) + 1:] if line.strip()]


def write_manifest(path: Path, files: list[str], settings: dict) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"FoodTree render, {ts}"] + [f"{k}: {v}" for k, v in settings.items()]
    lines += ["", MANIFEST_HEADER] + [f"  {f}" for f in files]
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def publish_outputs(tmp_dir: Path, output_dir: Path, files: list[str], slug: str,
                    metric: str, settings: dict) -> None:
    """Move a finished render and its manifest into place, then delete the files that the
    previous render of this analysis and metric listed and this render did not write.

    Files being replaced or removed are first moved aside; if any step fails they are put
    back, so the folder holds either the previous render or the new one. If some cannot be
    put back, the copies are kept in a .foodtree-recovery-<time> folder and an error says so.
    """
    manifest = manifest_path(output_dir, slug, metric)
    previous = set(read_manifest(manifest))
    owned = previous & set(output_files(slug, metric, ("upgma", "nj"), True))
    obsolete = sorted(owned - set(files))
    backup = tmp_dir / ".previous"
    backup.mkdir()
    moved_aside: list[str] = []
    placed: list[str] = []
    try:
        for name in obsolete:
            if (output_dir / name).exists():
                os.replace(output_dir / name, backup / name)
                moved_aside.append(name)
        for name in files + [manifest.name]:
            dest = output_dir / name
            if dest.exists():
                if name not in previous and name != manifest.name:
                    print(f"  Overwriting {name} (not listed by a previous render)")
                os.replace(dest, backup / name)
                moved_aside.append(name)
            if name == manifest.name:
                write_manifest(manifest, files, settings)
            else:
                os.replace(tmp_dir / name, dest)
            placed.append(name)
    except BaseException as exc:
        unrestored = []
        for name in placed:
            try:
                (output_dir / name).unlink(missing_ok=True)
            except OSError:
                unrestored.append(name)
        for name in moved_aside:
            try:
                os.replace(backup / name, output_dir / name)
            except OSError:
                unrestored.append(name)
        if unrestored:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            keep = output_dir / f".foodtree-recovery-{stamp}"
            try:
                os.replace(backup, keep)
                where, keep_tmp = keep, False
            except OSError:
                where, keep_tmp = backup, True
            raise RecoveryNeeded(
                f"Publishing failed ({exc}) and these files could not be put back: "
                f"{', '.join(unrestored)}. The previous versions are in {where}.",
                keep_tmp) from exc
        raise
    for name in moved_aside:
        if name in obsolete:
            print(f"  Removed {name} (written by the previous render, not by this one)")


class RecoveryNeeded(RuntimeError):
    """Publishing failed and previous outputs are kept for manual recovery."""

    def __init__(self, message: str, keep_tmp: bool):
        super().__init__(message)
        self.keep_tmp = keep_tmp


def compute_supports(foods_csv: Path, pivot: pd.DataFrame, metric: str, methods, level: str,
                     classes: dict, categories: dict, tree, n: int) -> tuple[dict, int, int, str]:
    """Branch support from the replicate compositions written by normalise.py.

    Each replicate table goes through the same steps as the main one (shares of 100,
    collapsing to the level, the same columns kept) before its tree is built. Every branch
    of the reference tree starts at zero, so unsupported branches are reported as 0. A
    replicate whose distances are undefined (a product with nothing left after filtering)
    is excluded and counted. Returns supports, replicates used, replicates excluded, and a
    note when support could not be computed.
    """
    if n <= 0:
        return {}, 0, 0, "--support 0 was given"
    replicates, why = diagnostics.load_replicates(foods_csv)
    if replicates is None:
        return {}, 0, 0, why
    raw = pd.read_csv(foods_csv)
    point = raw.pivot_table(index="FoodItem", columns="Ingredient", values="Percentage",
                            aggfunc="sum").fillna(0)
    labels = pivot.index.tolist()
    reference_dist = build_distance_matrix(pivot, metric)
    counts = {m: {} for m in methods}
    if "upgma" in methods:
        counts["upgma"] = {c: 0 for c in diagnostics.upgma_clades(reference_dist, labels)}
    if "nj" in methods:
        counts["nj"] = {sp: 0 for sp in diagnostics.splits(build_nj_tree(pivot, metric), labels)}
    used = excluded = 0
    for table in diagnostics.replicate_tables(replicates, point, n):
        table = table.reindex(index=labels).fillna(0)
        table = table.div(table.sum(axis=1), axis=0) * 100
        if level != "ingredient":
            table = ontology.collapse(table, tree, ontology.LEVELS[level], classes, categories)
        table = table.reindex(columns=pivot.columns, fill_value=0)
        if (table.sum(axis=1) <= 0).any():
            excluded += 1
            continue
        try:
            dist = build_distance_matrix(table, metric)
        except ValueError:
            excluded += 1
            continue
        used += 1
        if "upgma" in methods:
            for clade in diagnostics.upgma_clades(dist, labels):
                if clade in counts["upgma"]:
                    counts["upgma"][clade] += 1
        if "nj" in methods:
            for split in diagnostics.splits(build_nj_tree(table, metric), labels):
                if split in counts["nj"]:
                    counts["nj"][split] += 1
    if not used:
        return {}, 0, excluded, "every replicate had undefined distances"
    return ({m: {k: 100 * v / used for k, v in c.items()} for m, c in counts.items()},
            used, excluded, "")


def render_pcoa(pivot: pd.DataFrame, metric: str, output_png: Path, output_csv: Path,
                title: str) -> tuple[np.ndarray, float]:
    labels = pivot.index.tolist()
    coords, explained, negative = diagnostics.pcoa(build_distance_matrix(pivot, metric), labels)
    coords.round(6).to_csv(output_csv)
    fig, ax = plt.subplots(figsize=STYLE["fig_square"])
    apply_dark_style(fig, ax)
    x = coords.iloc[:, 0] if coords.shape[1] > 0 else pd.Series(0.0, index=labels)
    y = coords.iloc[:, 1] if coords.shape[1] > 1 else pd.Series(0.0, index=labels)
    ax.scatter(x, y, color=STYLE["accent"], s=30)
    for name in labels:
        ax.annotate(name, (x[name], y[name]), color=STYLE["leaf"], fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    pct = [f"{100 * e:.0f}%" for e in explained[:2]] + ["0%", "0%"]
    ax.set_xlabel(f"PCo1 ({pct[0]})", color=STYLE["fg"])
    ax.set_ylabel(f"PCo2 ({pct[1]})", color=STYLE["fg"])
    ax.set_title(f"{title} — Principal coordinates ({metric})", color=STYLE["accent"],
                 fontsize=13, pad=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_png, dpi=STYLE["dpi"], bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"  ✓ PCoA → {output_png.name}")
    return explained, negative


def write_diagnostics(path: Path, pivot: pd.DataFrame, metric: str, methods, supports: dict,
                      n_used: int, explained: np.ndarray, negative: float,
                      n_excluded: int = 0, why: str = "") -> None:
    dist = build_distance_matrix(pivot, metric)
    lines = [f"FoodTree diagnostics ({metric})", ""]
    if len(pivot) > 2:
        r = diagnostics.cophenetic_correlation(dist)
        lines += [f"Cophenetic correlation of the UPGMA tree: {r:.3f}",
                  "  (agreement between the distances and the tree; well below 0.8, the tree "
                  "distorts them)", ""]
    shares = ", ".join(f"PCo{i + 1} {100 * e:.1f}%" for i, e in enumerate(explained[:3]))
    lines += [f"PCoA: {shares}" if shares else "PCoA: no positive axis",
              f"  share of variation on negative eigenvalues (non-Euclidean part): {100 * negative:.1f}%",
              ""]
    if n_used:
        lines.append(f"Branch support from {n_used} replicate compositions (% of replicate trees "
                     "with each group of products):")
        if n_excluded:
            lines.append(f"  {n_excluded} further replicates were excluded because some distance "
                         "was undefined (a product with nothing left after filtering); support is "
                         f"over the {n_used} used.")
        for m in methods:
            lines.append(f"  {m.upper()}:")
            reference = (diagnostics.upgma_clades(dist, pivot.index.tolist()) if m == "upgma"
                         else diagnostics.splits(build_nj_tree(pivot, metric), pivot.index.tolist()))
            for group in sorted(reference, key=len):
                lines.append(f"    {supports.get(m, {}).get(group, 0.0):5.1f}  "
                             + ", ".join(sorted(group)))
    else:
        lines.append(f"Branch support: not computed ({why}).")
    lines += ["", "Support measures robustness to what the labels leave undetermined. It does "
              "not cover errors in the labels, the ingredient tree, the distance or the tree "
              "method."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_all(foods_csv: Path, output_dir: Path, analysis_name: str,
               method: str = "upgma", metric: str = "braycurtis",
               outgroup: str | None = None,
               min_pct: float = 2.0, export_data: bool = True,
               level: str = "ingredient", support: int = 100) -> None:
    """Render the trees, distance matrix, heatmap and table, and export the tree data.

    level collapses ingredients to their source (e.g. all potato products) or group (e.g.
    starchy bases) in the ingredient tree before distances are computed.

    Files are written to a temporary folder and moved into output_dir only when all of them
    have been produced, so a failed render leaves the previous outputs untouched.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = analysis_name.lower().replace(" ", "-")

    print(f"\nLoading {foods_csv}...")
    if level != "ingredient" and level not in ontology.LEVELS:
        raise ValueError(f"Unknown level '{level}'. Choose from: ingredient, "
                         + ", ".join(ontology.LEVELS))
    pivot, meta, classes = load_foods(foods_csv, min_pct=min_pct if level == "ingredient" else 0)
    categories = meta["Category"].to_dict() if not meta.empty else {}
    tree = ontology.IngredientTree()
    other = ontology.share_under(pivot, tree, "other", classes, categories)
    for product, share in other[other > 5].items():
        print(f"  WARNING: {share:.1f}% of {product} is under 'other' in the ingredient tree "
              "(unknown class and category); it is compared as one undifferentiated group")
    orig_classes, orig_categories = dict(classes), dict(categories)
    if level != "ingredient":
        slug = f"{slug}-{level}"
        pivot = ontology.collapse(pivot, tree, ontology.LEVELS[level], classes, categories)
        pivot = apply_min_pct(pivot, min_pct, unit=level)
        meta, classes, categories = pd.DataFrame(), {}, {}
    _CONTEXT.update(classes=classes, categories=categories, tree=tree)
    print(f"  {len(pivot)} products × {len(pivot.columns)} "
          f"{'ingredients' if level == 'ingredient' else level + 's'}")

    if outgroup:
        if outgroup in pivot.index:
            print(f"  Outgroup (reference): {outgroup}")
        else:
            print(f"  WARNING: outgroup '{outgroup}' not found in foods.csv — ignored")
            outgroup = None

    methods = ("upgma", "nj") if method == "both" else (method,)
    if any(m not in ("upgma", "nj") for m in methods):
        raise ValueError(f"Unknown method '{method}'. Choose from: upgma, nj, both")
    files = output_files(slug, metric, methods, export_data)

    supports, n_used, n_excluded, why = {}, 0, 0, "fewer than three products"
    if len(pivot) > 2:
        print(f"\nBranch support from replicate compositions...")
        supports, n_used, n_excluded, why = compute_supports(
            foods_csv, pivot, metric, methods, level, orig_classes, orig_categories, tree, support)
        if n_used:
            print(f"  {n_used} replicate trees" + (f" ({n_excluded} excluded: undefined distances)"
                                                   if n_excluded else ""))
        else:
            print(f"  support not computed: {why}")

    tmp_dir = Path(tempfile.mkdtemp(prefix=".foodtree-render-", dir=output_dir))
    keep_tmp = False
    try:
        print("\nRendering trees...")
        leaf_order = get_upgma_leaf_order(pivot, metric)

        if "upgma" in methods:
            render_upgma(pivot, tmp_dir / f"{slug}-tree-upgma-{metric}.png",
                         analysis_name, metric, outgroup=outgroup,
                         support=supports.get("upgma"))
        if "nj" in methods:
            render_nj(pivot, tmp_dir / f"{slug}-tree-nj-{metric}.png",
                      analysis_name, metric, outgroup=outgroup, support=supports.get("nj"))

        if export_data:
            print("\nExporting tree data...")
            export_tree_data(pivot, metric, outgroup, tmp_dir, slug, analysis_name, methods,
                             supports)

        print("\nDiagnostics...")
        explained, negative = render_pcoa(pivot, metric, tmp_dir / f"{slug}-pcoa-{metric}.png",
                                          tmp_dir / f"{slug}-pcoa-{metric}.csv", analysis_name)
        write_diagnostics(tmp_dir / f"{slug}-diagnostics-{metric}.txt", pivot, metric, methods,
                          supports, n_used, explained, negative, n_excluded, why)

        print("\nRendering distance matrix...")
        render_distance_matrix(pivot,
                               tmp_dir / f"{slug}-distance-{metric}.png",
                               analysis_name, metric, leaf_order=leaf_order)

        key_ings = get_key_ingredients(pivot, meta)

        print(f"\nRendering ingredient overview...")
        render_heatmap_combined(
            pivot, key_ings,
            tmp_dir / f"{slug}-heatmap-{metric}.png",
            analysis_name, leaf_order,
        )

        print(f"\nSaving ingredient table (CSV)...")
        render_table(
            pivot, pivot.columns.tolist(),
            tmp_dir / f"{slug}-table-{metric}.csv",
            analysis_name, leaf_order,
        )

        missing = [f for f in files if not (tmp_dir / f).exists()]
        if missing:
            raise RuntimeError(f"Render did not produce: {', '.join(missing)}")
        publish_outputs(tmp_dir, output_dir, files, slug, metric, {
            "input": foods_csv, "method": method, "distance": metric, "min_pct": min_pct,
            "outgroup": outgroup or "none", "export": export_data, "level": level,
            "products": len(pivot), "ingredients": len(pivot.columns),
            "support replicates": n_used, "support excluded": n_excluded,
        })
    except RecoveryNeeded as exc:
        keep_tmp = exc.keep_tmp
        raise
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n✓ All assets saved to {output_dir}/ (list in {slug}-{metric}-outputs.txt)")


def main():
    parser = argparse.ArgumentParser(description="FoodTree Stage 3: Render trees and heatmaps")
    parser.add_argument("--input", required=True, help="Path to foods.csv")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--name", required=True, help="Analysis name (e.g. 'Crisps')")
    parser.add_argument("--method", choices=["upgma", "nj", "both"], default="upgma",
                        help="Tree algorithm (default: upgma)")
    parser.add_argument("--distance",
                        choices=list(DISTANCE_FUNCTIONS), default="braycurtis",
                        help="Distance metric (default: braycurtis)")
    parser.add_argument("--outgroup", default=None,
                        help="Product name to use as outgroup/reference (e.g. 'Cod Fillet')")
    parser.add_argument("--min-pct", type=float, default=2.0,
                        help="Drop ingredients whose max across all products is below this %% "
                             "(default: 2.0). Use 0 to keep everything. "
                             "Lower for analyses where key ingredients are trace amounts "
                             "(e.g. 0.1 for energy drinks to keep caffeine/taurine).")
    parser.add_argument("--no-export", action="store_true",
                        help="Do not write the tree data files (Newick, JSON, distance CSV)")
    parser.add_argument("--support", type=int, default=100,
                        help="Replicate trees for branch support (default 100; 0 to skip)")
    parser.add_argument("--level", choices=["ingredient", "source", "group"], default="ingredient",
                        help="Compare ingredients as listed, or collapsed to their source or group "
                             "in data/ingredient_tree.csv (default: ingredient)")
    args = parser.parse_args()

    render_all(
        foods_csv=Path(args.input),
        output_dir=Path(args.output),
        analysis_name=args.name,
        method=args.method,
        metric=args.distance,
        outgroup=args.outgroup,
        min_pct=args.min_pct,
        export_data=not args.no_export,
        level=args.level,
        support=args.support,
    )


if __name__ == "__main__":
    main()
