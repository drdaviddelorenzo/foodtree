"""Stage 3: build trees (UPGMA or Neighbor Joining) and ingredient heatmaps from foods.csv.

    python render.py --input foods.csv --output analyses/crisps/ --name "Crisps" [--method nj] [--distance braycurtis]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

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


def load_foods(csv_path: Path,
               min_pct: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load foods.csv into a pivot table and drop minor ingredients."""
    df = pd.read_csv(csv_path)
    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce").fillna(0)

    pivot = df.pivot_table(
        index="FoodItem", columns="Ingredient", values="Percentage", aggfunc="mean"
    ).fillna(0)

    # Ensure rows sum to 100
    row_sums = pivot.sum(axis=1)
    pivot = pivot.div(row_sums, axis=0) * 100

    if min_pct > 0:
        before = len(pivot.columns)
        pivot = pivot.loc[:, pivot.max(axis=0) >= min_pct]
        dropped = before - len(pivot.columns)
        if dropped:
            print(f"  Dropped {dropped} ingredient(s) with max < {min_pct}% "
                  f"({len(pivot.columns)} remaining)")

    meta = pd.DataFrame()
    if "Category" in df.columns:
        meta = df[["Ingredient", "Category"]].drop_duplicates().set_index("Ingredient")
        meta = meta[meta.index.isin(pivot.columns)]

    return pivot, meta


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


def build_distance_matrix(pivot: pd.DataFrame, metric: str) -> np.ndarray:
    fn = DISTANCE_FUNCTIONS.get(metric)
    if fn is None:
        raise ValueError(f"Unknown distance metric '{metric}'. "
                         f"Choose from: {list(DISTANCE_FUNCTIONS)}")
    dist = fn(pivot)
    dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
    return dist


def get_upgma_leaf_order(pivot: pd.DataFrame, metric: str) -> list[str]:
    dist = build_distance_matrix(pivot, metric)
    linked = linkage(dist, method="average")
    labels = pivot.index.tolist()
    result = dendrogram(linked, labels=labels, no_plot=True)
    return [labels[i] for i in result["leaves"]]


def render_upgma(pivot: pd.DataFrame, output_path: Path, title: str,
                 metric: str, outgroup: str | None = None) -> None:
    dist = build_distance_matrix(pivot, metric)
    labels = pivot.index.tolist()
    linked = linkage(dist, method="average")

    fig, ax = plt.subplots(figsize=STYLE["fig_portrait"])
    apply_dark_style(fig, ax)

    dendrogram(
        linked,
        labels=labels,
        orientation="left",
        ax=ax,
        color_threshold=0,
        above_threshold_color=STYLE["branch"],
    )

    for lbl in ax.get_yticklabels():
        is_og = outgroup and lbl.get_text() == outgroup
        lbl.set_color(STYLE["accent"] if is_og else STYLE["leaf"])
        lbl.set_fontsize(11)
        lbl.set_fontfamily(STYLE["font_family"])
        lbl.set_fontweight("bold" if is_og else "normal")

    ax.set_xlabel(f"Distance ({metric})", color=STYLE["fg"], fontsize=10)
    subtitle = f"UPGMA · {metric}" + (" · outgroup rooted" if outgroup else "")
    ax.set_title(f"{title}\nIngredient Similarity Tree ({subtitle})",
                 color=STYLE["accent"], fontsize=13, pad=14, fontweight="bold")
    if outgroup:
        fig.text(0.02, 0.01, f"◆ outgroup (reference): {outgroup}",
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
                   outgroup: str | None = None) -> float:
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

    child_ys = [_draw_nj_clade(ax, child, x, leaf_y, max_depth, outgroup)
                for child in clade.clades]

    node_y = (min(child_ys) + max(child_ys)) / 2

    ax.plot([x, x], [min(child_ys), max(child_ys)],
            color=STYLE["branch"], lw=1.5, solid_capstyle="butt")
    ax.plot([parent_x, x], [node_y, node_y],
            color=STYLE["branch"], lw=1.5, solid_capstyle="butt")

    return node_y


def render_nj(pivot: pd.DataFrame, output_path: Path, title: str,
              metric: str, outgroup: str | None = None) -> None:
    dist_condensed = build_distance_matrix(pivot, metric)
    labels = pivot.index.tolist()
    full = squareform(dist_condensed)
    full = (full + full.T) / 2
    np.fill_diagonal(full, 0.0)

    dm = DistanceMatrix(names=labels)
    dm.matrix = [full[i, :i + 1].tolist() for i in range(len(labels))]

    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # otherwise fall back to midpoint rooting
    if outgroup and outgroup in labels:
        tree.root_with_outgroup(outgroup)
    else:
        tree.root_at_midpoint()

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
    _draw_nj_clade(ax, tree.root, root_branch, leaf_y, max_depth, outgroup)

    ax.set_xlim(-max_depth * 0.02, max_depth * 1.30)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(f"Distance ({metric})", color=STYLE["fg"], fontsize=10)
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


def get_key_ingredients(pivot: pd.DataFrame, meta: pd.DataFrame,
                        max_ingredients: int = 8) -> list[str]:
    """Select the most informative ingredients for the heatmap, in this order:"""
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


def render_all(foods_csv: Path, output_dir: Path, analysis_name: str,
               method: str = "upgma", metric: str = "braycurtis",
               outgroup: str | None = None,
               min_pct: float = 2.0) -> None:
    """Always produces:"""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = analysis_name.lower().replace(" ", "-")

    print(f"\nLoading {foods_csv}...")
    pivot, meta = load_foods(foods_csv, min_pct=min_pct)
    print(f"  {len(pivot)} products × {len(pivot.columns)} ingredients")

    if outgroup:
        if outgroup in pivot.index:
            print(f"  Outgroup (reference): {outgroup}")
        else:
            print(f"  WARNING: outgroup '{outgroup}' not found in foods.csv — ignored")
            outgroup = None

    print("\nRendering trees...")
    leaf_order = get_upgma_leaf_order(pivot, metric)

    render_upgma(pivot, output_dir / f"{slug}-tree-upgma-{metric}.png",
                 analysis_name, metric, outgroup=outgroup)
    render_nj(pivot, output_dir / f"{slug}-tree-nj-{metric}.png",
              analysis_name, metric, outgroup=outgroup)

    print("\nRendering distance matrix...")
    render_distance_matrix(pivot,
                           output_dir / f"{slug}-distance-{metric}.png",
                           analysis_name, metric, leaf_order=leaf_order)

    key_ings = get_key_ingredients(pivot, meta)

    print(f"\nRendering ingredient overview...")
    render_heatmap_combined(
        pivot, key_ings,
        output_dir / f"{slug}-heatmap.png",
        analysis_name, leaf_order,
    )

    print(f"\nSaving ingredient table (CSV)...")
    render_table(
        pivot, pivot.columns.tolist(),
        output_dir / f"{slug}-table.png",
        analysis_name, leaf_order,
    )

    print(f"\n✓ All assets saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="FoodTree Stage 3: Render trees and heatmaps")
    parser.add_argument("--input", required=True, help="Path to foods.csv")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--name", required=True, help="Analysis name (e.g. 'Crisps')")
    parser.add_argument("--method", choices=["upgma", "nj"], default="upgma",
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
    args = parser.parse_args()

    render_all(
        foods_csv=Path(args.input),
        output_dir=Path(args.output),
        analysis_name=args.name,
        method=args.method,
        metric=args.distance,
        outgroup=args.outgroup,
        min_pct=args.min_pct,
    )


if __name__ == "__main__":
    main()
