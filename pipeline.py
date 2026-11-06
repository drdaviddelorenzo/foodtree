"""FoodTree pipeline: fetch -> normalise -> render.

    python pipeline.py run --input products.csv --name "Crisps" --dir analyses/crisps/
    python pipeline.py fetch|normalise|render --dir analyses/crisps/    # one stage only

Optional: export USDA_API_KEY to also use USDA FoodData Central.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import csv as _csv


def prompt_min_pct(foods_csv: Path, default: float = 2.0) -> float:
    """Read foods.csv, print a table of ingredients and their max % across all"""
    ingredient_maxes: dict[str, float] = {}
    try:
        with open(foods_csv, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                row_lower = {k.lower(): v for k, v in row.items()}
                ing = row_lower.get("ingredient", "").strip()
                try:
                    pct = float(row_lower.get("percentage", 0) or 0)
                except ValueError:
                    pct = 0.0
                if ing:
                    ingredient_maxes[ing] = max(ingredient_maxes.get(ing, 0.0), pct)
    except Exception:
        pass

    if ingredient_maxes:
        sorted_ings = sorted(ingredient_maxes.items(), key=lambda x: -x[1])
        print("\n  Ingredients found (max % across all products):\n")
        for ing, mx in sorted_ings:
            bar = "█" * min(int(mx / 2), 30)
            print(f"    {mx:6.2f}%  {bar}  {ing}")
        print()

    while True:
        try:
            raw = input(f"  Min % threshold to include an ingredient? [{default}]: ").strip()
            if not raw:
                chosen = default
            else:
                chosen = float(raw)
            if chosen < 0:
                print("  Must be 0 or above.")
                continue
            break
        except ValueError:
            print("  Please enter a number (e.g. 2, 0.1, 0).")
        except (EOFError, KeyboardInterrupt):
            print()
            chosen = default
            break

    kept = sum(1 for mx in ingredient_maxes.values() if mx >= chosen)
    dropped = len(ingredient_maxes) - kept
    print(f"\n  → {kept} ingredients kept, {dropped} dropped at {chosen}% threshold.\n")
    return chosen


def prompt_distance(default: str = "braycurtis") -> str:
    """Ask the user which distance metric to use, with a short description of each."""
    options = {
        "1": ("braycurtis", "Bray-Curtis  — share of the composition two products do not have in common (default)"),
        "2": ("aitchison",  "Aitchison    — differences in ratios (better for trace ingredients: caffeine, vitamins)"),
        "3": ("jaccard",    "Jaccard      — presence/absence only, ignores quantities entirely"),
        "4": ("euclidean",  "Euclidean    — raw percentage differences, dominated by the largest ingredient"),
        "5": ("unifrac",    "UniFrac      — like Bray-Curtis, but related ingredients (two oils) count as close"),
    }
    default_num = next(k for k, (m, _) in options.items() if m == default)

    print("\n  Distance metric:")
    for num, (_, desc) in options.items():
        marker = "  ◀ default" if num == default_num else ""
        print(f"    {num}. {desc}{marker}")
    print()

    while True:
        try:
            raw = input(f"  Choose 1–5 [{default_num}]: ").strip()
            if not raw:
                chosen = default
                break
            if raw in options:
                chosen = options[raw][0]
                break
            if raw in {m for m, _ in options.values()}:
                chosen = raw
                break
            print("  Please enter a number 1–5.")
        except (EOFError, KeyboardInterrupt):
            print()
            chosen = default
            break

    label = next(desc for _, (m, desc) in options.items() if m == chosen)
    print(f"\n  → {label}\n")
    return chosen


def run_fetch(products_csv: Path, output_dir: Path, usda_key: str,
              name_search: bool = False) -> Path:
    from fetch import fetch_all
    output_json = output_dir / "raw_products.json"
    fetch_all(products_csv=products_csv, output_json=output_json, usda_key=usda_key,
              name_search=name_search)
    return output_json


def run_normalise(output_dir: Path, manual_csv: Path | None = None,
                  allow_missing: bool = False, moisture: str | None = None,
                  allow_inconsistent: bool = False) -> Path:
    from normalise import normalise_all, parse_moisture
    input_json = output_dir / "raw_products.json"
    output_csv = output_dir / "foods.csv"
    if not input_json.exists():
        print(f"Error: {input_json} not found. Run fetch stage first.", file=sys.stderr)
        sys.exit(1)
    normalise_all(input_json=input_json, output_csv=output_csv, manual_csv=manual_csv,
                  allow_missing=allow_missing, moisture=parse_moisture(moisture),
                  allow_inconsistent=allow_inconsistent)
    return output_csv


def read_omitted(output_dir: Path) -> list[str]:
    from normalise import omitted_path
    path = omitted_path(output_dir / "foods.csv")
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_render(output_dir: Path, analysis_name: str, method: str,
               metric: str,
               outgroup: str | None = None,
               min_pct: float = 2.0,
               export_data: bool = True,
               level: str = "ingredient",
               support: int = 100) -> None:
    from render import render_all
    foods_csv = output_dir / "foods.csv"
    if not foods_csv.exists():
        print(f"Error: {foods_csv} not found. Run normalise stage first.", file=sys.stderr)
        sys.exit(1)
    render_all(
        foods_csv=foods_csv,
        output_dir=output_dir,
        analysis_name=analysis_name,
        method=method,
        metric=metric,
        outgroup=outgroup,
        min_pct=min_pct,
        export_data=export_data,
        level=level,
        support=support,
    )


def add_common_args(p):
    p.add_argument("--dir", required=True,
                   help="Analysis directory (all files read/written here)")
    p.add_argument("--usda-key",
                   default=os.environ.get("USDA_API_KEY", "DEMO_KEY"),
                   help="USDA API key")


def add_render_args(p):
    p.add_argument("--method", choices=["upgma", "nj", "both"], default="upgma",
                   help="Tree algorithm (default: upgma)")
    p.add_argument("--no-export", action="store_true",
                   help="Do not write the tree data files (Newick, JSON, distance CSV)")
    p.add_argument("--distance",
                   choices=["braycurtis", "aitchison", "euclidean", "jaccard", "unifrac"],
                   default=None,
                   help="Distance metric (default: braycurtis, or prompted interactively)")
    p.add_argument("--outgroup", default=None,
                   help="Product name to use as outgroup/reference (e.g. 'Cod Fillet')")
    p.add_argument("--min-pct", type=float, default=None,
                   help="Drop ingredients whose max across all products is below this %% "
                        "(default: 2.0, or prompted interactively). Use 0 to keep everything.")
    p.add_argument("--support", type=int, default=100,
                   help="Replicate trees for branch support (default 100; 0 to skip)")
    p.add_argument("--level", choices=["ingredient", "source", "group"], default="ingredient",
                   help="Compare ingredients as listed, or collapsed to their source or group "
                        "in data/ingredient_tree.csv (default: ingredient)")


def add_fetch_args(p):
    p.add_argument("--name-search", action="store_true",
                   help="Search Open Food Facts by name when a barcode is missing or not found")


def add_manual_arg(p):
    p.add_argument("--manual", default=None,
                   help="CSV of products entered by hand (FoodItem, Ingredient, Percentage"
                        "[, Category, Source])")
    p.add_argument("--allow-missing", action="store_true",
                   help="Leave out products without usable data instead of stopping "
                        "(they are listed in foods-omitted.txt and run-config.txt)")
    p.add_argument("--moisture", default=None,
                   help="Water of the finished products, g per 100 g, as LO-HI (e.g. 1-4 for "
                        "crisps); by default it is left to the nutrition table")
    p.add_argument("--allow-inconsistent", action="store_true",
                   help="Keep products whose label no composition satisfies, using the "
                        "least-violating composition")


def build_parser():
    parser = argparse.ArgumentParser(
        description="FoodTree Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Full pipeline: fetch + normalise + render")
    add_common_args(run_p)
    add_render_args(run_p)
    run_p.add_argument("--input", required=True, help="products.csv path")
    run_p.add_argument("--name", required=True, help="Analysis name (e.g. 'Crisps')")
    add_fetch_args(run_p)
    add_manual_arg(run_p)

    fetch_p = sub.add_parser("fetch", help="Stage 1 only: fetch product data")
    add_common_args(fetch_p)
    fetch_p.add_argument("--input", required=True, help="products.csv path")
    add_fetch_args(fetch_p)

    norm_p = sub.add_parser("normalise", help="Stage 2 only: normalise ingredients")
    add_common_args(norm_p)
    add_manual_arg(norm_p)

    rend_p = sub.add_parser("render", help="Stage 3 only: generate trees and heatmaps")
    add_common_args(rend_p)
    add_render_args(rend_p)
    rend_p.add_argument("--name", required=True, help="Analysis name")

    return parser


def save_run_config(output_dir: Path, args, min_pct: float) -> None:
    """Write a run-config.txt recording all options used in this run."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cmd_parts = ["python pipeline.py", args.command]
    if hasattr(args, "input") and args.input:
        cmd_parts.append(f"--input {args.input}")
    if hasattr(args, "name") and args.name:
        cmd_parts.append(f"--name \"{args.name}\"")
    cmd_parts.append(f"--dir {args.dir}")
    if hasattr(args, "method"):
        cmd_parts.append(f"--method {args.method}")
    if hasattr(args, "distance"):
        cmd_parts.append(f"--distance {args.distance}")
    cmd_parts.append(f"--min-pct {min_pct}")
    if hasattr(args, "outgroup") and args.outgroup:
        cmd_parts.append(f"--outgroup \"{args.outgroup}\"")
    if getattr(args, "name_search", False):
        cmd_parts.append("--name-search")
    if getattr(args, "manual", None):
        cmd_parts.append(f"--manual {args.manual}")
    if getattr(args, "allow_missing", False):
        cmd_parts.append("--allow-missing")
    if getattr(args, "moisture", None):
        cmd_parts.append(f"--moisture {args.moisture}")
    if getattr(args, "allow_inconsistent", False):
        cmd_parts.append("--allow-inconsistent")
    if getattr(args, "no_export", False):
        cmd_parts.append("--no-export")
    if getattr(args, "level", "ingredient") != "ingredient":
        cmd_parts.append(f"--level {args.level}")
    if getattr(args, "support", 100) != 100:
        cmd_parts.append(f"--support {args.support}")

    cmd =" \\\n    ".join(cmd_parts)

    lines = [
        "=" * 60,
        "  FoodTree — Run Configuration",
        "=" * 60,
        f"  Run date   : {ts}",
        f"  Command    : {args.command}",
    ]
    if hasattr(args, "name") and args.name:
        lines.append(f"  Name       : {args.name}")
    if hasattr(args, "input") and args.input:
        lines.append(f"  Input      : {args.input}")
    lines += [
        f"  Output dir : {args.dir}",
    ]
    if hasattr(args, "method"):
        lines.append(f"  Tree method: {args.method}")
    if hasattr(args, "distance"):
        lines.append(f"  Distance   : {args.distance}")
    lines.append(f"  Min pct    : {min_pct}%")
    if hasattr(args, "outgroup") and args.outgroup:
        lines.append(f"  Outgroup   : {args.outgroup}")
    omitted = read_omitted(output_dir)
    lines.append(f"  Omitted    : {', '.join(omitted) if omitted else 'none'}")
    lines += [
        "",
        "Equivalent command:",
        "",
        "  " + cmd,
        "",
        "=" * 60,
    ]

    config_path = output_dir / "run-config.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"  ✓ Run config saved → {config_path.name}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        print("=" * 60)
        print(f"  FoodTree — {args.name}")
        print("=" * 60)
        print("\n▶ Stage 1: Fetch")
        run_fetch(Path(args.input), output_dir, args.usda_key, args.name_search)
        print("\n▶ Stage 2: Normalise")
        run_normalise(output_dir, Path(args.manual) if args.manual else None,
                      args.allow_missing, args.moisture, args.allow_inconsistent)
        min_pct = args.min_pct
        if min_pct is None:
            min_pct = prompt_min_pct(output_dir / "foods.csv")
        distance = args.distance
        if distance is None:
            distance = prompt_distance()
        print("\n▶ Stage 3: Render")
        run_render(output_dir, args.name, args.method, distance,
                   outgroup=args.outgroup, min_pct=min_pct,
                   export_data=not args.no_export, level=args.level,
                   support=args.support)
        args.distance = distance
        save_run_config(output_dir, args, min_pct)
        print("\n" + "=" * 60)
        print(f"  ✓ Done. Output in: {output_dir}/")
        print("=" * 60)

    elif args.command == "fetch":
        run_fetch(Path(args.input), output_dir, args.usda_key, args.name_search)

    elif args.command == "normalise":
        run_normalise(output_dir, Path(args.manual) if args.manual else None,
                      args.allow_missing, args.moisture, args.allow_inconsistent)

    elif args.command == "render":
        min_pct = args.min_pct
        if min_pct is None:
            min_pct = prompt_min_pct(output_dir / "foods.csv")
        distance = args.distance
        if distance is None:
            distance = prompt_distance()
        run_render(output_dir, args.name, args.method, distance,
                   outgroup=args.outgroup, min_pct=min_pct,
                   export_data=not args.no_export, level=args.level,
                   support=args.support)
        args.distance = distance
        save_run_config(output_dir, args, min_pct)


if __name__ == "__main__":
    main()
