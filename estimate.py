"""Label-constrained estimation of ingredient quantities.

The unknowns are, for each listed ingredient, the grams used per 100 g of finished product
(u, the basis of the ingredient order and of declared percentages: Regulation (EU)
No 1169/2011, Article 18 and Annex VIII) and the grams of its dry matter left in 100 g of
finished product (c). Residual water W completes the finished product: sum(c) + W = 100.

Constraints:
  - declared percentages (u, within 0.5 points of the label value);
  - descending order of u, except for ingredients under 2% of the finished product, which may
    be listed in any order after the others (Article 18 and Annex VII, Part A, point 6);
  - u(1 - water_hi) <= c <= u(1 - water_lo) for each ingredient;
  - fat, carbohydrate, protein and salt of the nutrition table, within the tolerances of the
    European Commission guidance on nutrient values declared on a label (2012), using the
    composition ranges of data/ingredient_composition.csv;
  - an optional range for the moisture of the finished product.

Bounds are outer bounds: they contain every composition compatible with the label, but
can be wider than that set, because the composition ranges enter as a linear relaxation and
the under-2% test uses dry matter only (one linear programme per position where the tail
can start).

Samples follow a two-level scheme, not a uniform distribution over all compatible
(composition, recipe) pairs: (1) each ingredient's composition is drawn uniformly within its
class ranges, with parts adding up to at most 100%; (2) for that composition, a fixed number
of recipes is drawn uniformly, by hit-and-run, among those the label allows, every allowed
ingredient ordering included (candidates from a polytope containing all orderings, rejected
if they break the exact order rule). A composition is kept only if 5 valid recipes turn up
among its first 200 candidates. This cutoff keeps run times bounded, but it under-represents
compositions whose valid recipes fill only a small part of the candidate polytope, so
compositions are not drawn exactly uniformly among those compatible with the label.
Variables whose two bounds coincide (for example a single moisture value) are treated as
equalities; a composition whose recipes have no interior left after those equalities yields
no samples rather than one repeated vertex. The exact order rule counts, for the under-2%
exception, each ingredient's dry matter plus a share of the residual water proportional to
the water it brought (an assumption). Each sample is then weighted by how close its
nutrients are to the declared values (a normal likelihood whose tolerance band is two
standard deviations wide on each side).
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

COMPOSITION_CSV = Path(__file__).with_name("data") / "ingredient_composition.csv"
NUTRIENTS = ("fat", "carb", "protein", "salt")
QUID_TOLERANCE = 0.5
TAIL_LIMIT = 2.0

SODIUM_ADDITIVES = {"e211", "e250", "e251", "e262", "e301", "e331", "e339", "e450", "e451",
                    "e452", "e481", "e500", "e621", "e627", "e631", "e635"}
POTATO_DRY_MARKERS =("dehydrat", "deshydrat", "deshidrat", "dried", "flake", "copos", "flocon")


def load_composition(path: Path = COMPOSITION_CSV) -> dict[str, dict[str, tuple[float, float]]]:
    """Composition ranges per class, as fractions of the ingredient as used."""
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ranges = {key: (float(row[f"{key}_lo"]) / 100, float(row[f"{key}_hi"]) / 100)
                      for key in ("water",) + NUTRIENTS}
            if any(not 0 <= lo <= hi <= 1 for lo, hi in ranges.values()):
                raise ValueError(f"{path.name}: ranges of '{row['class']}' must lie within 0-100")
            if sum(lo for lo, _ in ranges.values()) > 1:
                raise ValueError(f"{path.name}: the lower limits of '{row['class']}' add up to "
                                 "more than 100 g")
            table[row["class"]] = ranges
    return table


def ingredient_class(off_id: str, text: str = "", processing: str = "",
                     sub_ids: tuple[str, ...] = ()) -> str:
    """Composition class of one top-level ingredient."""
    i, t = off_id.lower(), f"{text} {processing}".lower()
    if i in ("en:potato", "en:potatoes", "en:select potatoes") or (not i and "potato" in t):
        if any(m in t for m in POTATO_DRY_MARKERS):
            return "potato_dehydrated"
        return "potato_raw"
    fixed = {
        "en:potato-flakes": "potato_dehydrated", "en:dehydrated-potato": "potato_dehydrated",
        "en:potato-starch": "potato_starch", "en:potato-flour": "potato_flour",
        "en:wheat-flour": "wheat_flour", "en:corn-flour": "corn_flour",
        "en:maize-flour": "corn_flour", "en:cornmeal": "cornmeal", "en:corn": "corn_grain",
        "en:rice-flour": "rice_flour", "en:maltodextrin": "maltodextrin",
        "en:salt": "salt", "en:sea-salt": "salt", "en:e508": "potassium_chloride",
        "en:vegetable-oil": "oil", "en:vegetable-fat": "oil",
    }
    if i in fixed:
        return fixed[i]
    if i.endswith("-oil"):
        return "oil"
    e_number = re.match(r"^en:(e\d{3})", i)
    if e_number:
        return "sodium_additive" if e_number.group(1) in SODIUM_ADDITIVES else "additive"
    subs = " ".join(sub_ids).lower()
    if any(w in i + " " + t for w in ("condiment", "seasoning", "sazonador")) and \
            any(w in subs for w in ("cheese", "whey", "milk")):
        return "cheese_seasoning"
    return "unknown"


def split_top_level(text: str) -> tuple[list[str], int | None]:
    """Split an ingredient list at depth-0 commas and semicolons.

    Returns the items and the index of the item whose parenthesis is never closed (None if
    the text is balanced).
    """
    items, cur, depth, open_at = [], [], 0, None
    for ch in text:
        if ch in "([":
            if depth == 0:
                open_at = len(items)
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        if ch in ",;" and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    last = "".join(cur).strip().rstrip(".").strip()
    if last:
        items.append(last)
    return [s for s in items if s], (open_at if depth > 0 else None)


@dataclass
class Item:
    name: str
    off_id: str
    cls: str
    declared: float | None
    subs: tuple[str, ...] = ()


def top_level_items(off: dict, name_of) -> tuple[list[Item], list[str]]:
    """Top-level ingredients from an Open Food Facts record, and notes on repairs made.

    Open Food Facts flattens a compound ingredient whose parenthesis is never closed on the
    label: its components then appear as top-level ingredients. When the raw text has such an
    unclosed parenthesis, the entries after that ingredient are merged into it.
    """
    ings = off.get("ingredients") or []
    notes = []
    parts, open_at = split_top_level(off.get("ingredients_raw_text") or "")
    keep = len(ings)
    if open_at is not None and open_at + 1 < len(ings):
        keep = open_at + 1
        merged = [i.get("id", "") for i in ings[keep:]]
        notes.append(f"unclosed parenthesis in the label text: {len(merged)} entries treated "
                     f"as part of '{ings[open_at].get('text', '')}'")
    items = []
    for k, ing in enumerate(ings[:keep]):
        subs = tuple(s.get("id", "") for s in ing.get("ingredients") or [])
        if k == keep - 1 and keep < len(ings):
            subs += tuple(i.get("id", "") for i in ings[keep:])
        pct = ing.get("percent")
        items.append(Item(
            name=name_of(ing),
            off_id=ing.get("id", ""),
            cls=ingredient_class(ing.get("id", ""), ing.get("text", ""),
                                 ing.get("processing", ""), subs),
            declared=float(pct) if pct is not None else None,
            subs=subs,
        ))
    return items, notes


def tolerance_band(nutrient: str, value: float) -> tuple[float, float]:
    """Range compatible with a declared value (European Commission guidance, 2012)."""
    if nutrient == "salt":
        tol = 0.375 if value < 1.25 else 0.2 * value
    elif nutrient == "fat":
        tol = 1.5 if value < 10 else (0.2 * value if value <= 40 else 8.0)
    else:
        tol = 2.0 if value < 10 else (0.2 * value if value <= 40 else 8.0)
    return max(value - tol, 0.0), value + tol


NUTRIENT_KEYS = {"fat": "fat_g", "carb": "carbohydrates_g", "protein": "protein_g", "salt": "salt_g"}


def nutrient_value(nutrients: dict, nut: str) -> float | None:
    v = nutrients.get(NUTRIENT_KEYS[nut])
    if v is None and nut == "salt" and nutrients.get("sodium_g") is not None:
        v = 2.5 * float(nutrients["sodium_g"])
    return float(v) if v is not None else None


def nutrient_targets(nutrients: dict) -> dict[str, tuple[float, float]]:
    targets = {}
    for nut in NUTRIENT_KEYS:
        v = nutrient_value(nutrients, nut)
        if v is not None:
            targets[nut] = tolerance_band(nut, v)
    return targets


@dataclass
class Estimate:
    names: list[str]
    feasible: bool
    used_low: np.ndarray = field(default_factory=lambda: np.empty(0))
    used_high: np.ndarray = field(default_factory=lambda: np.empty(0))
    dry_low: np.ndarray = field(default_factory=lambda: np.empty(0))
    dry_high: np.ndarray = field(default_factory=lambda: np.empty(0))
    water_low: float = float("nan")
    water_high: float = float("nan")
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    used_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    weights: np.ndarray = field(default_factory=lambda: np.empty(0))
    acceptance: float = 0.0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sampling_failed: bool = False


class _Model:
    """Linear constraints on x = (u_1..u_n, c_1..c_n, W, m_1,1..m_n,T) for one tail start k.

    m[i, t] is the mass of nutrient t contributed by ingredient i. It lies between c_i times the
    lowest and highest content of t per gram of the ingredient's dry matter, the nutrients of
    one ingredient cannot exceed its dry matter, and their sums over ingredients must match
    the nutrition table within tolerance.
    """

    def __init__(self, items, comp, targets, moisture):
        self.n = n = len(items)
        self.items = items
        self.comp = [comp[i.cls] for i in items]
        self.targets = targets
        self.nuts = list(targets)
        self.moisture = moisture
        self.nm = n * len(self.nuts)
        self.nv = 2 * n + 1 + self.nm

    def m(self, i, t):
        return 2 * self.n + 1 + i * len(self.nuts) + t

    def constraints(self, k: int, quid_as_rows: bool = False):
        n, nv = self.n, self.nv
        A, b, labels = [], [], []

        def row():
            return np.zeros(nv)

        for i, (it, cp) in enumerate(zip(self.items, self.comp)):
            w_lo, w_hi = cp["water"]
            r = row(); r[i] = 1 - w_hi; r[n + i] = -1
            A.append(r); b.append(0.0); labels.append(f"dry matter of {it.name}")
            r = row(); r[n + i] = 1; r[i] = -(1 - w_lo)
            A.append(r); b.append(0.0); labels.append(f"dry matter of {it.name}")
            r_sum = row(); r_sum[n + i] = -1
            for t, nut in enumerate(self.nuts):
                n_lo, n_hi = cp[nut]
                dm_lo = n_lo / (1 - w_lo) if w_lo < 1 else 0.0
                dm_hi = min(n_hi / (1 - w_hi), 1.0) if w_hi < 1 else 1.0
                r = row(); r[self.m(i, t)] = 1; r[n + i] = -dm_hi
                A.append(r); b.append(0.0); labels.append(f"{nut} content of {it.name}")
                r = row(); r[self.m(i, t)] = -1; r[n + i] = dm_lo
                A.append(r); b.append(0.0); labels.append(f"{nut} content of {it.name}")
                r_sum[self.m(i, t)] = 1
            if self.nuts:
                A.append(r_sum); b.append(0.0)
                labels.append(f"nutrients of {it.name} within its dry matter")
        for t, nut in enumerate(self.nuts):
            lo, hi = self.targets[nut]
            r = row()
            for i in range(n):
                r[self.m(i, t)] = 1
            A.append(r); b.append(hi); labels.append(f"{nut} (label {self._label(nut)})")
            A.append(-r); b.append(-lo); labels.append(f"{nut} (label {self._label(nut)})")
        for i in range(min(k, n) - 1):
            r = row(); r[i + 1] = 1; r[i] = -1
            A.append(r); b.append(0.0)
            labels.append(f"order: {self.items[i].name} before {self.items[i + 1].name}")
        for i in range(k, n):
            r = row(); r[n + i] = 1
            A.append(r); b.append(TAIL_LIMIT); labels.append(f"{self.items[i].name} under 2%")
        bounds = [(0, None)] * (2 * n) + [self.moisture] + [(0, None)] * self.nm
        for i, it in enumerate(self.items):
            if it.declared is None:
                continue
            lo, hi = max(it.declared - QUID_TOLERANCE, 0), it.declared + QUID_TOLERANCE
            if quid_as_rows:
                r = row(); r[i] = 1
                A.append(r); b.append(hi); labels.append(f"declared {it.name} {it.declared:g}%")
                r = row(); r[i] = -1
                A.append(r); b.append(-lo); labels.append(f"declared {it.name} {it.declared:g}%")
            else:
                bounds[i] = (lo, hi)
        a_eq = np.zeros((1, nv)); a_eq[0, n:2 * n + 1] = 1
        return np.array(A).reshape(-1, nv), np.array(b), labels, a_eq, np.array([100.0]), bounds

    def _label(self, nut):
        lo, hi = self.targets[nut]
        return f"{lo:.2f}-{hi:.2f} g"

    def solve(self, k, c):
        """Optimum of c.x, -inf if unbounded below, None if infeasible."""
        A, b, _, a_eq, b_eq, bounds = self.constraints(k)
        res = linprog(c, A_ub=A, b_ub=b, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.status == 0:
            return res.fun
        if res.status == 3:
            return -np.inf
        return None


def _bounds(model: _Model):
    """Outer bounds on u, c and W over all tail starts k, and the feasible k."""
    n = model.n
    nb = 2 * n + 1
    lo = np.full(nb, np.inf)
    hi = np.full(nb, -np.inf)
    feasible_k = []
    for k in range(n + 1):
        if model.solve(k, np.zeros(model.nv)) is None:
            continue
        feasible_k.append(k)
        for j in range(nb):
            e = np.zeros(model.nv); e[j] = 1
            lo[j] = min(lo[j], model.solve(k, e))
            hi[j] = max(hi[j], -model.solve(k, -e))
    return lo, hi, feasible_k


PHYSICAL_PREFIXES = ("dry matter of", "nutrients of")


def _diagnose(model: _Model) -> tuple[list[str], np.ndarray | None]:
    """Label constraints that must be relaxed for the label to become consistent, and the
    composition (u, c, W) that relaxes them least (sum of violations). Composition constraints
    (dry matter and nutrient content of each ingredient) are never relaxed."""
    best = None
    for k in range(model.n + 1):
        A, b, labels, a_eq, b_eq, bounds = model.constraints(k, quid_as_rows=True)
        m = len(b)
        hard = [lab.startswith(PHYSICAL_PREFIXES) or " content of " in lab for lab in labels]
        A_s = np.hstack([A, -np.eye(m)])
        c = np.concatenate([np.zeros(model.nv), np.ones(m)])
        a_eq_s = np.hstack([a_eq, np.zeros((1, m))])
        slack_bounds = [(0, 0) if h else (0, None) for h in hard]
        res = linprog(c, A_ub=A_s, b_ub=b, A_eq=a_eq_s, b_eq=b_eq,
                      bounds=list(bounds) + slack_bounds, method="highs")
        if res.status == 0 and (best is None or res.fun < best[0]):
            best = (res.fun, res.x[model.nv:], labels, res.x[:2 * model.n + 1])
    if best is None:
        return ["declared percentages are incompatible with each other or with the moisture "
                "range"], None
    _, slack, labels, x = best
    return sorted({f"{lab} (off by {s:.2f})" for s, lab in zip(slack, labels) if s > 1e-6}), x


def _draw_composition(cp, rng, tries: int = 200):
    """One composition (fractions of the ingredient as used) drawn uniformly from the class
    ranges subject to the parts adding up to at most 1, or None if no draw fits."""
    if all(lo == 0 and hi == 1 for lo, hi in cp.values()):
        parts = rng.dirichlet(np.ones(len(cp) + 1))
        return dict(zip(cp.keys(), parts[:-1]))
    for _ in range(tries):
        d = {k: rng.uniform(lo, hi) for k, (lo, hi) in cp.items()}
        if sum(d.values()) <= 1.0:
            return d
    return None


def _fixed_constraints(items, draw, keep, targets, moisture):
    """Constraints on y = (c_1..c_n, W) for one drawn composition, with u = c / keep.

    Every composition compatible with the label satisfies them: the order rule is relaxed to
    u[i+1] - u[i] <= 2 / keep[i+1], which holds both for a pair in descending order and for
    a tail ingredient (dry matter under 2 g). Samples are then checked with _order_ok.
    """
    n = len(items)
    nv = n + 1
    A, b = [], []
    for nut, (lo, hi) in targets.items():
        r = np.zeros(nv)
        r[:n] = [d[nut] / kp for d, kp in zip(draw, keep)]
        A.append(r); b.append(hi)
        A.append(-r); b.append(-lo)
    for i in range(n - 1):
        r = np.zeros(nv)
        r[i + 1], r[i] = 1 / keep[i + 1], -1 / keep[i]
        A.append(r); b.append(TAIL_LIMIT / keep[i + 1])
    bounds = [(0.0, None)] * n + [tuple(moisture)]
    for i, it in enumerate(items):
        if it.declared is not None:
            bounds[i] = (max(it.declared - QUID_TOLERANCE, 0) * keep[i],
                         (it.declared + QUID_TOLERANCE) * keep[i])
    a_eq = np.ones((1, nv))
    return np.array(A).reshape(-1, nv), np.array(b), a_eq, np.array([100.0]), bounds


def finished_mass(u: np.ndarray, c: np.ndarray, water: float) -> np.ndarray:
    """Grams of each ingredient in 100 g of finished product: its dry matter plus a share of
    the residual water proportional to the water it brought (an assumption: the label does
    not say which ingredient keeps the water)."""
    brought = np.clip(u - c, 0, None)
    total = brought.sum()
    return c + (water * brought / total if total > 0 else 0.0)


def _order_ok(u: np.ndarray, mass: np.ndarray) -> bool:
    """Exact order rule: descending order of u, except that ingredients under 2% of the
    finished product may follow in any order after the others."""
    k = 1
    while k < len(u) and u[k] <= u[k - 1] + 1e-9:
        k += 1
    return bool(np.all(mass[k:] <= TAIL_LIMIT + 1e-9))


def _hit_and_run(A, b, a_eq, b_eq, bounds, rng, n_samples, steps, accept=None,
                 max_candidates=None):
    """Samples from {x: A x <= b, a_eq x = b_eq, bounds} by hit-and-run: a candidate every
    `steps` moves, kept if accept(x) holds, until n_samples are kept or max_candidates are
    tried. Kept samples are uniform over the part of the polytope where accept holds.
    Variables whose lower and upper bounds coincide are treated as equalities, so sampling
    happens within the remaining subspace. Returns None if the polytope is empty; otherwise
    the kept samples (none if the polytope has no interior in that subspace, rather than a
    single vertex repeated) and the number of candidates tried."""
    nv = A.shape[1]
    G, h = [A], [b]
    eq_rows, eq_vals = [a_eq], [b_eq]
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None and hi is not None and hi - lo < 1e-9:
            r = np.zeros(nv); r[j] = 1
            eq_rows.append(r[None]); eq_vals.append([lo])
            continue
        if lo is not None:
            r = np.zeros(nv); r[j] = -1; G.append(r[None]); h.append([-lo])
        if hi is not None:
            r = np.zeros(nv); r[j] = 1; G.append(r[None]); h.append([hi])
    G, h = np.vstack(G), np.concatenate([np.atleast_1d(x) for x in h])
    a_eq = np.vstack(eq_rows)
    b_eq = np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)) for v in eq_vals])
    _, _, vt = np.linalg.svd(a_eq)
    N = vt[a_eq.shape[0]:].T
    norms = np.linalg.norm(G @ N, axis=1)
    c = np.zeros(nv + 1); c[-1] = -1
    res = linprog(c, A_ub=np.hstack([G, norms[:, None]]), b_ub=h,
                  A_eq=np.hstack([a_eq, np.zeros((a_eq.shape[0], 1))]), b_eq=b_eq,
                  bounds=[(None, None)] * nv + [(0, None)], method="highs")
    if res.status != 0:
        return None
    x = res.x[:nv]
    accept = accept or (lambda _: True)
    max_candidates = max_candidates or n_samples
    if res.x[-1] <= 1e-9:
        return np.empty((0, nv)), 1
    out, candidates = [], 0
    while len(out) < n_samples and candidates < max_candidates:
        for _ in range(steps):
            d = N @ rng.standard_normal(N.shape[1])
            d /= np.linalg.norm(d)
            gd, slack = G @ d, h - G @ x
            with np.errstate(divide="ignore", invalid="ignore"):
                t = slack / gd
            t_max = np.min(t[gd > 1e-12], initial=np.inf)
            t_min = np.max(t[gd < -1e-12], initial=-np.inf)
            if np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min:
                x = x + rng.uniform(t_min, t_max) * d
        candidates += 1
        if accept(x):
            out.append(x.copy())
    return np.array(out).reshape(-1, nv), candidates


def estimate_product(items: list[Item], nutrients: dict, comp: dict,
                     moisture: tuple[float, float] = (0.0, 100.0),
                     target_samples: int = 2000, max_draws: int = 10000, per_draw: int = 5,
                     steps: int = 20, max_tries: int = 40, min_effective: int = 200, seed: int = 1,
                     allow_inconsistent: bool = False) -> Estimate:
    """Bounds and weighted samples of the composition of one product, given its label.

    If no composition satisfies the label, the result has feasible=False and lists the
    constraints to relax; with allow_inconsistent its only sample is the composition that
    relaxes them least.
    """
    targets = nutrient_targets(nutrients)
    model = _Model(items, comp, targets, moisture)
    n = model.n
    names = [i.name for i in items]
    lo, hi, feasible_k = _bounds(model)
    if not feasible_k:
        problems, x = _diagnose(model)
        est = Estimate(names, False, problems=problems)
        if allow_inconsistent and x is not None:
            est.samples = x[n:][None]
            est.used_samples = x[:n][None]
            est.weights = np.ones(1)
        return est
    est = Estimate(names, True, used_low=lo[:n], used_high=hi[:n], dry_low=lo[n:2 * n],
                   dry_high=hi[n:2 * n], water_low=lo[-1], water_high=hi[-1])

    rng = np.random.default_rng(seed)
    declared = {nut: nutrient_value(nutrients, nut) for nut in targets}
    sigma = {nut: max((hi - declared[nut]) / 2, 1e-6) for nut, (lo, hi) in targets.items()}
    samples, used, logw, tried, n_draws = [], [], [], 0, 0
    while len(samples) < target_samples and n_draws < max_draws:
        n_draws += 1
        draw = [_draw_composition(cp, rng) for cp in model.comp]
        if any(d is None for d in draw):
            continue
        keep = np.array([1 - min(d["water"], 0.999) for d in draw])
        A, b, a_eq, b_eq, bounds = _fixed_constraints(items, draw, keep, targets, moisture)

        def valid(x, keep=keep):
            u = x[:n] / keep
            return _order_ok(u, finished_mass(u, x[:n], x[-1]))

        result = _hit_and_run(A, b, a_eq, b_eq, bounds, rng, per_draw, steps, accept=valid,
                              max_candidates=per_draw * max_tries)
        if result is None:
            continue
        xs, candidates = result
        tried += candidates
        if len(xs) < per_draw:
            continue
        for x in xs:
            c = x[:n]
            u = c / keep
            samples.append(x)
            used.append(u)
            logw.append(-0.5 * sum(
                ((sum(ci * d[nut] / kp for ci, d, kp in zip(c, draw, keep)) - declared[nut])
                 / sigma[nut]) ** 2 for nut in targets))
    est.acceptance = len(samples) / tried if tried else 0.0
    est.samples = np.array(samples).reshape(-1, n + 1)
    est.used_samples = np.array(used).reshape(-1, n)
    if logw:
        w = np.exp(np.array(logw) - max(logw))
        est.weights = w / w.sum()
        ess = 1 / np.sum(est.weights ** 2)
        if ess < min_effective:
            est.warnings.append(f"only {ess:.0f} effective samples ({len(samples)} accepted of "
                                f"{tried} drawn); the point estimate is imprecise")
    else:
        est.sampling_failed = True
        est.problems.append("no sampled composition fits the label, although the outer bounds "
                            "allow one: the label may be satisfiable only at the edges of the "
                            "composition ranges, or the ingredient classes may be wrong")
    return est
