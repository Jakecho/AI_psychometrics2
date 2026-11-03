"""Simulated calibration data for every model, design and generating ability distribution.

Generating distributions are standardised at the population level (not per sample), so a group declared as
mean μ and SD σ has exactly those population moments whatever its shape. Anchor drift changes the parameters
that generate the responses while the bank keeps the original values — the situation drift studies examine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import skewnorm

from .data import Dataset
from .models import Item, ItemSpec, make_item, sample_responses

SHAPES = ("normal", "skewed", "bimodal", "discrete")
SKEW_A = 5.0


@dataclass
class SimSpec:
    model: str = "GPCM"
    design: str = "single"            # single | concurrent | fipc | mgfipc
    n_per_group: int = 1000
    n_items: int = 10                 # items per form (single/fipc: total free items)
    ncat: int = 4
    groups: int = 2
    n_common: int = 4                 # concurrent / mgfipc: free items shared by every form
    n_anchor: int = 4                 # fipc / mgfipc: fixed bank items
    shape: str = "normal"
    group_means: list = field(default_factory=lambda: [0.0, 0.5, -0.3, 0.8])
    group_sds: list = field(default_factory=lambda: [1.0, 1.1, 0.9, 1.2])
    group_shapes: list | None = None  # per-group shapes (group-specific mixtures)
    discrete_nodes: list | None = None
    discrete_weights: list | None = None
    anchor_drift: float = 0.0         # shift added to drifting anchors' generating locations (bank keeps old values)
    n_drift: int = 0                  # number of anchors that drift
    anchor_error: float = 0.0         # SD of random error in the bank values of anchors
    a_sd: float = 0.25                # log-normal SD of slopes (GPCM/GRM)
    seed: int = 2026

    def as_dict(self):
        return asdict(self)


def standardized_draw(shape, n, rng, nodes=None, weights=None):
    """Draws with population mean 0 and SD 1 for the named shape."""
    if shape == "normal":
        return rng.standard_normal(n)
    if shape == "skewed":
        d = SKEW_A / np.sqrt(1 + SKEW_A ** 2)
        m, s = d * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * d ** 2 / np.pi)
        return (skewnorm.rvs(SKEW_A, size=n, random_state=rng) - m) / s
    if shape == "bimodal":
        mu, sd = 1.0, 0.6
        x = rng.normal(np.where(rng.random(n) < 0.5, -mu, mu), sd)
        return x / np.sqrt(mu ** 2 + sd ** 2)
    if shape == "discrete":
        x = np.asarray(nodes if nodes is not None else [-2, -1, 0, 1, 2], float)
        w = np.asarray(weights if weights is not None else [0.1, 0.2, 0.4, 0.2, 0.1], float)
        w = w / w.sum()
        m = w @ x
        s = np.sqrt(w @ (x - m) ** 2)
        return (rng.choice(x, size=n, p=w) - m) / s
    raise ValueError(f"Unknown shape {shape!r}.")


def population_weights(shape, nodes, mean=0.0, sd=1.0, dnodes=None, dweights=None):
    """The generating distribution as masses on a grid (used to compare fitted histograms with the truth)."""
    x = np.asarray(nodes, float)
    z = (x - mean) / sd
    if shape == "normal":
        dens = np.exp(-z ** 2 / 2)
    elif shape == "skewed":
        d = SKEW_A / np.sqrt(1 + SKEW_A ** 2)
        m, s = d * np.sqrt(2 / np.pi), np.sqrt(1 - 2 * d ** 2 / np.pi)
        dens = skewnorm.pdf(z * s + m, SKEW_A)
    elif shape == "bimodal":
        mu, s0 = 1.0, 0.6
        k = np.sqrt(mu ** 2 + s0 ** 2)
        dens = np.exp(-(z * k - mu) ** 2 / (2 * s0 ** 2)) + np.exp(-(z * k + mu) ** 2 / (2 * s0 ** 2))
    elif shape == "discrete":
        dx = np.asarray(dnodes if dnodes is not None else [-2, -1, 0, 1, 2], float)
        dw = np.asarray(dweights if dweights is not None else [0.1, 0.2, 0.4, 0.2, 0.1], float)
        dw = dw / dw.sum()
        m = dw @ dx
        s = np.sqrt(dw @ (dx - m) ** 2)
        pts = (dx - m) / s * sd + mean
        dens = np.zeros_like(x)
        for p, w in zip(pts, dw):                      # put each support point's mass on the nearest node
            dens[np.argmin(np.abs(x - p))] += w
        return dens / dens.sum()
    else:
        raise ValueError(shape)
    return dens / dens.sum()


def generate_item(id, model, ncat, rng, a_sd=0.25, tau=None):
    K = ncat - 1
    loc = rng.uniform(-1.5, 1.5)
    if model == "RSM":
        return Item(id, "RSM", 1.0, (), b=float(loc), tau=tuple(tau))
    a = float(np.exp(rng.normal(0.1, a_sd))) if model in ("GPCM", "GRM") else 1.0
    if model == "GRM":
        offs = np.linspace(-1.2, 1.2, K) + rng.normal(0, 0.2, K) if K > 1 else np.zeros(1)
        b = np.sort(loc + offs - offs.mean())
        for k in range(1, K):                          # keep thresholds clearly ordered
            b[k] = max(b[k], b[k - 1] + 0.2)
        return make_item(id, "GRM", a, tuple(b))
    steps = np.linspace(-1.0, 1.0, K) * (K > 1) + rng.normal(0, 0.3, K)
    steps = steps - steps.mean()
    return make_item(id, model, a, tuple(loc + steps))


def common_tau(ncat, rng):
    K = ncat - 1
    t = np.linspace(-1.0, 1.0, K) + rng.normal(0, 0.15, K)
    return tuple(t - t.mean())


def simulate(spec: SimSpec, name=None):
    rng = np.random.default_rng(spec.seed)
    design = spec.design
    G = 1 if design in ("single", "fipc") else max(2, int(spec.groups))
    tau = common_tau(spec.ncat, rng) if spec.model == "RSM" else None
    anchored = design in ("fipc", "mgfipc")
    n_anchor = int(spec.n_anchor) if anchored else 0
    n_common = int(spec.n_common) if G > 1 else 0
    n_unique = int(spec.n_items) - (n_common if G > 1 else 0)
    if n_unique < 0:
        raise ValueError("Common items cannot exceed the items per form.")
    # Item layout: anchors (on every form), common free items, unique free items per form.
    true_items, forms = [], [[] for _ in range(G)]
    for k in range(n_anchor):
        it = generate_item(f"A{k + 1:02d}", spec.model, spec.ncat, rng, spec.a_sd, tau)
        true_items.append(it)
        for g in range(G):
            forms[g].append(len(true_items) - 1)
    for k in range(n_common):
        true_items.append(generate_item(f"C{k + 1:02d}", spec.model, spec.ncat, rng, spec.a_sd, tau))
        for g in range(G):
            forms[g].append(len(true_items) - 1)
    for g in range(G):
        for k in range(n_unique):
            prefix = "N" if G == 1 else f"F{g + 1}_"
            true_items.append(generate_item(f"{prefix}{k + 1:02d}", spec.model, spec.ncat, rng, spec.a_sd, tau))
            forms[g].append(len(true_items) - 1)
    # Bank values (what the calibration fixes) vs generating values (what produced the responses).
    # Drift affects later administrations: the new group in FIPC, groups 2.. in multi-group designs.
    bank, drifted = {}, list(true_items)
    for k in range(n_anchor):
        it = true_items[k]
        err = rng.normal(0, spec.anchor_error) if spec.anchor_error > 0 else 0.0
        bank[it.id] = _shift(it, err)
        if k < spec.n_drift:
            drifted[k] = _shift(it, spec.anchor_drift)
    gen_by_group = [drifted if (G == 1 or g >= 1) else list(true_items) for g in range(G)]
    J = len(true_items)
    N = int(spec.n_per_group) * G
    Y = np.full((N, J), np.nan)
    group = np.repeat(np.arange(G), int(spec.n_per_group))
    theta = np.empty(N)
    shapes = spec.group_shapes or [spec.shape] * G
    groups_truth = []
    for g in range(G):
        mu = float(spec.group_means[g]) if G > 1 or anchored else 0.0
        sd = float(spec.group_sds[g]) if G > 1 or anchored else 1.0
        if g == 0 and not anchored:
            mu, sd = 0.0, (1.0 if spec.model not in ("PCM", "RSM") else float(spec.group_sds[0]))
        idx = group == g
        theta[idx] = mu + sd * standardized_draw(shapes[g], idx.sum(), rng, spec.discrete_nodes, spec.discrete_weights)
        cols = forms[g]
        Y[np.ix_(idx, cols)] = sample_responses([gen_by_group[g][j] for j in cols], theta[idx], rng)
        groups_truth.append(dict(name=f"G{g + 1}" if G > 1 else "All examinees", mean=mu, sd=sd, shape=shapes[g]))
    specs = [ItemSpec(it.id, spec.ncat) for it in true_items]
    names = [gt["name"] for gt in groups_truth]
    label = name or (f"Simulated · {spec.model} · {design} · {shapes[0]} · N={N:,} · J={J}")
    truth = dict(items={it.id: it for it in true_items}, generating={it.id: it for it in drifted}, theta=theta,
                 groups=groups_truth, spec=spec.as_dict(), drifted=[true_items[k].id for k in range(min(spec.n_drift, n_anchor))])
    desc = (f"{G} group(s); {n_anchor} anchors, {n_common} common free items, {n_unique} unique items per form; "
            f"{spec.ncat} categories; seed {spec.seed}.")
    return Dataset(label, Y, group, names, specs, anchors=bank, truth=truth, description=desc)


def _shift(it: Item, delta):
    if delta == 0:
        return it
    if it.model == "RSM":
        return Item(it.id, "RSM", 1.0, (), b=it.b + delta, tau=it.tau)
    return make_item(it.id, it.model, it.a, tuple(np.asarray(it.d) + delta))


def example_datasets():
    """Ready-made datasets offered on first load."""
    out = []
    out.append(simulate(SimSpec(model="GPCM", design="single", n_per_group=1000, n_items=10, ncat=4, seed=11),
                        "Example 1 · GPCM single group (N=1000, J=10)"))
    out.append(simulate(SimSpec(model="PCM", design="concurrent", n_per_group=600, n_items=8, n_common=4, ncat=3,
                                groups=2, seed=12), "Example 2 · PCM concurrent, 2 forms"))
    out.append(simulate(SimSpec(model="GRM", design="fipc", n_per_group=1000, n_items=8, n_anchor=5, ncat=4,
                                group_means=[0.4], group_sds=[1.1], seed=13), "Example 3 · GRM FIPC (5 anchors)"))
    out.append(simulate(SimSpec(model="RSM", design="mgfipc", n_per_group=700, n_items=6, n_common=2, n_anchor=4,
                                ncat=4, groups=2, group_means=[0.2, 0.6], group_sds=[1.0, 1.1], seed=14),
                        "Example 4 · RSM MG-FIPC, 2 years"))
    out.append(simulate(SimSpec(model="GPCM", design="single", n_per_group=2000, n_items=12, ncat=4, shape="bimodal",
                                seed=15), "Example 5 · GPCM bimodal ability (prior misspecification)"))
    return out
