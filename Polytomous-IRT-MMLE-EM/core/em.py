"""One MMLE–EM engine for PCM, RSM, GPCM and GRM under four calibration designs and four ability priors.

Building blocks (each used directly by the walkthrough pages):
    estep(...)            posterior w_igq, conditional log-likelihoods, marginal log-likelihood
    expected_counts(...)  C_gjkq = Σ_{i∈g} I(Y_ij = k) w_iq
    mstep_items(...)      model-specific maximisation of Σ C log P (fixed anchors excluded)
    mstep_prior(...)      latent distribution update per distribution unit
    fit(...)              the EM loop with full per-cycle history
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import priors as P
from .data import Dataset, validate
from .models import ADJACENT, RASCH, Item, adjacent_objective, grm_objective, item_objective, log_probs, make_item
from .quadrature import GRID_NAMES, make_grid, moments, normal_weights, validate_weights

DESIGNS = ("single", "concurrent", "fipc", "mgfipc")
DESIGN_NAMES = dict(single="Single-group", concurrent="Concurrent", fipc="Fixed-item (FIPC)",
                    mgfipc="Concurrent fixed-item (MG-FIPC)")
ANCHORED = frozenset({"fipc", "mgfipc"})
LOC_BOUND = 10.0
A_BOUNDS = (0.05, 8.0)
GAP_BOUNDS = (1e-3, 30.0)


@dataclass
class FitConfig:
    model: str = "GPCM"
    design: str = "single"
    prior: str = "fixed_normal"
    pooled: bool = False                    # one common latent distribution across groups
    grid: str = "equal"                     # gauss_hermite | equal | custom
    n_nodes: int = 41
    bound: float = 6.0
    custom_nodes: list | None = None
    mean: float = 0.0                       # reference / starting mean
    sd: float = 1.0                         # reference / starting SD
    group_means: list | None = None         # fixed-normal values (or starting values) per group
    group_sds: list | None = None
    fixed_weights: list | None = None       # fixed empirical masses: one list (all groups) or one per group
    initial_weights: list | None = None     # optional starting masses for an estimated empirical prior
    normal_update: str = "exact"            # exact | moments
    empirical_update: str = "constrained"   # constrained | woods
    fipc_method: str = "MEM"                # MEM (MWU–MEM) | OEM (NWU–OEM)
    D: float = 1.0
    max_cycles: int = 500
    tol: float = 1e-4                       # largest item / group-parameter change
    prior_tol: float = 1e-4                 # largest prior-mass change
    se: bool = False

    @property
    def anchored(self):
        return self.design in ANCHORED

    def as_dict(self):
        return asdict(self)


# ---- design & identification -------------------------------------------------------------------

@dataclass
class Unit:
    """A latent-distribution unit: the groups that share one distribution, plus its constraints."""
    groups: list
    estimated: bool
    fixed_mean: float | None
    fixed_sd: float | None
    role: str

    def describe(self, names):
        g = ", ".join(names[k] for k in self.groups)
        if not self.estimated:
            return f"{g}: fixed (not updated)"
        fm = "free" if self.fixed_mean is None else f"fixed at {self.fixed_mean:g}"
        fs = "free" if self.fixed_sd is None else f"fixed at {self.fixed_sd:g}"
        return f"{g}: estimated · mean {fm} · SD {fs} · {self.role}"


def _components(adj):
    """Connected components of groups that share administered items."""
    G = adj.shape[0]
    seen, comps = set(), []
    for s in range(G):
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            g = stack.pop()
            if g in seen:
                continue
            seen.add(g)
            comp.append(g)
            stack.extend(np.flatnonzero(adj[g]).tolist())
        comps.append(sorted(comp))
    return comps


def plan(data: Dataset, cfg: FitConfig):
    """Check the design and identification. Returns dict(errors, warnings, units, fixed mask, notes)."""
    errors, warnings, notes = [], [], []
    if cfg.model not in ("PCM", "RSM", "GPCM", "GRM"):
        errors.append(f"Unknown model {cfg.model}.")
    if cfg.design not in DESIGNS:
        errors.append(f"Unknown design {cfg.design}.")
    if cfg.prior not in P.PRIOR_TYPES:
        errors.append(f"Unknown prior {cfg.prior}.")
    if errors:
        return dict(errors=errors, warnings=warnings, units=[], fixed=np.zeros(data.J, bool), notes=notes)
    e, w = validate(data, cfg.model, cfg.anchored)
    errors += e
    warnings += w
    fixed = np.array([cfg.anchored and s.id in data.anchors for s in data.specs])
    if cfg.design in ("single", "fipc") and data.G > 1:
        errors.append(f"{DESIGN_NAMES[cfg.design]} calibration uses one group; this dataset has {data.G}. "
                      "Choose a concurrent design or a single-group dataset.")
    if cfg.design in ("concurrent", "mgfipc") and data.G < 2:
        errors.append(f"{DESIGN_NAMES[cfg.design]} calibration needs at least two groups.")
    if cfg.anchored and not fixed.any():
        errors.append("An anchored design needs at least one bank item with fixed parameters in the data.")
    if not cfg.anchored and data.anchors:
        notes.append("Bank parameters in this dataset are ignored: a non-anchored design estimates every item.")
    if cfg.anchored and fixed.all():
        warnings.append("Every item is a fixed anchor: only the latent distribution can change.")
    if cfg.grid == "gauss_hermite" and cfg.prior != "fixed_normal":
        errors.append("Gauss–Hermite nodes move with μ and σ, so they only serve a fixed normal prior here. "
                      "Estimated or empirical priors need a fixed grid (equally spaced or custom).")
    if cfg.fipc_method not in ("MEM", "OEM"):
        errors.append("FIPC method must be MEM or OEM.")
    if cfg.anchored and cfg.fipc_method == "OEM" and cfg.prior.startswith("estimated"):
        notes.append("NWU–OEM: the prior is not updated before the single item M-step; the posterior-implied "
                     "distribution after that cycle is reported but was not used for estimation.")
    adm = data.administered()
    if data.G > 1:
        share = adm.astype(int) @ adm.T.astype(int) > 0
        comps = _components(share)
        if len(comps) > 1 and not cfg.anchored:
            errors.append("Groups " + " | ".join("{" + ", ".join(data.group_names[g] for g in c) + "}" for c in comps)
                          + " share no items, so their scales are not linked.")
    # Distribution units and their constraints.
    units_groups = [list(range(data.G))] if (cfg.pooled or data.G == 1) else [[g] for g in range(data.G)]
    estimated = cfg.prior.startswith("estimated")
    rasch = cfg.model in RASCH
    units = []
    anchor_groups = set(np.flatnonzero(adm[:, fixed].any(axis=1)).tolist()) if fixed.any() else set()
    for ug in units_groups:
        if not estimated:
            units.append(Unit(ug, False, None, None, "fixed by assumption"))
        elif cfg.anchored:
            units.append(Unit(ug, True, None, None, "identified by the fixed anchors"))
        elif 0 in ug:
            units.append(Unit(ug, True, cfg.mean, None if rasch else cfg.sd,
                              "reference: mean fixed" + (" (Rasch slopes fix the unit, SD estimated)" if rasch
                                                         else " and SD fixed")))
        else:
            units.append(Unit(ug, True, None, None, "focal: moments on the reference metric"))
    if cfg.anchored and estimated:
        nanch = int(fixed.sum())
        if nanch < 2:
            warnings.append(f"Only {nanch} anchor item: the metric (location and unit) rests on a single item — "
                            "weak identification.")
        for g in range(data.G):
            if g not in anchor_groups:
                warnings.append(f"Group {data.group_names[g]} saw no anchor; its scale is linked only through "
                                "shared free items.")
        if anchor_groups:
            share = adm.astype(int) @ adm.T.astype(int) > 0
            for comp in _components(share):
                if not anchor_groups & set(comp):
                    errors.append("Groups " + ", ".join(data.group_names[g] for g in comp)
                                  + " are not connected to any anchor, so their metric is not identified.")
    if estimated and not cfg.anchored and rasch:
        notes.append("Rasch identification: slopes are fixed at 1, so only the location needs a constraint; "
                     "the reference SD is estimated.")
    if cfg.prior == "estimated_normal" and not cfg.anchored and not rasch and len(units) == 1:
        notes.append("Single distribution with mean and SD fixed for identification: the estimated normal has no "
                     "free parameters here, so it equals the fixed normal N(mean, SD²).")
    return dict(errors=errors, warnings=warnings, units=units, fixed=fixed, notes=notes)


# ---- grid and starting distribution ------------------------------------------------------------

def build_grid(cfg: FitConfig, G):
    """Nodes (G, Q) and initial weights (G, Q)."""
    gm = cfg.group_means or [cfg.mean] * G
    gs = cfg.group_sds or [cfg.sd] * G
    if len(gm) != G or len(gs) != G:
        raise ValueError("Give one mean and one SD per group.")
    if cfg.grid == "gauss_hermite":
        rows = [make_grid("gauss_hermite", cfg.n_nodes, mean=gm[g], sd=gs[g]) for g in range(G)]
        return np.array([r.nodes for r in rows]), np.array([r.base_weights for r in rows])
    base = make_grid(cfg.grid, cfg.n_nodes, cfg.bound, nodes=cfg.custom_nodes)
    x = base.nodes
    nodes = np.tile(x, (G, 1))
    if cfg.prior == "fixed_empirical":
        if cfg.fixed_weights is None:
            raise ValueError("A fixed empirical prior needs weights on the grid.")
        fw = np.asarray(cfg.fixed_weights, float)
        rows = [fw] * G if fw.ndim == 1 else list(fw)
        if len(rows) != G:
            raise ValueError("Give one fixed weight vector, or one per group.")
        return nodes, np.array([validate_weights(r, len(x), "Fixed empirical weights") for r in rows])
    if cfg.prior == "estimated_empirical" and cfg.initial_weights is not None:
        iw = np.asarray(cfg.initial_weights, float)
        rows = [iw] * G if iw.ndim == 1 else list(iw)
        return nodes, np.array([validate_weights(r, len(x), "Initial weights") for r in rows])
    return nodes, np.array([normal_weights(x, gm[g], gs[g]) for g in range(G)])


# ---- E-step ------------------------------------------------------------------------------------

def _node_rows(nodes):
    """Unique node rows and the row used by each group (fixed grids share one row)."""
    if np.allclose(nodes, nodes[0]):
        return nodes[:1], np.zeros(len(nodes), int)
    return nodes, np.arange(len(nodes))


def item_log_probs(items, nodes, D=1.0):
    rows, _ = _node_rows(nodes)
    return [np.stack([log_probs(it, r, D) for r in rows]) for it in items]   # each (R, Q, K)


def conditional_loglik(data: Dataset, items, nodes, D=1.0, use=None):
    """log L_i(θ_q) = Σ_{j∈O_i} log P_{j,y_ij}(θ_q) for each examinee's own group nodes. (N, Q)."""
    rows, row_of = _node_rows(nodes)
    prow = row_of[data.group]
    L = np.zeros((data.N, nodes.shape[1]))
    lps = item_log_probs(items, nodes, D)
    obs = data.observed
    for j, lp in enumerate(lps):
        if use is not None and not use[j]:
            continue
        m = obs[:, j]
        y = data.Y[m, j].astype(int)
        L[m] += lp[prow[m], :, y] if len(rows) > 1 else lp[0][:, y].T
    return L


def estep(data: Dataset, items, nodes, weights, D=1.0, use=None):
    """Posterior w_iq ∝ π_{g(i)q} L_i(θ_q) and the marginal log-likelihood (log-sum-exp)."""
    L = conditional_loglik(data, items, nodes, D, use)
    with np.errstate(divide="ignore"):
        logpi = np.log(weights)[data.group]
    joint = L + logpi
    marg = logsumexp(joint, axis=1)
    post = np.exp(joint - marg[:, None])
    return dict(logL=L, log_joint=joint, marginal=marg, post=post, loglik=float(marg.sum()))


def marginal_loglik(data, items, nodes, weights, D=1.0):
    return estep(data, items, nodes, weights, D)["loglik"]


def expected_counts(data: Dataset, post):
    """C[j] has shape (G, Q, K_j): expected number of group-g examinees at node q answering category k."""
    obs = data.observed
    out = []
    gmask = [data.group == g for g in range(data.G)]
    for j, s in enumerate(data.specs):
        oh = np.zeros((data.N, s.ncat))
        m = obs[:, j]
        oh[np.flatnonzero(m), data.Y[m, j].astype(int)] = 1.0
        out.append(np.stack([post[gm].T @ oh[gm] for gm in gmask]))
    return out


def node_masses(data: Dataset, post):
    """n_gq = Σ_{i∈g} w_iq (expected number of group-g examinees at each node). (G, Q)."""
    return np.stack([post[data.group == g].sum(axis=0) for g in range(data.G)])


# ---- M-step for items --------------------------------------------------------------------------

def _stack(nodes, C):
    """Flatten groups: θ (G·Q,), counts (G·Q, K). The item objective is a sum over (g, q)."""
    return nodes.ravel(), C.reshape(-1, C.shape[-1])


def _optimize(fun, x0, bounds):
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=300, maxls=40, ftol=1e-13, gtol=1e-8))
    return res


def _single_item_mstep(item: Item, theta, C, D):
    """Maximise Σ C log P for one PCM/GPCM/GRM item. Returns (new item, status)."""
    K = item.K
    if item.model in ("PCM", "GPCM"):
        slope = item.model == "GPCM"

        def unpack(x):
            return (np.exp(x[0]), x[1:]) if slope else (1.0, x)

        def fun(x):
            a, d = unpack(x)
            f, ga, gd = adjacent_objective(theta, C, a, d, D)
            return -f, -(np.r_[ga, gd] if slope else gd)
        x0 = np.r_[np.log(item.a), item.d] if slope else np.array(item.d)
        bounds = ([tuple(np.log(A_BOUNDS))] if slope else []) + [(-LOC_BOUND, LOC_BOUND)] * K
    else:
        def unpack(x):
            a = np.exp(x[0])
            b = x[1] + np.r_[0.0, np.cumsum(np.exp(x[2:]))]
            return a, b

        def fun(x):
            a, b = unpack(x)
            f, ga, gb = grm_objective(theta, C, a, b, D)
            # chain rule: b_1 = u_1, b_k = u_1 + Σ_{m≤k} exp(u_m)
            gu = np.empty(K)
            gu[0] = gb.sum()
            tail = np.cumsum(gb[::-1])[::-1]
            gu[1:] = np.exp(x[2:]) * tail[1:]
            return -f, -np.r_[ga * a, gu]
        d = np.asarray(item.d)
        x0 = np.r_[np.log(item.a), d[0], np.log(np.maximum(np.diff(d), GAP_BOUNDS[0]))]
        bounds = [tuple(np.log(A_BOUNDS)), (-LOC_BOUND, LOC_BOUND)] + [tuple(np.log(GAP_BOUNDS))] * (K - 1)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    before = -fun(x0)[0]
    res = _optimize(fun, x0, bounds)
    after = -res.fun
    accepted = bool(np.isfinite(after) and after >= before - 1e-9)
    x = res.x if accepted else x0
    a, d = unpack(x)
    new = make_item(item.id, item.model, float(a), tuple(float(v) for v in d))
    at_bound = bool(np.any(np.isclose(x, [b[0] for b in bounds], atol=1e-6) |
                           np.isclose(x, [b[1] for b in bounds], atol=1e-6)))
    return new, dict(item=item.id, success=bool(res.success), accepted=accepted, iterations=int(res.nit),
                     q_before=float(before), q_after=float(after if accepted else before),
                     gradient=float(np.max(np.abs(res.jac))), at_bound=at_bound, message=str(res.message))


def _rsm_mstep(items, idx_free, idx_fixed, thetaC, D):
    """Joint M-step for one RSM scale group: free locations b_j and (if no anchor fixes them) shared steps τ.

    δ_jh = b_j + τ_h, τ_K = −Σ_{h<K} τ_h. Gradient: ∂/∂b_j = Σ_h g_jh, ∂/∂τ_h = Σ_j (g_jh − g_jK)."""
    ref = items[(idx_free + idx_fixed)[0]]
    K = ref.K
    tau_free = len(idx_fixed) == 0
    tau_fixed = np.array(items[idx_fixed[0]].tau) if idx_fixed else None
    nf = len(idx_free)

    def unpack(x):
        b = x[:nf]
        tau = np.r_[x[nf:], -x[nf:].sum()] if tau_free else tau_fixed
        return b, tau

    def fun(x):
        b, tau = unpack(x)
        f = 0.0
        gb = np.zeros(nf)
        gt = np.zeros(K)
        for k, j in enumerate(idx_free):
            theta, C = thetaC[j]
            fj, _, gd = adjacent_objective(theta, C, 1.0, b[k] + tau, D)
            f += fj
            gb[k] = gd.sum()
            gt += gd
        g = np.r_[gb, (gt[:-1] - gt[-1]) if tau_free else []]
        return -f, -g
    x0 = np.r_[[items[j].b for j in idx_free], (np.array(items[idx_free[0]].tau[:-1]) if tau_free else [])]
    bounds = [(-LOC_BOUND, LOC_BOUND)] * len(x0)
    before = -fun(x0)[0]
    res = _optimize(fun, x0, bounds)
    after = -res.fun
    accepted = bool(np.isfinite(after) and after >= before - 1e-9)
    x = res.x if accepted else x0
    b, tau = unpack(x)
    tau = tau - tau.mean()                     # exact zero sum (already zero up to rounding)
    new = {j: Item(items[j].id, "RSM", 1.0, (), b=float(b[k]), tau=tuple(tau)) for k, j in enumerate(idx_free)}
    status = dict(item="RSM scale: " + ", ".join(items[j].id for j in idx_free), success=bool(res.success),
                  accepted=accepted, iterations=int(res.nit), q_before=float(before),
                  q_after=float(after if accepted else before), gradient=float(np.max(np.abs(res.jac))),
                  at_bound=bool(np.any(np.isclose(np.abs(x), LOC_BOUND))),
                  message=("steps τ fixed by anchors · " if not tau_free else "joint b and shared τ · ") + str(res.message))
    return new, status


def mstep_items(data: Dataset, items, counts, nodes, fixed, D=1.0, use_items=None):
    """Update every free item; fixed anchors are returned unchanged. Returns (items, statuses)."""
    new = list(items)
    statuses = []
    thetaC = {j: _stack(nodes, counts[j]) for j in range(len(items))}
    rsm_scales = {}
    for j, it in enumerate(items):
        if fixed[j]:
            continue
        if it.model == "RSM":
            rsm_scales.setdefault(data.specs[j].scale, []).append(j)
            continue
        new[j], st = _single_item_mstep(it, *thetaC[j], D)
        statuses.append(st)
    for scale, free in rsm_scales.items():
        anchors = [j for j in range(len(items)) if fixed[j] and data.specs[j].scale == scale]
        upd, st = _rsm_mstep(items, free, anchors, thetaC, D)
        for j, it in upd.items():
            new[j] = it
        statuses.append(st)
    return new, statuses


def q_item_total(items, counts, nodes, fixed, D=1.0):
    return float(sum(item_objective(it, *_stack(nodes, counts[j]), D) for j, it in enumerate(items) if not fixed[j]))


# ---- M-step for the latent distribution ---------------------------------------------------------

def mstep_prior(units, n_g, nodes, weights, cfg: FitConfig):
    """Update each estimated unit from pooled node masses. Returns (weights (G,Q), infos)."""
    new = weights.copy()
    infos = []
    for u in units:
        if not u.estimated:
            continue
        n = n_g[u.groups].sum(axis=0)                 # pooled design: sample-size weighting by summing
        x = nodes[u.groups[0]]
        if cfg.prior == "estimated_empirical":
            if u.fixed_mean is None and u.fixed_sd is None:
                w, info = P.empirical_unconstrained(n), dict(method="Unconstrained empirical (fractional counts)")
            elif cfg.empirical_update == "woods":
                raw = P.empirical_unconstrained(n)
                sd = u.fixed_sd if u.fixed_sd is not None else moments(x, raw)[1]
                w = P.woods_rescale(x, raw, u.fixed_mean, sd)
                info = dict(method="Woods rescaling (irtQ EmpHist; generalised step)")
            else:
                w, info = P.empirical_constrained(n, x, u.fixed_mean, u.fixed_sd)
        else:
            m0, s0 = moments(x, weights[u.groups[0]])
            if cfg.normal_update == "moments":
                w, mu, sd, info = P.normal_moments(n, x, u.fixed_mean, u.fixed_sd)
            else:
                w, mu, sd, info = P.normal_exact(n, x, u.fixed_mean, u.fixed_sd, start=(m0, s0))
        info = dict(info, groups=list(u.groups), q_before=P.objective(n, weights[u.groups[0]]), q_after=P.objective(n, w))
        for g in u.groups:
            new[g] = w
        infos.append(info)
    return new, infos


# ---- starting values ----------------------------------------------------------------------------

def start_items(data: Dataset, model, fixed):
    items = []
    for j, s in enumerate(data.specs):
        if fixed[j]:
            items.append(data.anchors[s.id])
            continue
        c = data.counts(j) + 0.5
        if model == "GRM":
            ge = np.cumsum(c[::-1])[::-1][1:] / c.sum()
            b = np.clip(-np.log(ge / (1 - ge)), -4, 4)
            b = np.maximum.accumulate(b + np.arange(len(b)) * 1e-3)
            for k in range(1, len(b)):
                b[k] = max(b[k], b[k - 1] + 0.05)
            items.append(make_item(s.id, "GRM", 1.0, tuple(b)))
        else:
            d = np.clip(np.log(c[:-1] / c[1:]), -4, 4)
            items.append(make_item(s.id, model, 1.0, tuple(d)))
    if model == "RSM":                         # common τ start: average centred steps of free items per scale
        for scale in {s.scale for s in data.specs}:
            js = [j for j, s in enumerate(data.specs) if s.scale == scale and not fixed[j]]
            if not js:
                continue
            anchors = [j for j, s in enumerate(data.specs) if s.scale == scale and fixed[j]]
            tau = np.array(items[anchors[0]].tau) if anchors else np.mean([np.array(items[j].tau) for j in js], axis=0)
            tau = tau - tau.mean()
            for j in js:
                items[j] = Item(items[j].id, "RSM", 1.0, (), b=items[j].b, tau=tuple(tau))
    return items


def check_rsm_anchors(data, fixed):
    errs = []
    for scale in {s.scale for s in data.specs}:
        taus = [np.array(data.anchors[s.id].tau) for j, s in enumerate(data.specs) if fixed[j] and s.scale == scale]
        if len(taus) > 1 and any(np.max(np.abs(t - taus[0])) > 1e-6 for t in taus):
            errs.append(f"RSM anchors in '{scale}' carry different steps τ; a shared rating scale needs one τ.")
    return errs


# ---- the EM loop --------------------------------------------------------------------------------

@dataclass
class FitResult:
    config: FitConfig
    data_id: str
    ids: list
    items: list
    start: list
    nodes: np.ndarray
    weights: np.ndarray
    initial_weights: np.ndarray
    units: list
    fixed: np.ndarray
    history: list
    loglik: float
    converged: bool
    reason: str
    cycles: int
    seconds: float
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    implied_weights: np.ndarray | None = None
    se: dict | None = None
    group_names: list = field(default_factory=list)
    loglik_implied: float | None = None     # OEM: log-likelihood under the reported (implied) distribution

    @property
    def reported_weights(self):
        """Distribution reported to the user: the implied update for NWU–OEM, otherwise the fitted weights."""
        return self.implied_weights if self.implied_weights is not None else self.weights

    @property
    def group_moments(self):
        return np.array([moments(self.nodes[g], self.weights[g]) for g in range(len(self.weights))])

    def n_parameters(self):
        """Free item parameters + free latent-distribution parameters (for AIC/BIC)."""
        k = 0
        rsm_scales = set()
        for it, f in zip(self.items, self.fixed):
            if f:
                continue
            k += it.K if it.model != "RSM" else 1
            k += 1 if it.model in ("GPCM", "GRM") else 0
            if it.model == "RSM":
                rsm_scales.add(it.K)
        if self.items and self.items[0].model == "RSM":
            anchored_scales = {it.K for it, f in zip(self.items, self.fixed) if f}
            k += sum(K - 1 for K in rsm_scales if K not in anchored_scales)
        for u in self.units:
            if not u.estimated:
                continue
            if self.config.prior == "estimated_normal":
                k += (u.fixed_mean is None) + (u.fixed_sd is None)
            else:
                k += self.nodes.shape[1] - 1 - (u.fixed_mean is not None) - (u.fixed_sd is not None)
        return k

    def item_table(self):
        import pandas as pd
        rows = []
        for it, f in zip(self.items, self.fixed):
            r = it.as_row()
            r["status"] = "fixed anchor" if f else "estimated"
            if self.se and it.id in self.se.get("items", {}):
                for lab, v in self.se["items"][it.id].items():
                    r[f"SE {lab}"] = v
            rows.append(r)
        return pd.DataFrame(rows)


def _change(a_items, b_items):
    return float(max((np.max(np.abs(x.vector() - y.vector())) for x, y in zip(a_items, b_items)), default=0.0))


def fit(data: Dataset, cfg: FitConfig, start=None, callback=None, record=True):
    """Run MMLE–EM. Raises ValueError on design / identification errors."""
    t0 = time.perf_counter()
    pl = plan(data, cfg)
    fixed = pl["fixed"]
    errs = pl["errors"] + (check_rsm_anchors(data, fixed) if cfg.model == "RSM" and cfg.anchored else [])
    if errs:
        raise ValueError(" ".join(errs))
    nodes, weights = build_grid(cfg, data.G)
    init_w = weights.copy()
    items = list(start) if start is not None else start_items(data, cfg.model, fixed)
    start_list = list(items)
    units = pl["units"]
    estimate_prior = any(u.estimated for u in units)
    one_cycle = cfg.anchored and cfg.fipc_method == "OEM"
    anchors_first = cfg.anchored
    history = []
    converged, reason = False, "Maximum cycles reached"
    prev_ll = None
    implied = None
    limit = 1 if one_cycle else int(cfg.max_cycles)
    for t in range(1, limit + 1):
        use = fixed if (anchors_first and t == 1) else None
        es = estep(data, items, nodes, weights, cfg.D, use)
        ll_all = es["loglik"] if use is None else marginal_loglik(data, items, nodes, weights, cfg.D)
        counts = expected_counts(data, es["post"])
        q_before = q_item_total(items, counts, nodes, fixed, cfg.D)
        new_items, statuses = mstep_items(data, items, counts, nodes, fixed, cfg.D)
        n_g = node_masses(data, es["post"])
        if estimate_prior and not one_cycle:
            new_w, pinfo = mstep_prior(units, n_g, nodes, weights, cfg)
        else:
            new_w, pinfo = weights.copy(), []
            if one_cycle and estimate_prior:
                implied, _ = mstep_prior(units, n_g, nodes, weights, cfg)
        dp = _change(items, new_items)
        dw = float(np.max(np.abs(new_w - weights)))
        dm = float(np.max(np.abs(np.array([moments(nodes[g], new_w[g]) for g in range(data.G)])
                                 - np.array([moments(nodes[g], weights[g]) for g in range(data.G)]))))
        decrease = prev_ll is not None and ll_all < prev_ll - 1e-7 * max(1.0, abs(prev_ll)) and not (anchors_first and t == 2)
        entry = dict(cycle=t, loglik=ll_all, ll_change=None if prev_ll is None else ll_all - prev_ll,
                     estep_items="anchors only" if use is not None else "all items",
                     item_change=dp, weight_change=dw, group_change=dm,
                     q_item_before=q_before, q_item_after=q_item_total(new_items, counts, nodes, fixed, cfg.D),
                     optimizer_ok=all(s["success"] or s["accepted"] for s in statuses) if statuses else True,
                     ll_decrease=bool(decrease))
        if record:
            entry.update(items_before=items, items_after=new_items, weights_before=weights, weights_after=new_w,
                         statuses=statuses, prior_info=pinfo)
        history.append(entry)
        prev_ll = ll_all
        items, weights = new_items, new_w
        if callback:
            callback(entry)
        if one_cycle:
            reason = "NWU–OEM: one EM cycle by design (not a converged MEM solution)"
            break
        stable = dp < cfg.tol and (not estimate_prior or (dw < cfg.prior_tol and dm < cfg.tol))
        if stable and t > (1 if anchors_first else 0):
            converged = True
            reason = "Item" + (", prior-mass and group-moment" if estimate_prior else "") + " changes below tolerance"
            break
    ll = marginal_loglik(data, items, nodes, weights, cfg.D)
    res = FitResult(cfg, data.identity, data.ids, items, start_list, nodes, weights, init_w, units, fixed, history,
                    ll, converged, reason, len(history), time.perf_counter() - t0, pl["warnings"], pl["notes"],
                    implied, None, list(data.group_names))
    if implied is not None:
        res.loglik_implied = marginal_loglik(data, items, nodes, implied, cfg.D)
    if any(h["ll_decrease"] for h in history):
        res.warnings.append("The marginal log-likelihood decreased in at least one cycle — expected only for "
                            "generalised (non-exact) prior updates such as Woods rescaling or moment matching.")
    if any(s.get("at_bound") for h in history[-1:] for s in h.get("statuses", [])):
        res.warnings.append("Some parameters ended at an optimizer bound (sparse or empty categories, or weak "
                            "identification). Treat those estimates with caution.")
    if cfg.se:
        res.se = standard_errors(data, res)
    return res


# ---- standard errors -----------------------------------------------------------------------------

def _free_layout(items, fixed):
    """Canonical free parameters: (item index, label) list; RSM steps τ_1..τ_{K−1} shared per scale."""
    lay = []
    rsm_tau_done = set()
    for j, (it, f) in enumerate(zip(items, fixed)):
        if f:
            continue
        if it.model == "RSM":
            lay.append((j, "b"))
            if it.K not in rsm_tau_done and not any(fixed[k] and items[k].K == it.K for k in range(len(items))):
                rsm_tau_done.add(it.K)
                lay += [(j, f"τ{h + 1}") for h in range(it.K - 1)]
        else:
            if it.model in ("GPCM", "GRM"):
                lay.append((j, "a"))
            lay += [(j, ("b" if it.model == "GRM" else "δ") + str(h + 1)) for h in range(it.K)]
    return lay


def _set_params(items, lay, v):
    items = list(items)
    by = {}
    for (j, lab), val in zip(lay, v):
        by.setdefault(j, {})[lab] = val
    tau_by_K = {}
    for j, d in by.items():
        for lab, val in d.items():
            if lab.startswith("τ"):
                tau_by_K.setdefault(items[j].K, {})[int(lab[1:]) - 1] = val
    for j, it in enumerate(items):
        d = by.get(j, {})
        if it.model == "RSM":
            if j not in by and it.K not in tau_by_K:
                continue
            b = d.get("b", it.b)
            tau = np.array(it.tau)
            if it.K in tau_by_K:
                t = tau_by_K[it.K]
                tau[:-1] = [t.get(h, tau[h]) for h in range(it.K - 1)]
                tau[-1] = -tau[:-1].sum()
            items[j] = Item(it.id, "RSM", 1.0, (), b=b, tau=tuple(tau))
        elif d:
            a = d.get("a", it.a)
            dd = [d.get(("b" if it.model == "GRM" else "δ") + str(h + 1), it.d[h]) for h in range(it.K)]
            items[j] = make_item(it.id, it.model, a, tuple(dd))
    return items


def observed_gradient(data, items, nodes, weights, fixed, lay, D=1.0):
    """∇ log-likelihood in canonical parameters via Fisher's identity (gradient of Q at the current point)."""
    es = estep(data, items, nodes, weights, D)
    counts = expected_counts(data, es["post"])
    g = {}
    tau_g = {}
    for j, it in enumerate(items):
        if fixed[j]:
            continue
        theta, C = _stack(nodes, counts[j])
        if it.model == "GRM":
            _, ga, gb = grm_objective(theta, C, it.a, it.d, D)
            g[(j, "a")] = ga
            for h in range(it.K):
                g[(j, f"b{h + 1}")] = gb[h]
        else:
            _, gla, gd = adjacent_objective(theta, C, it.a, it.d, D)
            if it.model == "GPCM":
                g[(j, "a")] = gla / it.a
            if it.model == "RSM":
                g[(j, "b")] = gd.sum()
                tau_g.setdefault(it.K, np.zeros(it.K))
                tau_g[it.K] += gd
            else:
                for h in range(it.K):
                    g[(j, f"δ{h + 1}")] = gd[h]
    out = []
    for j, lab in lay:
        if lab.startswith("τ"):
            gt = tau_g[items[j].K]
            out.append(gt[int(lab[1:]) - 1] - gt[-1])
        else:
            out.append(g[(j, lab)])
    return np.array(out)


def standard_errors(data, res: FitResult, h=1e-4, max_params=400):
    """Observed-information SEs: numerical Jacobian of the Fisher-identity gradient, latent distribution held at
    its estimate (so SEs are conditional on the estimated ability distribution)."""
    lay = _free_layout(res.items, res.fixed)
    if not lay:
        return dict(status="No free item parameters")
    if len(lay) > max_params:
        return dict(status=f"Skipped: {len(lay)} free parameters exceed the limit of {max_params}")
    v0 = np.array([_get(res.items, j, lab) for j, lab in lay])
    H = np.zeros((len(lay), len(lay)))
    for k in range(len(lay)):
        e = np.zeros(len(lay))
        e[k] = h
        gp = observed_gradient(data, _set_params(res.items, lay, v0 + e), res.nodes, res.weights, res.fixed, lay, res.config.D)
        gm = observed_gradient(data, _set_params(res.items, lay, v0 - e), res.nodes, res.weights, res.fixed, lay, res.config.D)
        H[:, k] = (gp - gm) / (2 * h)
    H = (H + H.T) / 2
    try:
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        status = "Observed information (conditional on the estimated latent distribution)"
    except np.linalg.LinAlgError:
        se = np.full(len(lay), np.nan)
        status = "Information matrix is singular (weak identification)"
    out = {}
    for (j, lab), s in zip(lay, se):
        out.setdefault(res.items[j].id, {})[lab] = float(s)
    # RSM: τ_K = −Σ τ_h → its SE from the covariance of the free steps.
    return dict(status=status, items=out, layout=[(res.items[j].id, lab) for j, lab in lay])


def _get(items, j, lab):
    it = items[j]
    if lab == "a":
        return it.a
    if lab == "b" and it.model == "RSM":
        return it.b
    if lab.startswith("τ"):
        return it.tau[int(lab[1:]) - 1]
    return it.d[int(lab[1:]) - 1]


# ---- inspection of any recorded cycle ------------------------------------------------------------

def inspect_cycle(data: Dataset, res: FitResult, t):
    """Recompute every intermediate quantity of cycle t from its recorded starting state."""
    h = res.history[t - 1]
    if "items_before" not in h:
        raise ValueError("This fit was run without recording; re-run with recording on.")
    use = res.fixed if (res.config.anchored and t == 1) else None
    es = estep(data, h["items_before"], res.nodes, h["weights_before"], res.config.D, use)
    counts = expected_counts(data, es["post"])
    return dict(estep=es, counts=counts, n_g=node_masses(data, es["post"]), entry=h)
