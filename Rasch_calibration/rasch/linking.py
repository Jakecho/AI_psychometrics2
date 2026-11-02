"""Two-form linking designs for the Rasch model.

Setting: an item bank already exists. K anchor items have bank difficulties from an earlier
calibration. Two new forms are given to two groups: Form A (group A) and Form B (group B).
Both forms contain the anchors and a few internal common items (shared by the two forms but
not in the bank), plus items unique to each form. Every design below ends with
all items on the bank scale, so the designs can be compared with each other and with truth.

  separate           calibrate each form on its own (its own origin), then shift each form so
                     its anchor mean equals the bank anchor mean (mean/mean linking; for the
                     Rasch model the slope is 1, so only a shift is needed)
  concurrent         calibrate the stacked A + B matrix once (unadministered items missing),
                     then one shift so the anchor mean equals the bank mean
  fixed              calibrate each form on its own with the anchors fixed at their bank values
  fixed_concurrent   calibrate the stacked matrix once with the anchors fixed at bank values

For MML-EM the stacked designs use one normal ability distribution per group; with fixed
anchors every group mean is free (the anchors set the scale).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time

import numpy as np

from .cmle import cmle, displacement as cmle_displacement
from .jmle import JMLESettings, displacement as jmle_displacement, jmle
from .metrics import recovery
from .mml_em import MMLSettings, mml_em_groups
from .model import probability
from .scoring import score_persons

DESIGNS = ("separate", "concurrent", "fixed", "fixed_concurrent")
DESIGN_LABELS = dict(separate="Single-form calibration + mean/mean linking",
                     concurrent="Concurrent calibration",
                     fixed="Fixed-anchor calibration",
                     fixed_concurrent="Fixed anchor + concurrent calibration")
SHORT = dict(separate="Separate + link", concurrent="Concurrent", fixed="Fixed anchor",
             fixed_concurrent="Fixed + concurrent")
METHODS = ("JMLE", "CMLE", "MML")


@dataclass
class LinkingData:
    X: np.ndarray                  # stacked persons x items, NaN = not administered
    groups: np.ndarray             # 0 = group A (Form A), 1 = group B (Form B)
    item_ids: list
    roles: np.ndarray              # "anchor" (in bank, on both forms), "common" (both forms, no bank value), "A", "B"
    true_b: np.ndarray             # bank-scale truth (as administered in Form A)
    true_b_form_b: np.ndarray      # truth in Form B (anchor drift applied)
    true_theta: np.ndarray
    bank: np.ndarray               # bank difficulty for anchors, NaN otherwise
    settings: dict = field(default_factory=dict)

    @property
    def anchors(self):
        return np.flatnonzero(self.roles == "anchor")

    def form_items(self, form):
        return np.flatnonzero(np.isin(self.roles, ["anchor", "common", form]))

    def form_rows(self, form):
        return np.flatnonzero(self.groups == (0 if form == "A" else 1))

    @property
    def identity(self):
        h = hashlib.sha256(np.nan_to_num(self.X, nan=-1).astype(np.int8).tobytes())
        h.update(repr(sorted(self.settings.items())).encode())
        return h.hexdigest()[:12]

    @property
    def true_mean_difference(self):
        return float(self.true_theta[self.groups == 1].mean() - self.true_theta[self.groups == 0].mean())


def simulate_forms(n_a=500, n_b=500, n_anchor=6, n_common=4, n_unique_a=10, n_unique_b=10, mean_b=0.5, sd_b=1.0,
                   b_sd=1.0, drift_items=0, drift=0.5, bank_error=0.0, seed=2026, true_b=None):
    """Group A: theta ~ N(0, 1); group B: theta ~ N(mean_b, sd_b^2). Items b ~ N(0, b_sd^2).

    The first `drift_items` anchors are `drift` logits harder in Form B than in the bank (item
    parameter drift). Bank values = true anchor difficulties + N(0, bank_error^2) noise, standing
    for the error of the earlier calibration. `true_b` holds the items fixed (Monte Carlo).
    """
    if n_anchor < 1 or n_common < 0:
        raise ValueError("Use at least one anchor item (internal common items may be 0).")
    if min(n_unique_a, n_unique_b) < 1 or min(n_a, n_b) < 20:
        raise ValueError("Each form needs at least one unique item and 20 persons.")
    if not 0 <= drift_items <= n_anchor:
        raise ValueError("Drifting anchors must be between 0 and the number of anchors.")
    if sd_b <= 0 or b_sd <= 0 or bank_error < 0:
        raise ValueError("Standard deviations must be positive (bank error may be 0).")
    rng = np.random.default_rng(int(seed))
    J = n_anchor + n_common + n_unique_a + n_unique_b
    roles = np.array(["anchor"] * n_anchor + ["common"] * n_common + ["A"] * n_unique_a + ["B"] * n_unique_b)
    b = rng.normal(0, b_sd, J) if true_b is None else np.asarray(true_b, float)
    if b.shape != (J,):
        raise ValueError("true_b must have one value per item.")
    b_form_b = b.copy()
    b_form_b[:drift_items] += drift
    theta = np.r_[rng.normal(0, 1, n_a), rng.normal(mean_b, sd_b, n_b)]
    groups = np.r_[np.zeros(n_a, int), np.ones(n_b, int)]
    X = np.full((n_a + n_b, J), np.nan)
    a_items = np.flatnonzero(roles != "B")
    b_items = np.flatnonzero(roles != "A")
    X[:n_a][:, a_items] = (rng.random((n_a, len(a_items))) < probability(theta[:n_a], b[a_items])).astype(float)
    X[n_a:, b_items] = (rng.random((n_b, len(b_items))) < probability(theta[n_a:], b_form_b[b_items])).astype(float)
    bank = np.full(J, np.nan)
    bank[:n_anchor] = b[:n_anchor] + rng.normal(0, bank_error, n_anchor) if bank_error > 0 else b[:n_anchor]
    ids = ([f"K{k + 1:02d}" for k in range(n_anchor)] + [f"C{k + 1:02d}" for k in range(n_common)]
           + [f"A{k + 1:02d}" for k in range(n_unique_a)]
           + [f"B{k + 1:02d}" for k in range(n_unique_b)])
    settings = dict(n_a=n_a, n_b=n_b, n_anchor=n_anchor, n_common=n_common, n_unique_a=n_unique_a, n_unique_b=n_unique_b, mean_b=mean_b,
                    sd_b=sd_b, b_sd=b_sd, drift_items=drift_items, drift=drift, bank_error=bank_error, seed=int(seed))
    return LinkingData(X=X, groups=groups, item_ids=ids, roles=roles, true_b=b, true_b_form_b=b_form_b,
                       true_theta=theta, bank=bank, settings=settings)


# ---------------------------------------------------------------------------------------------
# One calibration run (any estimator, optional groups and anchors)
# ---------------------------------------------------------------------------------------------
@dataclass
class Run:
    """One call of an estimator inside a design.

    `items` are indices into the full item list for the first columns of `X`; any further columns
    are Form B copies of anchors dropped from the anchor set (split items, see run_design).
    """
    name: str
    rows: np.ndarray
    items: np.ndarray
    X: np.ndarray
    column_ids: list
    groups: np.ndarray | None
    fixed_b: np.ndarray | None     # per item in `items`, NaN = free
    b: np.ndarray                  # native estimates for `items`
    se_b: np.ndarray
    group_means: np.ndarray        # per group present (A, B order), native scale
    group_sds: np.ndarray
    displacement: np.ndarray       # per item in `items` (NaN for free items)
    theta: np.ndarray | None
    converged: bool
    status: str
    iterations: int
    seconds: float


def calibrate(method, X, groups=None, fixed_b=None, jmle_settings=None, mml_settings=None, compute_se=True):
    """Run one estimator and return native b, SEs, group means and anchor displacement."""
    started = time.perf_counter()
    groups = None if groups is None else np.asarray(groups)
    labels = [0] if groups is None else sorted(np.unique(groups).tolist())
    g_index = np.zeros(len(X), int) if groups is None else np.searchsorted(labels, groups)
    fixed = np.zeros(X.shape[1], bool) if fixed_b is None else np.isfinite(fixed_b)
    disp = np.full(X.shape[1], np.nan)
    if method == "JMLE":
        r = jmle(X, jmle_settings or JMLESettings(), fixed_b=fixed_b)
        b, se, theta = r.b, r.se_b, r.theta
        if fixed.any():
            disp[fixed] = jmle_displacement(X, r.theta, r.b_uncorrected)[fixed]
        means = np.array([theta[g_index == k].mean() for k in range(len(labels))])
        sds = np.array([theta[g_index == k].std() for k in range(len(labels))])
        out = (r.converged, r.status, r.iterations)
    elif method == "CMLE":
        r = cmle(X, fixed_b=fixed_b, compute_se=compute_se)
        b, se = r.b, r.se_b
        if fixed.any():
            disp[fixed] = cmle_displacement(X, b)[fixed]
        theta = score_persons(X, b, "WLE")[0]
        means = np.array([theta[g_index == k].mean() for k in range(len(labels))])
        sds = np.array([theta[g_index == k].std() for k in range(len(labels))])
        out = (r.converged, r.status, r.iterations)
    elif method == "MML":
        r = mml_em_groups(X, groups, fixed_b, mml_settings or MMLSettings())
        b, se, theta, means, sds = r.b, r.se_b, r.eap, r.means, r.sds
        disp = r.displacement
        out = (r.converged, r.status, r.iterations)
    else:
        raise ValueError(f"Unknown method {method}")
    return dict(b=np.asarray(b, float), se_b=np.asarray(se, float), theta=theta, group_means=np.asarray(means, float),
                group_sds=np.asarray(sds, float), displacement=disp, converged=out[0], status=out[1],
                iterations=out[2], seconds=time.perf_counter() - started)


def _run(name, method, ld, rows, items, use_groups, fixed_b, X=None, column_ids=None, **kw):
    X = ld.X[np.ix_(rows, items)] if X is None else X
    column_ids = [ld.item_ids[j] for j in items] if column_ids is None else column_ids
    groups = ld.groups[rows] if use_groups else None
    c = calibrate(method, X, groups, fixed_b, **kw)
    return Run(name=name, rows=rows, items=items, X=X, column_ids=column_ids, groups=groups, fixed_b=fixed_b, b=c["b"], se_b=c["se_b"],
               group_means=c["group_means"], group_sds=c["group_sds"], displacement=c["displacement"], theta=c["theta"],
               converged=c["converged"], status=c["status"], iterations=c["iterations"], seconds=c["seconds"])


# ---------------------------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------------------------
@dataclass
class DesignResult:
    design: str
    method: str
    b: np.ndarray                  # every item on the bank scale
    estimated: np.ndarray          # True for items estimated (not fixed at bank values)
    group_means: np.ndarray        # (A, B) on the bank scale
    shifts: dict                   # linking constants per run (0 for fixed designs)
    runs: list
    anchor_displacement: np.ndarray   # per anchor used: estimate - bank (free designs) or displacement (fixed)
    by_form: dict                  # form -> item estimates on the bank scale (NaN where not on the form)
    unlinked: dict                 # form -> native item estimates before linking (separate design)
    seconds: float


def run_design(design, method, ld: LinkingData, anchors=None, jmle_settings=None, mml_settings=None,
               single_population=False, compute_se=True):
    """Run one linking design. `anchors`: indices of anchor items used for linking (default all)."""
    if design not in DESIGNS:
        raise ValueError(f"Unknown design {design}")
    started = time.perf_counter()
    anchors = ld.anchors if anchors is None else np.asarray(anchors, int)
    if len(anchors) == 0:
        raise ValueError("Select at least one anchor item for linking.")
    J = len(ld.item_ids)
    kw = dict(jmle_settings=jmle_settings, mml_settings=mml_settings, compute_se=compute_se)
    bank_mean = float(np.mean(ld.bank[anchors]))
    by_form, unlinked, shifts, runs = {}, {}, {}, []
    group_means = np.full(2, np.nan)
    anchor_disp = np.full(len(anchors), np.nan)
    estimated = np.ones(J, bool)
    if design in ("fixed", "fixed_concurrent"):
        estimated[anchors] = False

    def fixed_for(items):
        f = np.full(len(items), np.nan)
        pos = np.flatnonzero(np.isin(items, anchors))
        f[pos] = ld.bank[items[pos]]
        return f

    if design in ("separate", "fixed"):
        for k, form in enumerate("AB"):
            rows, items = ld.form_rows(form), ld.form_items(form)
            fx = fixed_for(items) if design == "fixed" else None
            run = _run(f"Form {form}", method, ld, rows, items, False, fx, **kw)
            runs.append(run)
            pos = np.searchsorted(items, anchors)
            shift = 0.0 if design == "fixed" else bank_mean - float(np.mean(run.b[pos]))
            shifts[f"Form {form}"] = shift
            full = np.full(J, np.nan)
            full[items] = run.b + shift
            by_form[form] = full
            raw = np.full(J, np.nan)
            raw[items] = run.b
            unlinked[form] = raw
            group_means[k] = run.group_means[0] + shift
        b = np.nanmean(np.vstack([by_form["A"], by_form["B"]]), axis=0)
        if design == "separate":
            anchor_disp = by_form["B"][anchors] - by_form["A"][anchors]
        else:
            pos_b = np.searchsorted(ld.form_items("B"), anchors)
            anchor_disp = runs[1].displacement[pos_b]
    else:
        rows, items = np.arange(len(ld.X)), np.arange(J)
        # A bank anchor left out of the anchor set must not link the forms either: split it into a
        # Form A item (original column) and a Form B item (extra column), as for a drifted item.
        dropped = np.setdiff1d(ld.anchors, anchors)
        X = ld.X.copy()
        ids = list(ld.item_ids)
        in_b = ld.groups == 1
        for j in dropped:
            col = np.where(in_b, X[:, j], np.nan)
            X[in_b, j] = np.nan
            X = np.column_stack([X, col])
            ids.append(ld.item_ids[j] + " (Form B)")
        fx = None
        if design == "fixed_concurrent":
            fx = np.r_[fixed_for(items), np.full(len(dropped), np.nan)]
        use_groups = method == "MML" and not single_population
        run = _run("Forms A + B", method, ld, rows, items, use_groups, fx, X=X, column_ids=ids, **kw)
        runs.append(run)
        shift = 0.0 if design == "fixed_concurrent" else bank_mean - float(np.mean(run.b[anchors]))
        shifts["Forms A + B"] = shift
        b = run.b[:J] + shift
        split_b = run.b[J:] + shift
        if use_groups:
            group_means = run.group_means + shift
        else:
            # One population (or JMLE / CMLE): group means from the person estimates by group.
            theta = run.theta
            group_means = np.array([theta[ld.groups == g].mean() for g in (0, 1)]) + shift
        for form in "AB":
            full = np.full(J, np.nan)
            full[ld.form_items(form)] = b[ld.form_items(form)]
            by_form[form] = full
        by_form["B"][dropped] = split_b
        b[dropped] = (b[dropped] + split_b) / 2       # same convention as the separate designs: average the forms
        anchor_disp = b[anchors] - ld.bank[anchors] if design == "concurrent" else run.displacement[anchors]
    if design in ("fixed", "fixed_concurrent"):
        b[anchors] = ld.bank[anchors]
    return DesignResult(design=design, method=method, b=b, estimated=estimated, group_means=group_means, shifts=shifts,
                        runs=runs, anchor_displacement=anchor_disp, by_form=by_form, unlinked=unlinked,
                        seconds=time.perf_counter() - started)


def run_all(method, ld, anchors=None, single_population=False, **kw):
    return {d: run_design(d, method, ld, anchors, single_population=single_population, **kw) for d in DESIGNS}


def anchor_stability(ld: LinkingData, separate: DesignResult, threshold=0.5, z_crit=1.96):
    """Every bank anchor from the separate (free) calibrations of the two forms.

    d_k = b_k(Form B) - b_k(Form A) on the forms' own scales differs between anchors only through
    drift and error; the common part is the scale difference. B_minus_A = d_k - median(d) removes
    that common part robustly: centring on the mean of all anchors would let a few drifted anchors
    push the stable ones the other way (masking/swamping). z divides by the combined SE. An anchor is
    flagged when |B_minus_A| > threshold and |z| > z_crit. form_A / form_B are the linked estimates.
    """
    se = {}
    for run, form in zip(separate.runs, "AB"):
        full = np.full(len(ld.item_ids), np.nan)
        full[run.items] = run.se_b
        se[form] = full
    idx = ld.anchors
    d = separate.unlinked["B"][idx] - separate.unlinked["A"][idx]
    centred = d - np.median(d)
    rows = []
    for k, j in enumerate(idx):
        z = centred[k] / np.sqrt(se["A"][j] ** 2 + se["B"][j] ** 2)
        flag = "Unstable" if abs(centred[k]) > threshold and abs(z) > z_crit else "Stable"
        rows.append(dict(item=ld.item_ids[j], bank=float(ld.bank[j]), form_A=float(separate.by_form["A"][j]),
                         form_B=float(separate.by_form["B"][j]), B_minus_A=float(centred[k]), z=float(z), flag=flag))
    return rows


def evaluate(ld: LinkingData, res: DesignResult, anchors=None):
    """Recovery on the items that are not bank anchors (estimated in every design), plus group means.

    The target set does not depend on which anchors are used, so results with different anchor sets
    are comparable. `anchors` is accepted for backward compatibility and not needed.
    """
    target = np.flatnonzero(ld.roles != "anchor")
    m = recovery(res.b[target], ld.true_b[target])
    true_a = float(ld.true_theta[ld.groups == 0].mean())
    true_b = float(ld.true_theta[ld.groups == 1].mean())
    diff = float(res.group_means[1] - res.group_means[0])
    return dict(design=SHORT[res.design], items_compared=m["n"], bias=m["bias"], rmse=m["rmse"], max_abs=m["max_abs"],
                mean_A=float(res.group_means[0]), true_mean_A=true_a, mean_B=float(res.group_means[1]),
                true_mean_B=true_b, mean_difference=diff, true_difference=true_b - true_a,
                calibration_runs=len(res.runs), seconds=res.seconds)


def monte_carlo(method, base: LinkingData, replications=20, anchors=None, single_population=False, time_limit=180.0,
                **kw):
    """Hold the items (and bank values) fixed, redraw persons and responses; summarise each design."""
    s = dict(base.settings)
    started = time.perf_counter()
    rows = []
    done = 0
    for r in range(replications):
        if time.perf_counter() - started > time_limit:
            break
        s_r = dict(s, seed=s["seed"] + 1000 + r)
        ld = simulate_forms(**s_r, true_b=base.true_b)
        ld.bank = base.bank.copy()
        for d, res in run_all(method, ld, anchors, single_population, compute_se=False, **kw).items():
            e = evaluate(ld, res, anchors)
            rows.append(dict(replication=r + 1, design=e["design"], rmse=e["rmse"], bias=e["bias"],
                             diff_error=e["mean_difference"] - e["true_difference"],
                             mean_B_error=e["mean_B"] - e["true_mean_B"]))
        done += 1
    return rows, done


# ---------------------------------------------------------------------------------------------
# Reference checks for the individual calibration runs of a design
# ---------------------------------------------------------------------------------------------
def reference_rows(method, ld: LinkingData, res: DesignResult, jmle_settings=None, mml_settings=None):
    """Run the R reference for every calibration in a design and compare native estimates.

    JMLE -> TAM::tam.jml (xsi.fixed for anchors); MML -> TAM::tam.mml (group, xsi.fixed);
    CMLE -> eRm::RM for free runs. eRm cannot fix item parameters, so fixed-anchor CMLE runs are
    reported as "No R reference" (they are covered by the internal checks in the test suite).
    """
    from dataclasses import asdict

    from r_bridge.bridge import run_reference
    from .validation import TOLERANCES, max_abs

    rows = []
    for run in res.runs:
        X, ids = run.X, run.column_ids
        fixed = np.zeros(X.shape[1], bool) if run.fixed_b is None else np.isfinite(run.fixed_b)
        extra = {}
        if fixed.any():
            extra = dict(fixed_index=(np.flatnonzero(fixed) + 1).tolist(), fixed_value=run.fixed_b[fixed].tolist())
        base = dict(design=SHORT[res.design], run=run.name)
        if method == "CMLE" and fixed.any():
            rows.append(dict(base, check="Item difficulties (free items)", reference="eRm cannot fix items",
                             max_abs_difference=np.nan, tolerance=np.nan, status="No R reference"))
            continue
        if method == "JMLE":
            settings = dict(asdict(jmle_settings or JMLESettings()), **extra)
            pkg = "TAM tam.jml"
        elif method == "MML":
            settings = dict(asdict(mml_settings or MMLSettings()), **extra)
            if run.groups is not None:
                settings["groups"] = (np.asarray(run.groups) + 1).tolist()
            pkg = "TAM tam.mml"
        else:
            settings, pkg = {}, "eRm RM"
        ref = run_reference(method, X, settings, ids)
        if ref["status"] != "Executed":
            rows.append(dict(base, check="Reference", reference=pkg, max_abs_difference=np.nan, tolerance=np.nan,
                             status=ref["status"], reason=ref.get("reason", "")))
            continue
        free = ~fixed
        diff = max_abs(run.b[free], ref["b"][free])
        rows.append(dict(base, check="Item difficulties (free items)", reference=pkg, max_abs_difference=diff,
                         tolerance=TOLERANCES["b"], status="Pass" if diff <= TOLERANCES["b"] else "Fail"))
        if fixed.any():
            diff = max_abs(run.b[fixed], ref["b"][fixed])
            rows.append(dict(base, check="Anchors held at bank values", reference=pkg, max_abs_difference=diff,
                             tolerance=1e-8, status="Pass" if diff <= 1e-8 else "Fail"))
        if method == "JMLE" and ref.get("theta") is not None:
            diff = max_abs(run.theta, ref["theta"])
            rows.append(dict(base, check="Person abilities", reference=pkg, max_abs_difference=diff,
                             tolerance=TOLERANCES["theta"], status="Pass" if diff <= TOLERANCES["theta"] else "Fail"))
        if method == "MML" and ref.get("group_means") is not None:
            diff = max_abs(run.group_means, ref["group_means"])
            rows.append(dict(base, check="Group means", reference=pkg, max_abs_difference=diff,
                             tolerance=TOLERANCES["variance"], status="Pass" if diff <= TOLERANCES["variance"] else "Fail"))
            diff = max_abs(run.group_sds ** 2, ref["group_variances"])
            rows.append(dict(base, check="Group variances", reference=pkg, max_abs_difference=diff,
                             tolerance=TOLERANCES["variance"], status="Pass" if diff <= TOLERANCES["variance"] else "Fail"))
    return rows
