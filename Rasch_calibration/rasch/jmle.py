"""Educational joint maximum likelihood (JMLE) for the dichotomous Rasch model.

Each outer iteration
  1. updates every person ability by Newton steps on   sum_j (x_ij - P_ij) = 0,
  2. re-centres abilities (constraint "cases": mean theta = 0) or difficulties ("items"),
  3. updates every item difficulty by Newton steps on   sum_i (P_ij - x_ij) = 0.

Settings that mirror TAM::tam.jml (documented, switchable):
  adj = 0.3   extreme raw scores r = 0 / r = max are replaced by r + adj / r - adj, so
              all-correct and all-incorrect persons (and items) get finite estimates;
  bias=True   after convergence difficulties are multiplied by (J - 1) / J, a simple
              finite-test-length bias correction;
  constraint  "cases" (TAM default) or "items" (mean difficulty = 0).

Fixed items (anchors): `fixed_b` holds a difficulty for each fixed item and NaN for free
items. Fixed items are never updated and they set the scale, so no centring constraint is
applied. As in tam.jml(xsi.fixed = ...), the bias factor (J - 1) / J multiplies the free
items only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import joint_loglik, observed_values, probability


@dataclass
class JMLESettings:
    adj: float = 0.3
    bias: bool = True
    constraint: str = "cases"
    tol: float = 1e-4
    max_iter: int = 1000
    inner_steps: int = 10
    max_step: float = 1.0

    def validate(self):
        if self.adj < 0 or self.adj >= 0.5:
            raise ValueError("Extreme-score adjustment must be in [0, 0.5).")
        if self.constraint not in ("cases", "items"):
            raise ValueError("Constraint must be 'cases' or 'items'.")
        if self.tol <= 0 or self.max_iter < 1 or self.inner_steps < 1 or self.max_step <= 0:
            raise ValueError("Tolerance, iterations and step limits must be positive.")


@dataclass
class JMLEResult:
    settings: JMLESettings
    theta: np.ndarray
    b: np.ndarray                  # reported difficulties (bias-corrected if requested)
    b_uncorrected: np.ndarray
    se_b: np.ndarray
    se_theta: np.ndarray
    converged: bool
    iterations: int
    status: str
    adjusted_person_scores: np.ndarray
    adjusted_item_scores: np.ndarray
    history: list = field(default_factory=list)   # one dict per outer iteration (state before it)
    n_extreme_persons: int = 0


def _newton_persons(x, mask, theta, b, target, steps, max_step):
    """Newton steps on each person's score equation with fixed difficulties."""
    for _ in range(steps):
        p = probability(theta, b) * mask
        info = (p * (1 - p)).sum(axis=1)
        delta = np.clip((target - p.sum(axis=1)) / info, -max_step, max_step)
        theta = theta + delta
        if np.max(np.abs(delta)) < 1e-10:
            break
    return theta


def _newton_items(x, mask, theta, b, target, steps, max_step, free=None):
    """Newton steps on each item's score equation with fixed abilities (fixed items untouched)."""
    free = np.ones(len(b), bool) if free is None else free
    for _ in range(steps):
        p = probability(theta, b) * mask
        info = (p * (1 - p)).sum(axis=0)
        # d/db sum_i P_ij = -info, so Newton for sum_i P_ij = target gives b += (sum P - target)/info.
        delta = np.where(free, np.clip((p.sum(axis=0) - target) / info, -max_step, max_step), 0.0)
        b = b + delta
        if np.max(np.abs(delta)) < 1e-10:
            break
    return b


def jmle(X, settings: JMLESettings | None = None, callback=None, fixed_b=None):
    s = settings or JMLESettings()
    s.validate()
    fixed = _fixed_mask(fixed_b, np.shape(X)[1])
    free = ~fixed
    x, mask = observed_values(X)
    n_obs = mask.sum(axis=1)
    if (n_obs == 0).any():
        raise ValueError("Remove persons with no observed responses.")
    if (mask.sum(axis=0) == 0).any():
        raise ValueError("Remove items with no observed responses.")
    J = x.shape[1]
    r = x.sum(axis=1)
    r_adj = r.copy()
    r_adj[r == n_obs] -= s.adj
    r_adj[r == 0] += s.adj
    item_max = mask.sum(axis=0)
    t = x.sum(axis=0)
    t_adj = t.copy()
    t_adj[t == item_max] -= s.adj
    t_adj[t == 0] += s.adj
    if s.adj == 0 and ((r == 0) | (r == n_obs)).any():
        extreme = int(((r == 0) | (r == n_obs)).sum())
        raise ValueError(f"{extreme} person(s) have zero or perfect scores: their unadjusted MLE is infinite. "
                         "Use an extreme-score adjustment (e.g. 0.3) or remove them.")
    if s.adj == 0 and ((t == 0) | (t == item_max)).any():
        raise ValueError("An item answered all-correct or all-incorrect has an infinite unadjusted estimate.")
    # Logit starting values (TAM uses the same form).
    theta = np.log(r_adj / (n_obs - r_adj))
    b = -np.log(t_adj / (item_max - t_adj))
    anchored = bool(fixed.any())
    if anchored:
        # Start the free items (and persons) on the anchors' scale, then insert the fixed values.
        shift = float(np.mean(np.asarray(fixed_b, float)[fixed] - b[fixed]))
        b, theta = b + shift, theta + shift
        b[fixed] = np.asarray(fixed_b, float)[fixed]
    elif s.constraint == "cases":
        theta = theta - theta.mean()
    else:
        b = b - b.mean()
    history = []
    converged = False
    status = "Iteration limit reached"
    for it in range(1, s.max_iter + 1):
        theta_old, b_old = theta.copy(), b.copy()
        theta = _newton_persons(x, mask, theta, b, r_adj, s.inner_steps, s.max_step)
        if s.constraint == "cases" and not anchored:
            theta = theta - theta.mean()
        theta_for_items = theta.copy()
        b = _newton_items(x, mask, theta, b, t_adj, s.inner_steps, s.max_step, free)
        if s.constraint == "items" and not anchored:
            shift = b.mean()
            b, theta = b - shift, theta - shift
        d_theta = float(np.max(np.abs(theta - theta_old)))
        d_b = float(np.max(np.abs(b - b_old)))
        record = dict(iteration=it, theta_in=theta_old, b_in=b_old, theta_for_items=theta_for_items,
                      theta=theta.copy(), b=b.copy(),
                      max_theta_change=d_theta, max_b_change=d_b, loglik=joint_loglik(X, theta, b))
        history.append(record)
        if callback:
            callback(record)
        if not (np.isfinite(theta).all() and np.isfinite(b).all()):
            status = "Non-finite estimates"
            break
        if max(d_theta, d_b) < s.tol:
            converged, status = True, "Converged: max |change| below tolerance"
            break
    p = probability(theta, b) * mask
    se_b = 1.0 / np.sqrt((p * (1 - p)).sum(axis=0))
    se_b[fixed] = np.nan
    se_theta = 1.0 / np.sqrt((p * (1 - p)).sum(axis=1))
    b_rep = b.copy()
    if s.bias:
        b_rep[free] = b[free] * (J - 1) / J
    return JMLEResult(settings=s, theta=theta, b=b_rep, b_uncorrected=b, se_b=se_b, se_theta=se_theta,
                      converged=converged, iterations=len(history), status=status,
                      adjusted_person_scores=r_adj, adjusted_item_scores=t_adj, history=history,
                      n_extreme_persons=int(((r == 0) | (r == n_obs)).sum()))


def _fixed_mask(fixed_b, J):
    if fixed_b is None:
        return np.zeros(J, bool)
    fixed_b = np.asarray(fixed_b, float)
    if fixed_b.shape != (J,):
        raise ValueError("fixed_b needs one entry per item (NaN = free).")
    if np.isfinite(fixed_b).all():
        raise ValueError("At least one item must be free.")
    return np.isfinite(fixed_b)


def displacement(X, theta, b):
    """One Newton step each item would take if freed: sum_i (P - x) / sum_i P(1 - P)."""
    x, mask = observed_values(X)
    p = probability(theta, b) * mask
    return (p.sum(axis=0) - x.sum(axis=0)) / (p * (1 - p)).sum(axis=0)


def walkthrough(X, result: JMLEResult, iteration: int, person: int, item: int):
    """Recompute one iteration's person and item Newton step from its recorded inputs."""
    rec = result.history[iteration - 1]
    x, mask = observed_values(X)
    th, b = rec["theta_in"], rec["b_in"]
    p_i = probability([th[person]], b)[0] * mask[person]
    score_i = result.adjusted_person_scores[person] - p_i.sum()
    info_i = float((p_i * (1 - p_i)).sum())
    step_i = float(np.clip(score_i / info_i, -result.settings.max_step, result.settings.max_step))
    th_after = rec["theta_for_items"]
    p_j = probability(th_after, [b[item]])[:, 0] * mask[:, item]
    score_j = float(p_j.sum() - result.adjusted_item_scores[item])
    info_j = float((p_j * (1 - p_j)).sum())
    step_j = float(np.clip(score_j / info_j, -result.settings.max_step, result.settings.max_step))
    return dict(rec=rec, observed_items=int(mask[person].sum()),
                person_score=float(result.adjusted_person_scores[person]), expected_person=float(p_i.sum()),
                person_residual=float(score_i), person_info=info_i, person_step=step_i,
                theta_in=float(th[person]), theta_first_step=float(th[person] + step_i), theta_out=float(rec["theta"][person]),
                item_score=float(result.adjusted_item_scores[item]), expected_item=float(p_j.sum()),
                item_residual=score_j, item_info=info_j, item_step=step_j,
                b_in=float(b[item]), b_first_step=float(b[item] + step_j), b_out=float(rec["b"][item]))
