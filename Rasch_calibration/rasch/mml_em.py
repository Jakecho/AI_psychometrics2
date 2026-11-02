"""Educational marginal maximum likelihood via EM (MML-EM) for the Rasch model.

theta ~ N(0, sigma^2) is integrated out on a fixed quadrature grid x_1..x_Q with
normalised prior weights w_q (proportional to the normal density at each node).

E-step   tau_iq = w_q L_i(x_q | b) / sum_h w_h L_i(x_h | b)            (log-sum-exp)
         N_jq = sum_i m_ij tau_iq,   R_jq = sum_i m_ij x_ij tau_iq      (expected counts)
M-step   for each item solve  sum_q [R_jq - N_jq P_j(x_q; b_j)] = 0     (Newton, safeguarded)
         sigma^2 = sum_i sum_q tau_iq x_q^2 / N   (mean fixed at 0)  — if variance is estimated
Monitor  observed marginal log-likelihood  sum_i log sum_q w_q L_i(x_q | b).

Identification: the ability mean is fixed at 0 (TAM::tam.mml constraint = "cases"), so no
item constraint is added. Defaults mirror tam.mml: 21 nodes on [-6, 6], variance estimated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from scipy.special import expit, logsumexp

from .model import observed_values


@dataclass
class MMLSettings:
    n_nodes: int = 21
    extent: float = 6.0
    estimate_variance: bool = True
    fixed_sd: float = 1.0
    tol: float = 1e-4
    max_iter: int = 1000
    newton_steps: int = 20
    max_step: float = 1.0

    def validate(self):
        if self.n_nodes < 5 or self.extent <= 0:
            raise ValueError("Use at least 5 quadrature nodes and a positive grid range.")
        if self.fixed_sd <= 0 or self.tol <= 0 or self.max_iter < 1:
            raise ValueError("SD, tolerance and iteration limit must be positive.")

    @property
    def nodes(self):
        return np.linspace(-self.extent, self.extent, self.n_nodes)


def prior_weights(nodes, sd):
    w = np.exp(-0.5 * (nodes / sd) ** 2)
    return w / w.sum()


def node_loglik(x, mask, b, nodes):
    """log L_i(x_q | b) for every person i and node q: (N, Q)."""
    eta = nodes[:, None] - b[None, :]          # (Q, J)
    log_p = -np.logaddexp(0.0, -eta)
    log_q = -np.logaddexp(0.0, eta)
    return x @ log_p.T + (mask - x) @ log_q.T


def e_step(x, mask, b, nodes, sd):
    w = prior_weights(nodes, sd)
    ll = node_loglik(x, mask, b, nodes)
    joint = ll + np.log(w)[None, :]
    marg = logsumexp(joint, axis=1)
    tau = np.exp(joint - marg[:, None])
    return tau, float(marg.sum()), ll, w


def marginal_loglik(X, b, sd, nodes):
    x, mask = observed_values(X)
    return e_step(x, mask.astype(float), np.asarray(b, float), nodes, sd)[1]


def loglik_tam_convention(X, b, sd, nodes):
    """Marginal log-likelihood with weights = normal density x node spacing (as TAM::tam.mml reports it).

    These weights sum to less than 1 when the grid truncates the ability distribution; the difference from
    the renormalised value is N log(sum of weights). Parameter estimates are unaffected: the constant cancels
    in the posterior weights.
    """
    x, mask = observed_values(X)
    dens = np.exp(-0.5 * (nodes / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) * (nodes[1] - nodes[0])
    ll = node_loglik(x, mask.astype(float), np.asarray(b, float), nodes)
    return float(logsumexp(ll + np.log(dens)[None, :], axis=1).sum())


def m_step_items(N, R, b, nodes, steps=20, max_step=1.0):
    """Safeguarded Newton for each item: root of sum_q (N_q P_q - R_q) in b (vectorised)."""
    trace = []
    for _ in range(steps):
        p = expit(nodes[None, :] - b[:, None])              # (J, Q)
        score = (N * p - R).sum(axis=1)                    # d Q_j / d b_j
        info = (N * p * (1 - p)).sum(axis=1)
        delta = np.clip(score / info, -max_step, max_step)
        trace.append(dict(b=b.copy(), score=score.copy(), info=info.copy(), delta=delta.copy()))
        b = b + delta
        if np.max(np.abs(delta)) < 1e-10:
            break
    return b, trace


@dataclass
class MMLResult:
    settings: MMLSettings
    b: np.ndarray
    sd: float
    se_b: np.ndarray               # inverse observed information (accounts for unknown abilities)
    se_b_complete: np.ndarray      # complete-data information, 1/sqrt(sum_q N_jq P(1-P)) — TAM's se.xsi
    se_variance: float
    loglik: float                  # weights renormalised to sum to 1 on the grid
    loglik_tam: float              # TAM convention: weights = density x node spacing (tail mass beyond the grid lost)
    converged: bool
    status: str
    iterations: int
    history: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def variance(self):
        return self.sd ** 2


def mml_em(X, settings: MMLSettings | None = None, callback=None):
    s = settings or MMLSettings()
    s.validate()
    started = time.perf_counter()
    x, mask = observed_values(X)
    mask = mask.astype(float)
    if (mask.sum(axis=1) == 0).any():
        raise ValueError("Remove persons with no observed responses.")
    nodes = s.nodes
    p0 = np.clip(x.sum(axis=0) / mask.sum(axis=0), 1e-3, 1 - 1e-3)
    b = -np.log(p0 / (1 - p0))
    sd = s.fixed_sd
    history = []
    converged, status = False, "Iteration limit reached"
    for it in range(1, s.max_iter + 1):
        b_in, sd_in = b.copy(), sd
        tau, ll_in, _, _ = e_step(x, mask, b, nodes, sd)
        N = mask.T @ tau
        R = x.T @ tau
        b, trace = m_step_items(N, R, b, nodes, s.newton_steps, s.max_step)
        if s.estimate_variance:
            sd = float(np.sqrt((tau * nodes[None, :] ** 2).sum() / x.shape[0]))
        ll_out = marginal_loglik(X, b, sd, nodes)
        d_b = float(np.max(np.abs(b - b_in)))
        d_sd = abs(sd - sd_in)
        rec = dict(iteration=it, b_in=b_in, sd_in=sd_in, b=b.copy(), sd=sd, loglik_in=ll_in, loglik=ll_out,
                   max_b_change=d_b, sd_change=d_sd, newton_trace=trace)
        history.append(rec)
        if callback:
            callback(rec)
        if max(d_b, d_sd) < s.tol:
            converged, status = True, "Converged: max |change| in b and SD below tolerance"
            break
    se_b, se_var = _observed_information_se(X, b, sd, s)
    tau = e_step(x, mask, b, nodes, sd)[0]
    p_end = expit(nodes[None, :] - b[:, None])
    se_complete = 1 / np.sqrt(((mask.T @ tau) * p_end * (1 - p_end)).sum(axis=1))
    return MMLResult(settings=s, b=b, sd=sd, se_b=se_b, se_b_complete=se_complete, se_variance=se_var,
                     loglik=marginal_loglik(X, b, sd, nodes), loglik_tam=loglik_tam_convention(X, b, sd, nodes),
                     converged=converged, status=status,
                     iterations=len(history), history=history, seconds=time.perf_counter() - started)


def _gradient(x, mask, b, log_sd, nodes, estimate_variance):
    """Exact marginal score in b (Fisher identity) and a central difference in log SD."""
    sd = np.exp(log_sd)
    tau, _, _, _ = e_step(x, mask, b, nodes, sd)
    N, R = mask.T @ tau, x.T @ tau
    p = expit(nodes[None, :] - b[:, None])
    g_b = (N * p - R).sum(axis=1)
    if not estimate_variance:
        return g_b
    h = 1e-5
    up = e_step(x, mask, b, nodes, np.exp(log_sd + h))[1]
    down = e_step(x, mask, b, nodes, np.exp(log_sd - h))[1]
    return np.r_[g_b, (up - down) / (2 * h)]


def _observed_information_se(X, b, sd, s: MMLSettings):
    """SEs from the inverse observed information (finite differences of the score)."""
    x, mask = observed_values(X)
    mask = mask.astype(float)
    theta = np.r_[b, np.log(sd)] if s.estimate_variance else b.copy()
    k = len(theta)
    H = np.zeros((k, k))
    h = 1e-4
    for j in range(k):
        e = np.zeros(k)
        e[j] = h
        up = _gradient(x, mask, (theta + e)[:len(b)], (theta + e)[-1] if s.estimate_variance else np.log(sd), s.nodes, s.estimate_variance)
        dn = _gradient(x, mask, (theta - e)[:len(b)], (theta - e)[-1] if s.estimate_variance else np.log(sd), s.nodes, s.estimate_variance)
        H[:, j] = (up - dn) / (2 * h)
    H = (H + H.T) / 2
    try:
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return np.full(len(b), np.nan), np.nan
    se_b = se[:len(b)]
    # Delta method from log SD to variance: d(sigma^2)/d(log sigma) = 2 sigma^2.
    se_var = float(se[-1] * 2 * sd ** 2) if s.estimate_variance else np.nan
    return se_b, se_var


def estep_walkthrough(X, result: MMLResult, iteration: int, person: int, item: int):
    """Recompute one recorded E-step for a selected person and the M-step for one item."""
    rec = result.history[iteration - 1]
    x, mask = observed_values(X)
    mask = mask.astype(float)
    nodes = result.settings.nodes
    tau, _, ll, w = e_step(x, mask, rec["b_in"], nodes, rec["sd_in"])
    unnorm_log = ll[person] + np.log(w)
    table = dict(node=nodes, prior_weight=w, likelihood=np.exp(ll[person]),
                 unnormalised=np.exp(unnorm_log), posterior=tau[person])
    N = mask.T @ tau
    R = x.T @ tau
    m_trace = [dict(step=k + 1, b=t["b"][item], score=t["score"][item], information=t["info"][item],
                    update=t["delta"][item]) for k, t in enumerate(rec["newton_trace"])]
    return dict(rec=rec, table=table, posterior_sum=float(tau[person].sum()),
                N_item=N[item], R_item=R[item], m_trace=m_trace,
                eap=float(tau[person] @ nodes))


# ---------------------------------------------------------------------------------------------
# Multiple groups and fixed (anchor) items — used by the linking designs.
# ---------------------------------------------------------------------------------------------
@dataclass
class GroupMMLResult:
    """MML-EM with one normal ability distribution per group and optional fixed item difficulties.

    se_b is the complete-data SE 1/sqrt(sum_q N_jq P(1-P)) (TAM's se.xsi); NaN for fixed items.
    displacement[j] is the Newton step a fixed item would take if it were freed (NaN for free items).
    """
    settings: MMLSettings
    b: np.ndarray
    se_b: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    group_labels: list
    fixed: np.ndarray
    displacement: np.ndarray
    eap: np.ndarray
    loglik: float
    converged: bool
    status: str
    iterations: int
    history: list = field(default_factory=list)
    seconds: float = 0.0


def group_log_prior(nodes, mean, sd):
    lw = -0.5 * ((nodes - mean) / sd) ** 2
    return lw - logsumexp(lw)


def mml_em_groups(X, groups=None, fixed_b=None, settings: MMLSettings | None = None, callback=None):
    """EM for the Rasch model with group-specific N(mu_g, sigma_g^2) and optional anchors.

    Identification: without fixed items the first group's mean is fixed at 0 (TAM, group = ...);
    with fixed items every group mean is free, because the anchors carry the scale.
    E-step uses each person's own group prior; M-step updates the free items (Newton) and, per
    group, mu_g = mean posterior mean and sigma_g^2 = mean posterior E[(theta - mu_g)^2].
    """
    s = settings or MMLSettings()
    s.validate()
    started = time.perf_counter()
    x, mask = observed_values(X)
    mask = mask.astype(float)
    n, J = x.shape
    if (mask.sum(axis=1) == 0).any():
        raise ValueError("Remove persons with no observed responses.")
    if (mask.sum(axis=0) == 0).any():
        raise ValueError("Remove items with no observed responses.")
    labels, g = np.unique(np.zeros(n, int) if groups is None else np.asarray(groups), return_inverse=True)
    G = len(labels)
    onehot = np.eye(G)[g]                                  # (N, G)
    n_g = onehot.sum(axis=0)
    fixed = np.zeros(J, bool) if fixed_b is None else np.isfinite(np.asarray(fixed_b, float))
    if fixed.all():
        raise ValueError("At least one item must be free.")
    free = ~fixed
    nodes = s.nodes
    p0 = np.clip(x.sum(axis=0) / mask.sum(axis=0), 1e-3, 1 - 1e-3)
    b = -np.log(p0 / (1 - p0))
    free_mean = np.ones(G, bool) if fixed.any() else np.r_[False, np.ones(G - 1, bool)]
    if fixed.any():
        b = b + np.mean(np.asarray(fixed_b, float)[fixed] - b[fixed])
        b[fixed] = np.asarray(fixed_b, float)[fixed]
    means = np.zeros(G)
    sds = np.full(G, s.fixed_sd)
    history = []
    converged, status = False, "Iteration limit reached"

    def posterior(b, means, sds):
        logw = np.stack([group_log_prior(nodes, m, sd) for m, sd in zip(means, sds)])   # (G, Q)
        joint = node_loglik(x, mask, b, nodes) + logw[g]
        marg = logsumexp(joint, axis=1)
        return np.exp(joint - marg[:, None]), float(marg.sum())

    for it in range(1, s.max_iter + 1):
        b_in, means_in, sds_in = b.copy(), means.copy(), sds.copy()
        tau, ll_in = posterior(b, means, sds)
        N, R = mask.T @ tau, x.T @ tau
        b_free, trace = m_step_items(N[free], R[free], b[free], nodes, s.newton_steps, s.max_step)
        b[free] = b_free
        post_mean = tau @ nodes
        means = np.where(free_mean, (onehot.T @ post_mean) / n_g, 0.0)
        if s.estimate_variance:
            second = tau @ nodes ** 2
            var = (onehot.T @ (second - 2 * post_mean * means[g] + means[g] ** 2)) / n_g
            sds = np.sqrt(var)
        change = max(float(np.max(np.abs(b - b_in))), float(np.max(np.abs(means - means_in))),
                     float(np.max(np.abs(sds - sds_in))))
        rec = dict(iteration=it, b_in=b_in, b=b.copy(), means=means.copy(), sds=sds.copy(), loglik_in=ll_in,
                   max_change=change)
        history.append(rec)
        if callback:
            callback(rec)
        if change < s.tol:
            converged, status = True, "Converged: max |change| in b, means and SDs below tolerance"
            break
    tau, ll = posterior(b, means, sds)
    N, R = mask.T @ tau, x.T @ tau
    p = expit(nodes[None, :] - b[:, None])
    info = (N * p * (1 - p)).sum(axis=1)
    se_b = np.where(free, 1 / np.sqrt(info), np.nan)
    disp = np.where(fixed, (N * p - R).sum(axis=1) / info, np.nan)
    return GroupMMLResult(settings=s, b=b, se_b=se_b, means=means, sds=sds, group_labels=list(labels), fixed=fixed,
                          displacement=disp, eap=tau @ nodes, loglik=ll, converged=converged, status=status,
                          iterations=len(history), history=history, seconds=time.perf_counter() - started)
