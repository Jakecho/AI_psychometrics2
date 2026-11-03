"""M-step updates for the latent ability distribution on a fixed grid.

Input to every update: expected node masses n_q = Σ_{i∈g} w_iq (posterior counts) for one distribution unit
(a group, or all groups pooled). The prior part of the expected complete-data objective is

    Q_prior(π) = Σ_q n_q log π_q.

* unconstrained empirical : π_q = n_q / N                                       (exact maximiser)
* constrained empirical   : max Q_prior s.t. Σπ = 1, Σπθ = μ0 [, Σπ(θ−μ0)² = σ0²]  (exact maximiser, dual Newton)
* normal, exact           : max over (μ, σ) of Σ n_q log π_q(μ, σ) with π_q ∝ φ((θ_q−μ)/σ) on the fixed grid
* normal, moments         : μ, σ = posterior moments (TAM / irtQ style; exact only for a continuous normal)
* Woods rescaling         : irtQ EmpHist step: π = n/N, then shift/stretch the histogram to μ0, σ0 by
                            interpolation with exponential tails (a generalised step, not an exact M-step)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .quadrature import moments, normal_weights

PRIOR_TYPES = ("fixed_normal", "estimated_normal", "fixed_empirical", "estimated_empirical")
PRIOR_NAMES = dict(fixed_normal="A · Fixed normal", estimated_normal="B · Estimated normal",
                   fixed_empirical="C · Fixed empirical", estimated_empirical="D · Estimated empirical")


def objective(n, w):
    w = np.asarray(w, float)
    n = np.asarray(n, float)
    pos = n > 0
    if np.any(w[pos] <= 0):
        return -np.inf
    return float(np.sum(n[pos] * np.log(w[pos])))


def empirical_unconstrained(n):
    n = np.asarray(n, float)
    return n / n.sum()


def empirical_constrained(n, nodes, mean=0.0, sd=None, tol=1e-12, max_iter=200):
    """Maximise Σ n_q log π_q subject to Σπ_q = 1, Σπ_q θ_q = mean and (if sd is given) Σπ_q (θ_q − mean)² = sd².

    Lagrange conditions give π_q = p_q / (1 + λᵀ g_q) with p_q = n_q/N and g_q = (θ_q − μ0, (θ_q − μ0)² − σ0²);
    λ minimises the convex dual  −Σ p_q log(1 + λᵀ g_q). Newton steps with backtracking keep 1 + λᵀg_q > 0.
    Returns (weights, info) where info reports λ, iterations and the largest constraint violation."""
    p = empirical_unconstrained(n)
    x = np.asarray(nodes, float)
    g = [x - mean]
    if sd is not None:
        g.append((x - mean) ** 2 - sd ** 2)
    g = np.stack(g, axis=1)                                   # (Q, m)
    support = p > 0
    gs, ps = g[support], p[support]
    if not (gs.min(axis=0) < 0).all() or not (gs.max(axis=0) > 0).all():
        raise ValueError("The requested moments are outside what this grid can represent "
                         "(widen the grid or relax the constraint).")
    lam = np.zeros(g.shape[1])

    def dual(l):
        t = 1 + gs @ l
        return np.inf if np.any(t <= 0) else -float(ps @ np.log(t))

    it = 0
    for it in range(1, max_iter + 1):
        t = 1 + gs @ lam
        grad = -(ps / t) @ gs
        if np.max(np.abs(grad)) < tol:
            break
        H = (gs * (ps / t ** 2)[:, None]).T @ gs
        step = np.linalg.solve(H + 1e-14 * np.eye(len(lam)), -grad)
        f0, s = dual(lam), 1.0
        while dual(lam + s * step) > f0 + 1e-4 * s * grad @ step and s > 1e-12:
            s *= 0.5
        lam = lam + s * step
    w = np.zeros_like(p)
    w[support] = ps / (1 + gs @ lam)
    w = w / w.sum()
    m, s_ = moments(x, w)
    viol = abs(m - mean) if sd is None else max(abs(m - mean), abs(s_ ** 2 - sd ** 2))
    return w, dict(method="Constrained empirical (exact M-step, dual Newton)", lagrange=lam.tolist(),
                   iterations=it, violation=float(viol))


def normal_exact(n, nodes, mean=None, sd=None, start=(0.0, 1.0)):
    """Exact M-step for a discretised normal on a fixed grid: maximise Σ n_q log π_q(μ, σ).

    Pass mean (and/or sd) to hold that moment fixed (identification). Returns (weights, μ, σ, info)."""
    n = np.asarray(n, float)
    x = np.asarray(nodes, float)
    free_mu, free_sd = mean is None, sd is None
    m0, s0 = moments(x, empirical_unconstrained(n))
    init = []
    if free_mu:
        init.append(m0 if np.isfinite(m0) else start[0])
    if free_sd:
        init.append(np.log(max(s0, 0.05)))

    def unpack(v):
        k = 0
        mu = mean
        if free_mu:
            mu, k = v[0], 1
        s = sd if not free_sd else np.exp(v[k])
        return mu, s

    def f(v):
        mu, s = unpack(v)
        return -objective(n, normal_weights(x, mu, s))

    if not init:
        w = normal_weights(x, mean, sd)
        return w, mean, sd, dict(method="Both moments fixed: nothing to estimate", success=True)
    res = minimize(f, np.array(init), method="L-BFGS-B",
                   bounds=([(x[0], x[-1])] if free_mu else []) + ([(np.log(0.02), np.log(20))] if free_sd else []),
                   options=dict(ftol=1e-14, gtol=1e-9, maxiter=200))
    mu, s = unpack(res.x)
    return normal_weights(x, mu, s), float(mu), float(s), dict(method="Discretised normal (exact M-step)",
                                                               success=bool(res.success), message=str(res.message),
                                                               iterations=int(res.nit))


def normal_moments(n, nodes, mean=None, sd=None):
    """Moment-matching update (TAM / irtQ style): μ, σ = posterior moments, fixed moments kept."""
    x = np.asarray(nodes, float)
    m, s = moments(x, empirical_unconstrained(n))
    mu = m if mean is None else mean
    if sd is None:
        s = float(np.sqrt(max(empirical_unconstrained(n) @ (x - mu) ** 2, 1e-8)))
    else:
        s = sd
    return normal_weights(x, mu, s), float(mu), float(s), dict(method="Posterior moments (generalised step)",
                                                              success=True)


def woods_rescale(nodes, weights, target_mean, target_sd):
    """irtQ EmpHist rescaling (Woods 2007): standardise the histogram to target moments by interpolation,
    with log-linear (exponential) tails outside the transformed grid. Port of irtQ scale_prior.R logic."""
    x = np.asarray(nodes, float)
    mu, sd = moments(x, weights)
    if sd < 1e-12:
        raise ValueError("Histogram collapsed to a point; widen the grid.")
    transformed = (x - mu) / sd * target_sd + target_mean
    freq = np.maximum(np.asarray(weights, float), 1e-20)
    delta = transformed[1] - transformed[0]
    out = np.interp(x, transformed, freq)
    left, right = x <= transformed[0], x >= transformed[-1]
    out[left] = np.exp(np.clip(np.log(freq[0]) + (transformed[0] - x[left]) / delta * np.log(freq[0] / freq[1]),
                               -745, 700))
    out[right] = np.exp(np.clip(np.log(freq[-1]) + (x[right] - transformed[-1]) / delta
                                * np.log(freq[-1] / freq[-2]), -745, 700))
    return out / out.sum()
