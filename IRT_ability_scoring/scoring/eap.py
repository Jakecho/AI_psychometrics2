"""Expected a posteriori (EAP) estimation by numerical quadrature.

    θ̂_EAP = ∫ θ L(θ) g(θ) dθ / ∫ L(θ) g(θ) dθ ≈ Σ_q θ_q L(θ_q) w_q / Σ_q L(θ_q) w_q

Nodes and weights (`quadrature`):
  gauss-hermite: θ_q = μ + √2 σ x_q, w_q = h_q / √π (Gauss–Hermite x_q, h_q; equals
                 statmod::gauss.quad.prob(n, "norm"), used by irtQ). The prior is in the weights.
  rectangular:   θ_q equally spaced on [−lim, lim], w_q ∝ φ((θ_q − μ)/σ), normalised (mirt).

All products are formed on the log scale and normalised with log-sum-exp.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import integrate
from scipy.special import logsumexp
from scipy.stats import norm

from irt_models import loglik_grid

from .result import ScoreResult, skip_empty


def quadrature(settings, rule=None, n=None):
    rule = rule or settings.quad_rule
    n = int(n or settings.n_nodes)
    mu, sd = settings.prior_mean, settings.prior_sd
    if rule == "gauss-hermite":
        x, h = np.polynomial.hermite.hermgauss(n)
        return mu + np.sqrt(2) * sd * x, h / np.sqrt(np.pi)
    nodes = np.linspace(-settings.theta_lim, settings.theta_lim, n)
    dens = norm.pdf(nodes, mu, sd)
    return nodes, dens / dens.sum()


def posterior(items, Y, settings, rule=None, n=None):
    """(nodes, prior weights, log-likelihood (N,Q), normalised posterior weights (N,Q))."""
    nodes, w = quadrature(settings, rule, n)
    ll = loglik_grid(items, Y, nodes)
    lp = ll + np.log(w)
    post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    return nodes, w, ll, post


@skip_empty
def eap(items, Y, settings, rule=None, n=None):
    t0 = time.perf_counter()
    Y = np.atleast_2d(np.asarray(Y, float))
    nodes, w, ll, post = posterior(items, Y, settings, rule, n)
    m = post @ nodes
    var = post @ nodes ** 2 - m ** 2
    psd = np.sqrt(np.maximum(var, 0))
    status = np.array([f"{len(nodes)}-node {rule or settings.quad_rule} quadrature"] * len(m), dtype=object)
    return ScoreResult("EAP", m, psd, status, np.zeros(len(m), int), "posterior SD",
                       extra=dict(raw=np.nansum(Y, 1)), seconds=time.perf_counter() - t0)


def node_table(items, y, settings, rule=None, n=None):
    """Every quadrature quantity for one response vector (rows = nodes)."""
    nodes, w, ll, post = posterior(items, np.atleast_2d(y), settings, rule, n)
    ll, post = ll[0], post[0]
    log_unnorm = ll + np.log(w)
    m = float(post @ nodes)
    return dict(node=nodes, prior_weight=w, log_likelihood=ll, likelihood=np.exp(ll), log_unnormalised=log_unnorm,
                unnormalised=np.exp(log_unnorm), posterior_weight=post, theta_x_weight=nodes * post,
                sq_dev_x_weight=(nodes - m) ** 2 * post), m, float(np.sqrt(post @ (nodes - m) ** 2))


def reference_integral(items, y, settings):
    """Posterior mean and SD by adaptive quadrature (scipy.integrate.quad) on the continuous prior."""
    mu, sd = settings.prior_mean, settings.prior_sd
    y = np.atleast_2d(np.asarray(y, float))
    grid = np.linspace(mu - 10 * sd, mu + 10 * sd, 2001)
    ref = (loglik_grid(items, y, grid)[0] + norm.logpdf(grid, mu, sd)).max()

    def f(t, power):
        return t ** power * np.exp(loglik_grid(items, y, np.array([t]))[0, 0] + norm.logpdf(t, mu, sd) - ref)

    lims = (mu - 12 * sd, mu + 12 * sd)
    z = integrate.quad(f, *lims, args=(0,), limit=200, epsabs=0, epsrel=1e-12)[0]
    m1 = integrate.quad(f, *lims, args=(1,), limit=200, epsabs=0, epsrel=1e-12)[0] / z
    m2 = integrate.quad(f, *lims, args=(2,), limit=200, epsabs=0, epsrel=1e-12)[0] / z
    return m1, float(np.sqrt(max(m2 - m1 ** 2, 0)))


def convergence_table(items, y, settings, sizes=(3, 5, 7, 11, 15, 21, 31, 41, 61, 81, 121)):
    m_ref, s_ref = reference_integral(items, y, settings)
    rows = []
    for rule in ("gauss-hermite", "rectangular"):
        for n in sizes:
            r = eap(items, np.atleast_2d(y), settings, rule, n)
            rows.append(dict(rule=rule, nodes=n, EAP=float(r.theta[0]), PSD=float(r.se[0]),
                             error_EAP=float(r.theta[0] - m_ref), error_PSD=float(r.se[0] - s_ref)))
    return rows, m_ref, s_ref


def posterior_quantiles(items, Y, settings, probs=(0.025, 0.5, 0.975), n_grid=801):
    """Posterior quantiles on a fine rectangular grid (piecewise-linear CDF)."""
    mu, sd = settings.prior_mean, settings.prior_sd
    grid = np.linspace(mu - 8 * sd, mu + 8 * sd, n_grid)
    lp = loglik_grid(items, np.atleast_2d(Y), grid) + norm.logpdf(grid, mu, sd)
    post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    cdf = np.cumsum(post, axis=1) - post / 2
    return np.array([[np.interp(p, c, grid) for p in probs] for c in cdf])
