"""Likelihood-side helpers: per-pattern curves over a θ grid and the global-maximum check."""
from __future__ import annotations

import numpy as np

from irt_models import DICHOTOMOUS, loglik_grid, pattern_terms


def curves(items, y, grid):
    """ℓ, ℓ', ℓ'', I, J, dI/dθ, T, T', Σ Var for one response vector over a θ grid (dict of arrays)."""
    y = np.asarray(y, float)
    grid = np.asarray(grid, float)
    return pattern_terms(items, np.tile(y, (len(grid), 1)), grid)


def nonconcave(items):
    """True if some item has a lower or upper asymptote (3PL / 4PL): ℓ(θ) need not be concave."""
    return any(it.model in DICHOTOMOUS and (it.c > 0 or it.d < 1) for it in items)


def global_check(items, Y, theta_hat, loglik_hat, bound, n_grid=801, log_prior=None):
    """Compare a Newton solution with the grid maximum of ℓ (+ log prior, for MAP) on [−bound, bound].

    Returns (messages, grid_argmax, at_edge) per person. Message is '' when the solution is the grid maximum.
    """
    grid = np.linspace(-bound, bound, n_grid)
    L = loglik_grid(items, Y, grid) + (0 if log_prior is None else log_prior(grid))
    k = L.argmax(1)
    top = L[np.arange(len(k)), k]
    msg = np.array([""] * len(k), dtype=object)
    better = (top > loglik_hat + 1e-8) & (np.abs(grid[k] - theta_hat) > 2 * (grid[1] - grid[0]))
    msg[better] = [f"local maximum only: ℓ is higher near θ = {grid[j]:.2f}" for j in k[better]]
    edge = (k == 0) | (k == n_grid - 1)
    msg[edge & better] = [f"ℓ is highest at the search bound θ = {grid[j]:.0f}: no finite maximum in range"
                          for j in k[edge & better]]
    return msg, grid[k], edge
