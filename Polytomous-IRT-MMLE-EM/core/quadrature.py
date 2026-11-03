"""Quadrature grids and discrete ability distributions.

Three grid kinds:
* "gauss_hermite"  nodes μ + σ x_q, weights from probabilists' Gauss–Hermite rule (moves with μ, σ)
* "equal"          evenly spaced nodes on [−bound, bound] (irtQ's Quadrature = c(Q, bound) is this kind,
                   NOT Gauss–Hermite); weights are the normalised normal density at the nodes
* "custom"         user-supplied nodes (e.g. a historical grid), weights supplied separately
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm

GRID_KINDS = ("gauss_hermite", "equal", "custom")
GRID_NAMES = dict(gauss_hermite="Gauss–Hermite", equal="Equally spaced", custom="Custom fixed grid")


@dataclass(frozen=True)
class Grid:
    kind: str
    nodes: np.ndarray
    base_weights: np.ndarray      # weights implied by the grid itself (GH weights or N(μ, σ²) at nodes)

    @property
    def Q(self):
        return len(self.nodes)


def gauss_hermite(n, mean=0.0, sd=1.0):
    x, w = hermegauss(int(n))
    return mean + sd * x, w / w.sum()


def normal_weights(nodes, mean=0.0, sd=1.0):
    """Discretised normal: π_q ∝ φ((θ_q − μ)/σ), normalised to sum to one on the fixed grid."""
    logw = norm.logpdf(np.asarray(nodes, float), mean, sd)
    w = np.exp(logw - logw.max())
    return w / w.sum()


def make_grid(kind="equal", n=41, bound=6.0, mean=0.0, sd=1.0, nodes=None):
    if kind == "gauss_hermite":
        x, w = gauss_hermite(n, mean, sd)
        return Grid(kind, x, w)
    if kind == "equal":
        if n < 2 or bound <= 0:
            raise ValueError("An equally spaced grid needs at least two nodes and a positive bound.")
        x = np.linspace(-bound, bound, int(n))
        return Grid(kind, x, normal_weights(x, mean, sd))
    if kind == "custom":
        x = np.asarray(nodes, float)
        if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("Custom nodes must be a strictly increasing list of at least two values.")
        return Grid(kind, x, normal_weights(x, mean, sd))
    raise ValueError(f"Unknown grid kind {kind!r}.")


def moments(nodes, weights):
    """(mean, SD) of a discrete distribution; accepts (Q,) or (G, Q) weights."""
    w = np.asarray(weights, float)
    x = np.asarray(nodes, float)
    m = w @ x
    v = w @ x ** 2 - m ** 2
    return np.stack([m, np.sqrt(np.maximum(v, 0.0))], axis=-1)


def skewness(nodes, weights):
    m, s = moments(nodes, weights)
    return float(np.asarray(weights) @ ((np.asarray(nodes) - m) / max(s, 1e-12)) ** 3)


def validate_weights(w, Q=None, name="weights"):
    w = np.asarray(w, float)
    if w.ndim != 1:
        raise ValueError(f"{name} must be a list of numbers.")
    if Q is not None and len(w) != Q:
        raise ValueError(f"{name} has {len(w)} values but the grid has {Q} nodes.")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    if w.sum() <= 0:
        raise ValueError(f"{name} must have positive total mass.")
    return w / w.sum()


def illustrative_distribution(nodes, shape="normal", mean=0.0, sd=1.0):
    """Shapes used to build fixed empirical priors and generating distributions on a grid."""
    x = np.asarray(nodes, float)
    if shape == "normal":
        w = norm.pdf(x)
    elif shape == "skewed":
        from scipy.stats import skewnorm
        w = skewnorm.pdf(x, 5)
    elif shape == "bimodal":
        w = 0.5 * norm.pdf(x, -1.2, 0.6) + 0.5 * norm.pdf(x, 1.2, 0.6)
    elif shape == "uniform":
        w = np.ones_like(x)
    else:
        raise ValueError(f"Unknown shape {shape!r}.")
    w = w / w.sum()
    m, s = moments(x, w)
    # Place the shape at the requested mean/SD by interpolating the standardized density onto the grid.
    z = (x - mean) / sd
    src = (x - m) / s
    dens = np.interp(z, src, w / np.gradient(src), left=0.0, right=0.0)
    if dens.sum() <= 0:
        raise ValueError("The requested distribution lies outside the grid.")
    return dens / dens.sum()
