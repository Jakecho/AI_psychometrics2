"""Dichotomous Rasch model: P(X=1 | theta, b) = exp(theta - b) / (1 + exp(theta - b)).

All functions accept a response matrix X (persons x items) with np.nan for missing
responses. Missing responses are excluded through an explicit observation mask; they are
never treated as incorrect answers.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def probability(theta, b):
    """P(X_ij = 1) for every person/item pair (shape: len(theta) x len(b))."""
    theta = np.asarray(theta, float)
    b = np.asarray(b, float)
    return expit(np.subtract.outer(theta, b))


def information(theta, b):
    """Item information P(1 - P) at each (theta, b) pair."""
    p = probability(theta, b)
    return p * (1.0 - p)


def test_information(theta, b):
    """Sum of item information over items, for each theta."""
    return information(theta, b).sum(axis=-1)


def observed_mask(X):
    return ~np.isnan(np.asarray(X, float))


def observed_values(X):
    """Responses with missing entries set to 0, plus the observation mask."""
    X = np.asarray(X, float)
    mask = ~np.isnan(X)
    return np.where(mask, X, 0.0), mask


def joint_loglik(X, theta, b):
    """Joint log-likelihood over observed responses only (stable log-sigmoid form)."""
    x, mask = observed_values(X)
    eta = np.subtract.outer(np.asarray(theta, float), np.asarray(b, float))
    # x*eta - log(1 + exp(eta)) is the Bernoulli log-likelihood on the logit scale.
    ll = x * eta - np.logaddexp(0.0, eta)
    return float(np.sum(np.where(mask, ll, 0.0)))


def raw_scores(X):
    x, mask = observed_values(X)
    return x.sum(axis=1), mask.sum(axis=1)


def item_scores(X):
    x, mask = observed_values(X)
    return x.sum(axis=0), mask.sum(axis=0)
