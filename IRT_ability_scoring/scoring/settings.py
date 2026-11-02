"""Scoring settings shared by every method (one object, so all pages use identical choices).

Defaults follow irtQ::est_score(), the primary reference: normal prior N(0, 1), 41 Gauss–Hermite
nodes for EAP / EAPsum (statmod::gauss.quad.prob), search range θ ∈ [−5, 5], steps capped at 1.
The convergence tolerance is tighter than irtQ's 1e-4 so the educational tables show settled values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoringSettings:
    prior_mean: float = 0.0        # normal prior g(θ) for EAP, MAP, EAPsum
    prior_sd: float = 1.0
    quad_rule: str = "gauss-hermite"  # "gauss-hermite" (irtQ) or "rectangular" (equally spaced, mirt)
    n_nodes: int = 41               # irtQ default (mirt uses 61 rectangular nodes on [−6, 6])
    theta_lim: float = 6.0          # rectangular grid on [−lim, lim]
    bound: float = 5.0              # search range for MLE / MAP / WLE / TCC: θ ∈ [−bound, bound] (irtQ: range)
    tol: float = 1e-6               # convergence: |Δθ| < tol
    max_iter: int = 100
    max_step: float = 1.0           # Newton steps are capped at this length (logits)
    start: float = 0.0              # starting value for iterative methods

    def as_dict(self):
        return asdict(self)

    def with_(self, **kw):
        d = asdict(self)
        d.update(kw)
        return ScoringSettings(**d)
