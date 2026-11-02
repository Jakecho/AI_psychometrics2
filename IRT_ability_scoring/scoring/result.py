"""Common result container for every scoring method."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScoreResult:
    method: str
    theta: np.ndarray                 # point estimate (±inf where no finite estimate exists)
    se: np.ndarray                    # method-appropriate uncertainty (SE or posterior SD)
    status: np.ndarray                # per person, plain words
    iterations: np.ndarray            # 0 for non-iterative methods
    uncertainty_label: str = "SE"
    extra: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n(self):
        return len(self.theta)

    @property
    def finite(self):
        return np.isfinite(self.theta)

    def person(self, i):
        out = dict(method=self.method, theta=float(self.theta[i]), uncertainty=float(self.se[i]),
                   uncertainty_label=self.uncertainty_label, status=str(self.status[i]),
                   iterations=int(self.iterations[i]))
        for k, v in self.extra.items():
            if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == self.n:
                out[k] = v[i].item() if hasattr(v[i], "item") else v[i]
        return out


def skip_empty(fn):
    """Score only examinees with at least one observed response; the others get NaN and a status."""
    import functools

    @functools.wraps(fn)
    def wrapper(items, Y, settings, *args, start=None, **kw):
        Y = np.atleast_2d(np.asarray(Y, float))
        has = (~np.isnan(Y)).any(1)
        if has.all():
            return fn(items, Y, settings, *args, start=start, **kw) if "start" in fn.__code__.co_varnames else \
                fn(items, Y, settings, *args, **kw)
        N = len(has)
        if start is not None:
            start = np.broadcast_to(np.asarray(start, float), (N,))[has]
        sub = fn(items, Y[has], settings, *args, start=start, **kw) if "start" in fn.__code__.co_varnames else \
            fn(items, Y[has], settings, *args, **kw)

        def expand(a, fill):
            a = np.asarray(a)
            out = np.full(N, fill, dtype=a.dtype if a.dtype != object else object)
            out[has] = a
            return out
        rows = np.flatnonzero(has)
        for h in sub.history:
            h["idx"] = rows[h["idx"]]
        def fill_extra(v):
            if not (isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == has.sum()):
                return v
            if v.dtype.kind in "iub":
                v = v.astype(float)
            return expand(v, np.nan if v.dtype.kind == "f" else "")
        extra = {k: fill_extra(v) for k, v in sub.extra.items()}
        status = expand(sub.status.astype(object), "no observed responses")
        return ScoreResult(sub.method, expand(sub.theta, np.nan), expand(sub.se, np.nan), status,
                           expand(sub.iterations, 0), sub.uncertainty_label, extra, sub.history, sub.seconds)
    return wrapper
