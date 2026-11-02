"""Agreement checks between the educational Python estimators and the R references.

Tolerances are absolute and were set from executed baselines (see docs/VALIDATION.md):
the observed agreement there was 1e-5 or better for estimates, so 1e-3 leaves room for
different optimisers and stopping rules without hiding real errors.
"""
from __future__ import annotations

import numpy as np

TOLERANCES = dict(b=1e-3, se_b=1e-3, theta=1e-3, variance=1e-3, loglik_per_person=1e-5)


def _row(check, value, tol):
    ok = np.isfinite(value) and value <= tol
    return dict(check=check, max_abs_difference=float(value), tolerance=tol, status="Pass" if ok else "Fail")


def max_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else np.nan


def compare(method, python, reference, n_persons=None):
    """Rows of (check, difference, tolerance, status) for one estimator."""
    if reference.get("status") != "Executed":
        return [dict(check=f"{method} reference", max_abs_difference=np.nan, tolerance=np.nan,
                     status=reference.get("status", "Not run"))]
    rows = [_row("Item difficulties b", max_abs(python["b"], reference["b"]), TOLERANCES["b"])]
    if python.get("se_b") is not None and reference.get("se_b") is not None:
        rows.append(_row("Item standard errors" + python.get("se_label", ""), max_abs(python["se_b"], reference["se_b"]),
                         TOLERANCES["se_b"]))
    if "theta" in python and reference.get("theta") is not None:
        rows.append(_row("Person abilities", max_abs(python["theta"], reference["theta"]), TOLERANCES["theta"]))
    if "variance" in python and reference.get("variance") is not None:
        rows.append(_row("Ability variance", abs(python["variance"] - reference["variance"]), TOLERANCES["variance"]))
    if "loglik" in python and reference.get("loglik") is not None and n_persons:
        rows.append(_row("Marginal log-likelihood per person (TAM convention)", abs(python["loglik"] - reference["loglik"]) / n_persons,
                         TOLERANCES["loglik_per_person"]))
    if "cond_loglik" in python and reference.get("cond_loglik") is not None and n_persons:
        rows.append(_row("Conditional log-likelihood per person", abs(python["cond_loglik"] - reference["cond_loglik"]) / n_persons,
                         TOLERANCES["loglik_per_person"]))
    return rows
