"""Compare Python estimates with reference packages under matched settings.

Matching rules (documented in docs/VALIDATION.md):
  irtQ  — same prior, nodes (Gauss–Hermite, or the rectangular grid passed as `weights`), search range
          [−bound, bound]; irtQ tolerance set to 1e-10. Its ML/WL results at the range bound and its
          INV.TCC values for extreme scores (interpolated to ±7) are counted, not compared numerically.
  mirt  — rectangular grid of n nodes on [−lim, lim] (mirt has no Gauss–Hermite option for fscores);
          ML/MAP SEs are observed-information SEs; mirt's "WLE" maximises L·√I and is compared with
          our L·√I mode. Our Warm WLE is shown against it as information (they differ only for 3PL/4PL items).
  TAM   — Rasch-family items only; WLE with SE 1/√I; MLE only for non-extreme patterns (TAM adjusts
          extreme scores by 0.3).
"""
from __future__ import annotations

import numpy as np

from scoring import FUNCTIONS
from scoring.wle import weighted_likelihood_mode, wle

from .bridge import METHOD_NAMES, run_reference

TOL = dict(irtQ=dict(theta=1e-5, se=1e-5), mirt=dict(theta=2e-3, se=2e-2), TAM=dict(theta=1e-3, se=1e-3))
EXACT = dict(EAP=1e-6, EAPsum=1e-6)


def _max(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.abs(a).max()) if len(a) else np.nan


def _row(package, method, ours, n, dtheta, dse, tol_t, tol_s, note, informational=False):
    ok_t = np.isnan(dtheta) or dtheta <= tol_t
    ok_s = np.isnan(dse) or dse <= tol_s
    result = "Information" if informational else ("Pass" if (ok_t and ok_s and n > 0) else ("No comparable values" if n == 0 else "Fail"))
    return dict(package=package, method=method, python=ours, compared=int(n), max_abs_diff_theta=dtheta,
                max_abs_diff_se=dse, tolerance_theta=tol_t, tolerance_se=tol_s, result=result, note=note)


def compare_package(package, items, Y, settings, methods=("MLE", "WLE", "MAP", "EAP", "EAPsum", "TCC")):
    Y = np.atleast_2d(np.asarray(Y, float))
    s = settings.with_(quad_rule="rectangular") if package == "mirt" else settings
    ref_settings = s.with_(tol=1e-10, max_iter=500) if package == "irtQ" else s
    methods = [m for m in methods if m in METHOD_NAMES[package]]
    ref = run_reference(package, items, Y, ref_settings, methods)
    if ref["status"] != "Executed":
        return ref, [dict(package=package, method=m, python="—", compared=0, max_abs_diff_theta=np.nan,
                          max_abs_diff_se=np.nan, tolerance_theta=np.nan, tolerance_se=np.nan, result=ref["status"],
                          note=ref["reason"]) for m in methods]
    rows = []
    tol = TOL[package]
    for m in methods:
        r = ref["results"].get(m)
        if r is None or "error" in r:
            rows.append(_row(package, m, "—", 0, np.nan, np.nan, np.nan, np.nan,
                             (r or {}).get("error", "not returned"), True))
            continue
        rt, rs = r["theta"], r["se"]
        tt = EXACT.get(m, tol["theta"])
        ts = EXACT.get(m, tol["se"])
        if package == "mirt" and m == "WLE":
            mine = weighted_likelihood_mode(items, Y, s)
            ours = "L·√I mode (mirt's definition)"
        else:
            mine = FUNCTIONS[m](items, Y, s)
            ours = m
        pt, ps = mine.theta, mine.se
        if package == "mirt" and m in ("MLE", "MAP"):
            ps = mine.extra["se_observed"]
            ours += " (observed-information SE)"
        note = ""
        both = np.isfinite(pt) & np.isfinite(rt)
        if package == "irtQ" and m in ("MLE", "WLE", "MAP"):
            at_bound = np.abs(rt) >= s.bound - 1e-9
            agree = (~np.isfinite(pt) | (np.abs(pt) >= s.bound - 1e-6)) & at_bound
            both &= ~at_bound & (np.abs(pt) < s.bound - 1e-6)
            if at_bound.any():
                note = (f"{int(at_bound.sum())} pattern(s) at irtQ's range bound (SE 99.9999); "
                        f"Python also non-finite / at bound for {int(agree.sum())}.")
        if package == "irtQ" and m == "TCC":
            ext = ~np.isfinite(pt)
            both &= ~ext
            if ext.any():
                note = f"{int(ext.sum())} score(s) with no finite TCC root; irtQ interpolates these toward ±7."
            rs = np.full_like(rs, np.nan)          # different SE definitions
            note += " SE not compared (irtQ's INV.TCC SE is a different quantity)."
        if package == "TAM" and m == "MLE":
            ext = ~np.isfinite(pt)
            both &= ~ext
            if ext.any():
                note = f"{int(ext.sum())} extreme pattern(s) skipped: TAM adjusts them by 0.3 (no finite MLE)."
        if package == "mirt" and m == "MLE":
            inf_agree = (~np.isfinite(pt)) & (~np.isfinite(rt)) & (np.sign(pt) == np.sign(rt))
            if inf_agree.any():
                note = f"{int(inf_agree.sum())} extreme pattern(s): both report ±∞."
        if package == "mirt" and m == "WLE":
            rs = np.full_like(rs, np.nan)
            note = "SE not compared: mirt uses the numerical Hessian of −(ℓ + ½ log I)."
        with np.errstate(invalid="ignore"):
            diff_t = np.where(both, pt - rt, np.nan)
            diff_s = np.where(both, ps - rs, np.nan)
        dt, ds = _max(diff_t), _max(diff_s)
        row = _row(package, m, ours, int(both.sum()), dt, ds, tt, ts, note)
        if package == "irtQ" and m == "TCC" and row["result"] == "Fail":
            # irtQ brackets the root on a 0.01 grid; a root exactly on a grid node is returned as the
            # neighbouring node. Explained only if every large difference is that case.
            big = np.abs(np.nan_to_num(diff_t)) > tt
            with np.errstate(invalid="ignore"):
                on_grid = np.abs(pt * 100 - np.round(pt * 100)) < 1e-6
            if np.all(on_grid[big]) and np.allclose(np.abs(diff_t[big]), 0.01, atol=1e-6):
                row["result"] = "Explained difference"
                row["note"] = (note + f" {int(big.sum())} root(s) lie exactly on irtQ's 0.01 search grid; irtQ returns "
                               "the neighbouring grid point (moving b by 0.003 removes the difference).").strip()
        rows.append(row)
        if package == "mirt" and m == "WLE":
            warm = wle(items, Y, s)
            fin = np.isfinite(warm.theta) & np.isfinite(rt)
            rows.append(_row(package, "WLE", "Warm WLE (ℓ′ + J/2I) — information only", int(fin.sum()),
                             _max(np.where(fin, warm.theta - np.where(fin, rt, 0), np.nan)), np.nan, np.nan, np.nan,
                             "Zero up to tolerance when J = I′ (Rasch, 2PL, PCM, GPCM, RSM, GRM); "
                             "differs when the test has 3PL / 4PL items.", informational=True))
    return ref, rows


def compare_all(items, Y, settings, packages=("irtQ", "mirt", "TAM")):
    rows, refs = [], {}
    for p in packages:
        ref, r = compare_package(p, items, Y, settings)
        refs[p] = ref
        rows += r
    return refs, rows
