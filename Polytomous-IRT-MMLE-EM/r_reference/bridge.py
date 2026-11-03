"""Rscript bridge to TAM (Rasch reference), irtQ (general polytomous reference) and mirt (supplementary).

The current dataset and settings are written to a temporary folder; the R script writes result.json.
Nothing is fabricated: when R, a package, or a design is unavailable the result says so with the reason.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.models import make_item
from core.quadrature import moments

HERE = Path(__file__).resolve().parent
SCRIPTS = dict(TAM="tam_validation.R", irtQ="irtq_validation.R", mirt="mirt_validation.R")
ROLE = dict(TAM="Primary Rasch reference (PCM, RSM); GPCM via tam.mml.2pl", irtQ="Primary general polytomous reference (PCM, GPCM, GRM)",
            mirt="Supplementary reference (PCM, GPCM, GRM; non-anchored designs)")


def r_environment():
    env = os.environ.copy()
    lib = env.get("POLY_R_LIBRARY") or env.get("SCORING_R_LIBRARY") or str(Path.home() / ".cache/ai_psychometrics2/R-library")
    if Path(lib).is_dir():
        env["POLY_R_LIBRARY"] = lib
    return env


@lru_cache(maxsize=1)
def check_r(timeout=90):
    exe = shutil.which("Rscript")
    if not exe:
        return dict(available=False, reason="Rscript was not found on PATH.")
    code = ('local_lib <- Sys.getenv("POLY_R_LIBRARY"); if (nzchar(local_lib)) .libPaths(c(local_lib, .libPaths()));'
            'v <- function(p) tryCatch(as.character(packageVersion(p)), error = function(e) "");'
            'cat(R.version.string, v("TAM"), v("irtQ"), v("mirt"), v("jsonlite"), sep = "|")')
    try:
        p = subprocess.run([exe, "--vanilla", "-e", code], capture_output=True, text=True, timeout=timeout,
                           env=r_environment())
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(available=False, reason=f"R could not start: {e}")
    if p.returncode:
        return dict(available=False, reason=p.stderr.strip()[-400:] or "R probe failed")
    r, tam, irtq, mirt, js = (p.stdout.strip().split("|") + [""] * 5)[:5]
    info = dict(available=bool(js), R=r, TAM=tam or None, irtQ=irtq or None, mirt=mirt or None, jsonlite=js or None)
    if not js:
        info["reason"] = "R package jsonlite is missing"
    else:
        missing = [k for k in ("TAM", "irtQ", "mirt") if not info[k]]
        info["reason"] = "R ready" if not missing else "Missing R package(s): " + ", ".join(missing)
    return info


def scope(package, cfg):
    """Return '' when the package can reproduce this configuration, else the reason it cannot."""
    m, design, prior = cfg.model, cfg.design, cfg.prior
    if cfg.grid != "equal":
        return "Reference runs use an equally spaced grid (irtQ's Quadrature = c(Q, bound)); choose that grid."
    if abs(cfg.D - 1) > 1e-12 and package == "TAM":
        return "TAM comparison requires D = 1."
    if package == "TAM":
        if m == "GPCM":
            if design != "single" or prior.endswith("empirical"):
                return "TAM's supplementary GPCM check (tam.mml.2pl) covers the single-group normal design."
            return ""
        if m not in ("PCM", "RSM"):
            return "TAM is the Rasch reference here (PCM, RSM; GPCM via tam.mml.2pl)."
        if prior.endswith("empirical"):
            return "Empirical priors are compared with irtQ, not TAM (different interface)."
        if prior == "fixed_normal" and design in ("concurrent", "mgfipc"):
            return "Fixed-normal multigroup runs are compared with irtQ."
    if package == "irtQ":
        if m == "RSM":
            return "irtQ has no rating scale model; use TAM."
        if m == "PCM" and prior.startswith("estimated") and design in ("single", "concurrent"):
            return ("irtQ fixes the reference group's variance at 1, whereas Rasch identification here (and in TAM) "
                    "estimates it; compare PCM with TAM, or use an anchored design.")
        if prior == "fixed_empirical" and design in ("fipc", "mgfipc"):
            return "irtQ FIPC estimates the new group's distribution; a fixed histogram is not combined with FIPC."
        if prior == "fixed_normal" and design in ("fipc", "mgfipc"):
            return "irtQ FIPC always updates the new group's distribution; choose an estimated prior."
        if design == "concurrent" and cfg.pooled:
            return "irtQ est_mg estimates a distribution per group (no pooled option)."
    if package == "mirt":
        if m == "RSM":
            return "Use TAM for RSM."
        if design in ("fipc", "mgfipc"):
            return "Fixed-item calibration is compared with irtQ."
        if prior == "fixed_empirical":
            return "mirt has no simple fixed-histogram prior interface."
        if prior == "estimated_normal" and design == "single" and m != "PCM":
            return "Single-group GPCM/GRM: the scale is fixed, so 'estimated normal' equals the fixed normal."
        if prior == "fixed_normal" and m == "PCM":
            return "mirt's Rasch itemtype estimates the latent variance; compare with an estimated prior."
        if design == "concurrent" and cfg.pooled:
            return "mirt multipleGroup estimates a distribution per group."
    return ""


def settings_payload(data, cfg):
    anchors = []
    if cfg.anchored:
        for k, it in data.anchors.items():
            row = dict(id=k, a=it.a, d=list(it.d))
            if it.model == "RSM":
                row.update(b=it.b, tau=list(it.tau))
            anchors.append(row)
    s = dict(model=cfg.model, design=cfg.design, prior=cfg.prior, n_nodes=int(cfg.n_nodes), bound=float(cfg.bound),
             mean=float(cfg.mean), sd=float(cfg.sd), D=float(cfg.D), fipc_method=cfg.fipc_method,
             Etol=float(min(cfg.tol, 1e-4)), MaxE=int(max(cfg.max_cycles, 500)), group_names=list(data.group_names),
             ncat=list(data.ncats), anchors=_columns(anchors))
    if cfg.prior == "fixed_empirical":
        s["fixed_weights"] = list(np.asarray(cfg.fixed_weights, float).ravel()[:cfg.n_nodes])
    return s


def _columns(rows):
    if not rows:
        return []
    keys = sorted({k for r in rows for k in r})
    return {k: [r.get(k) for r in rows] for k in keys}


def run_reference(package, data, cfg, timeout=900):
    """Returns dict(status, version, items {id: Item}, groups [...], weights, loglik, call, reason)."""
    info = check_r()
    if not info["available"]:
        return dict(status="Unavailable", reason=info["reason"])
    if not info.get(package):
        return dict(status="Unavailable", reason=f"R package {package} is not installed.")
    problem = scope(package, cfg)
    if problem:
        return dict(status="Not applicable", reason=problem)
    with tempfile.TemporaryDirectory(prefix="poly_ref_") as folder:
        folder = Path(folder)
        frame = pd.DataFrame(data.Y, columns=data.ids)
        frame.insert(0, "group", [data.group_names[g] for g in data.group])
        frame.to_csv(folder / "responses.csv", index=False, na_rep="", float_format="%.0f")
        (folder / "settings.json").write_text(json.dumps(settings_payload(data, cfg)))
        try:
            p = subprocess.run([shutil.which("Rscript"), "--vanilla", str(HERE / SCRIPTS[package]), str(folder)],
                               capture_output=True, text=True, timeout=timeout, env=r_environment())
        except subprocess.TimeoutExpired:
            return dict(status="Failed", reason=f"{package} exceeded {timeout} s.")
        if p.returncode or not (folder / "result.json").exists():
            return dict(status="Failed", reason=(p.stderr.strip() or p.stdout.strip())[-1500:])
        out = json.loads((folder / "result.json").read_text())
    if "error" in out:
        return dict(status="Failed", reason=str(out["error"]), version=out.get("version"))
    items = {}
    for r in out["items"]:
        d = np.atleast_1d(np.asarray(r["d"], float))
        items[r["id"]] = make_item(r["id"], cfg.model, float(r["a"]), tuple(d))
    return dict(status="Executed", version=out.get("version"), items=items, groups=out.get("groups", []),
                weights=[np.asarray(w, float) for w in out.get("weights", [])] or None,
                loglik=out.get("loglik"), niter=out.get("niter"), call=out.get("call"), seconds=out.get("seconds"),
                environment=info)


def compare(res, ref):
    """Item-parameter, group-moment and log-likelihood differences between the lab fit and a reference fit."""
    rows = []
    for it in res.items:
        r = ref["items"].get(it.id)
        if r is None:
            continue
        for lab, a, b in zip(it.labels(), it.vector(), r.vector()):
            if lab == "a" and it.model in ("PCM", "RSM"):
                continue
            rows.append(dict(item=it.id, parameter=lab, lab=a, reference=b, difference=a - b))
    items = pd.DataFrame(rows)
    groups = []
    rw = res.reported_weights
    mm = np.array([moments(res.nodes[g], rw[g]) for g in range(len(rw))])
    for g, gr in enumerate(ref.get("groups") or []):
        if g < len(mm):
            groups.append(dict(group=gr.get("name"), lab_mean=mm[g, 0], ref_mean=gr.get("mean"), lab_sd=mm[g, 1],
                               ref_sd=gr.get("sd")))
    gframe = pd.DataFrame(groups)
    wdist = None
    if ref.get("weights") is not None and len(ref["weights"]) == len(res.weights):
        try:
            wdist = max(0.5 * float(np.abs(np.asarray(w) - rw[g]).sum()) for g, w in enumerate(ref["weights"]))
        except ValueError:
            wdist = None
    ll = res.loglik_implied if res.loglik_implied is not None else res.loglik
    summary = dict(max_item_diff=float(items["difference"].abs().max()) if len(items) else None,
                   loglik_lab=ll, loglik_ref=ref.get("loglik"),
                   loglik_diff=(ll - ref["loglik"]) if ref.get("loglik") is not None else None,
                   max_density_TV=wdist,
                   max_moment_diff=float(np.nanmax(np.abs(np.r_[gframe["lab_mean"] - gframe["ref_mean"].astype(float),
                                                                 gframe["lab_sd"] - gframe["ref_sd"].astype(float)])))
                   if len(gframe) else None)
    return items, gframe, summary


def reference_moments(ref, nodes):
    if ref.get("weights"):
        return np.array([moments(nodes, w) for w in ref["weights"]])
    return None
