"""Rscript bridge to the reference packages: irtQ (primary), mirt, TAM.

Items go to a temporary folder as JSON, responses as CSV (blank = missing), settings as JSON; the
R script writes result.json. Nothing is ever fabricated: when R or a package is unavailable, or a
model is outside a package's scope, the result says "Unavailable" with the reason.
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

HERE = Path(__file__).resolve().parent
SCRIPTS = dict(irtQ="irtq_scores.R", mirt="mirt_scores.R", TAM="tam_scores.R")
# Our method name -> package method name.
METHOD_NAMES = dict(irtQ=dict(MLE="ML", WLE="WL", MAP="MAP", EAP="EAP", EAPsum="EAP.SUM", TCC="INV.TCC"),
                    mirt=dict(MLE="ML", WLE="WLE", MAP="MAP", EAP="EAP", EAPsum="EAPsum"),
                    TAM=dict(MLE="MLE", WLE="WLE"))
# Models each package can score with fixed parameters.
SCOPE = dict(irtQ={"Rasch", "2PL", "3PL", "PCM", "GPCM", "RSM", "GRM"},
             mirt={"Rasch", "2PL", "3PL", "4PL", "PCM", "GPCM", "RSM", "GRM"},
             TAM={"Rasch", "PCM", "RSM"})


def r_environment():
    env = os.environ.copy()
    lib = env.get("SCORING_R_LIBRARY") or str(Path.home() / ".cache/ai_psychometrics2/R-library")
    if Path(lib).is_dir():
        env["SCORING_R_LIBRARY"] = lib
    return env


@lru_cache(maxsize=1)
def check_r(timeout=60):
    exe = shutil.which("Rscript")
    if not exe:
        return dict(available=False, reason="Rscript was not found on PATH.")
    try:
        p = subprocess.run([exe, "--vanilla", str(HERE / "check_packages.R")], capture_output=True, text=True,
                           timeout=timeout, env=r_environment())
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(available=False, reason=f"R could not start: {e}")
    if p.returncode:
        return dict(available=False, reason=p.stderr.strip()[-500:] or "R probe failed")
    info = json.loads(p.stdout)
    info["available"] = True
    missing = [k for k in ("irtQ", "mirt", "TAM") if not info.get(k)]
    info["reason"] = "R ready" if not missing else "Missing R package(s): " + ", ".join(missing)
    return info


def scope_problem(package, items):
    out = sorted({it.model for it in items} - SCOPE[package])
    if out:
        return f"{package} cannot score {', '.join(out)} items with fixed parameters here"
    if package == "TAM" and any(abs(it.D - 1) > 1e-12 for it in items):
        return "TAM comparison requires D = 1"
    if package == "irtQ" and len({it.D for it in items}) > 1:
        return "irtQ takes one D for all items"
    return ""


def item_payload(items):
    return [dict(id=it.id, model=it.model, a=it.a, b=it.b, c=it.c, d=it.d, steps=list(it.steps),
                 # Divide-by-total step difficulties; a dichotomous Rasch item is a PCM item with one step b.
                 deltas=[it.b] if it.model in ("Rasch", "2PL", "3PL", "4PL") else list(it.deltas),
                 D=it.D) for it in items]


def run_reference(package, items, Y, settings, methods, timeout=600):
    """Returns dict(status, version, results={our method: dict(theta, se) or dict(error)}, reason)."""
    info = check_r()
    if not info["available"]:
        return dict(status="Unavailable", reason=info["reason"])
    if not info.get(package):
        return dict(status="Unavailable", reason=f"R package {package} is not installed")
    problem = scope_problem(package, items)
    if problem:
        return dict(status="Unavailable", reason=problem)
    names = METHOD_NAMES[package]
    wanted = [m for m in methods if m in names]
    if not wanted:
        return dict(status="Unavailable", reason=f"{package} has none of: {', '.join(methods)}")
    from scoring.eap import quadrature
    nodes, weights = quadrature(settings)
    payload = settings.as_dict()
    payload.update(methods=[names[m] for m in wanted], D=items[0].D, nodes=nodes.tolist(), weights=weights.tolist())
    cols = [it.id for it in items]
    with tempfile.TemporaryDirectory(prefix="scoring_ref_") as folder:
        folder = Path(folder)
        pd.DataFrame(np.atleast_2d(Y), columns=cols).to_csv(folder / "responses.csv", index=False, na_rep="")
        (folder / "items.json").write_text(json.dumps(item_payload(items)))
        (folder / "settings.json").write_text(json.dumps(payload))
        try:
            p = subprocess.run([shutil.which("Rscript"), "--vanilla", str(HERE / SCRIPTS[package]), str(folder)],
                               capture_output=True, text=True, timeout=timeout, env=r_environment())
        except subprocess.TimeoutExpired:
            return dict(status="Failed", reason=f"{package} exceeded {timeout} s")
        if p.returncode or not (folder / "result.json").exists():
            return dict(status="Failed", reason=(p.stderr.strip() or p.stdout.strip())[-1500:])
        out = json.loads((folder / "result.json").read_text())
    results = {}
    back = {v: k for k, v in names.items()}
    for pm, r in out.get("results", {}).items():
        ours = back.get(pm, pm)
        if "error" in r:
            results[ours] = dict(error=r["error"] if isinstance(r["error"], str) else str(r["error"]))
            continue
        results[ours] = {k: _decode(r.get(k)) for k in ("theta", "se")}
    return dict(status="Executed", version=out.get("version"), results=results, environment=info)


def _decode(v):
    a = np.array([np.nan if x is None else x for x in np.atleast_1d(v if v is not None else [])], float)
    a[a >= 1e307] = np.inf
    a[a <= -1e307] = -np.inf
    return a
