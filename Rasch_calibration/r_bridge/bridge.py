"""Rscript bridge to the reference packages (TAM, eRm).

Data go to a temporary folder as CSV (blank = missing) plus a JSON settings file; the R
script writes result.json. Nothing here ever fabricates a reference result: when R or a
package is unavailable the call returns status "Unavailable" with the reason.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = dict(JMLE="tam_jmle.R", MML="tam_mml.R", WLE="tam_wle.R", CMLE="erm_cmle.R")
PACKAGE = dict(JMLE="TAM", MML="TAM", WLE="TAM", CMLE="eRm")


def r_environment():
    """Prefer a project R library (RASCH_R_LIBRARY, else the shared project library if present)."""
    env = os.environ.copy()
    lib = env.get("RASCH_R_LIBRARY") or str(Path.home() / ".cache/ai_psychometrics2/R-library")
    if Path(lib).is_dir():
        env["RASCH_R_LIBRARY"] = lib
    return env


def check_r(timeout=60):
    exe = shutil.which("Rscript")
    if not exe:
        return dict(available=False, R=None, TAM=None, eRm=None, reason="Rscript was not found on PATH.")
    try:
        p = subprocess.run([exe, "--vanilla", str(HERE / "check_packages.R")], capture_output=True, text=True,
                           timeout=timeout, env=r_environment())
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(available=False, R=None, TAM=None, eRm=None, reason=f"R could not start: {e}")
    if p.returncode:
        return dict(available=False, R=None, TAM=None, eRm=None, reason=p.stderr.strip()[-500:] or "R probe failed")
    info = json.loads(p.stdout)
    info["available"] = True
    missing = [k for k in ("TAM", "eRm") if not info.get(k)]
    info["reason"] = "R ready" if not missing else "Missing R package(s): " + ", ".join(missing)
    return info


def run_reference(method, X, settings, item_ids=None, timeout=600):
    """Run one reference estimator. Returns a dict with status Executed / Unavailable / Failed."""
    info = check_r()
    pkg = PACKAGE[method]
    if not info["available"] or not info.get(pkg):
        return dict(status="Unavailable", reason=info["reason"] if info["available"] else info["reason"], environment=info)
    X = np.asarray(X, float)
    cols = item_ids or [f"I{j+1:02d}" for j in range(X.shape[1])]
    with tempfile.TemporaryDirectory(prefix="rasch_ref_") as folder:
        folder = Path(folder)
        pd.DataFrame(X, columns=cols).to_csv(folder / "responses.csv", index=False, na_rep="")
        (folder / "settings.json").write_text(json.dumps(_jsonable(settings)))
        try:
            p = subprocess.run([shutil.which("Rscript"), "--vanilla", str(HERE / SCRIPTS[method]), str(folder)],
                               capture_output=True, text=True, timeout=timeout, env=r_environment())
        except subprocess.TimeoutExpired:
            return dict(status="Failed", reason=f"{pkg} exceeded {timeout} s")
        if p.returncode or not (folder / "result.json").exists():
            return dict(status="Failed", reason=(p.stderr.strip() or p.stdout.strip())[-1500:])
        out = json.loads((folder / "result.json").read_text())
    for key in ("b", "se_b", "theta", "wle", "wle_se", "mle", "mle_se", "person_scores", "person_max",
                "group_means", "group_variances"):
        if key in out and out[key] is not None:
            out[key] = np.array([np.nan if v is None else v for v in np.atleast_1d(out[key])], float)
    out["status"] = "Executed"
    out["environment"] = info
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
