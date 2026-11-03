import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components.ui import config, data, guided, interpret, labels, latest_fit, page_setup, table
from core.em import DESIGN_NAMES, fit
from core.priors import PRIOR_NAMES
from r_reference.bridge import ROLE, check_r, compare, run_reference, scope

page_setup("12 · Reference Validation", "Run TAM (Rasch), irtQ (general polytomous) and mirt (supplementary) on the same "
                                        "data and settings, and compare item parameters, group moments, histograms and "
                                        "log-likelihoods.")
guided(["Use an equally spaced grid (irtQ's Quadrature = c(Q, bound)).",
        "Pick the reference packages whose scope covers the current model × design × prior.",
        "Run: the lab and each package fit the same responses; differences are tabulated.",
        "Read 'Not applicable' reasons: they mark real interface or identification differences."],
       "irtQ's `Quadrature = c(49, 6)` specifies…",
       ["49 evenly spaced points on [−6, 6]", "49 Gauss–Hermite points", "a 6-point Gauss–Hermite rule",
        "49 points from a fitted histogram"], "49 evenly spaced points on [−6, 6]",
       "It is an equally spaced grid with normalised normal masses — not Gauss–Hermite.", key="ref")
d = data()
cfg = config()
info = check_r()
if not info["available"]:
    st.warning(f"R reference runs are unavailable: {info['reason']}. Results below are **unverified**.")
else:
    st.success(f"{info['R']} · TAM {info.get('TAM') or '—'} · irtQ {info.get('irtQ') or '—'} · mirt {info.get('mirt') or '—'}")
st.markdown(f"**Settings:** {cfg.model} · {DESIGN_NAMES[cfg.design]} · {PRIOR_NAMES[cfg.prior]} · Q = {cfg.n_nodes} on "
            f"±{cfg.bound:g} ({cfg.grid}) · D = {cfg.D}" + (f" · {cfg.fipc_method}" if cfg.anchored else ""))
rows = [dict(package=p, role=ROLE[p], applicable="yes" if not scope(p, cfg) else "no", reason=scope(p, cfg) or "—")
        for p in ("TAM", "irtQ", "mirt")]
table(pd.DataFrame(rows))
tight = st.checkbox("Tight tolerances for both sides (1e-6)", value=True)
pk = st.multiselect("Packages", ["TAM", "irtQ", "mirt"], default=[r["package"] for r in rows if r["applicable"] == "yes"])
if st.button("Run lab engine and references", type="primary", disabled=not pk):
    c = replace(cfg, tol=1e-6 if tight else cfg.tol, prior_tol=1e-6 if tight else cfg.prior_tol,
                max_cycles=max(cfg.max_cycles, 3000 if tight else cfg.max_cycles))
    try:
        res = fit(d, c, record=False)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    out = {}
    for p in pk:
        with st.spinner(f"Running {p}…"):
            out[p] = run_reference(p, d, c)
    st.session_state.validation = (res, out, d.identity, c)
if "validation" in st.session_state and st.session_state.validation[2] == d.identity:
    res, out, _, c = st.session_state.validation
    labels("edu", "ref")
    st.caption(f"Lab: {res.cycles} cycles, ℓ = {res.loglik:.6f}, {'converged' if res.converged else res.reason}.")
    summary = []
    for p, ref in out.items():
        if ref["status"] != "Executed":
            summary.append(dict(package=p, status=ref["status"], reason=ref.get("reason", "")[:200]))
            continue
        items, groups, sm = compare(res, ref)
        summary.append(dict(package=p, status="Executed", version=ref.get("version"),
                            max_item_diff=sm["max_item_diff"], max_moment_diff=sm["max_moment_diff"],
                            loglik_lab=sm["loglik_lab"], loglik_ref=sm["loglik_ref"], loglik_diff=sm["loglik_diff"],
                            max_density_TV=sm["max_density_TV"], seconds=ref.get("seconds")))
        with st.expander(f"{p} · {ref.get('call')}"):
            table(items, digits=6, sci=("difference",), height=300)
            if len(groups):
                table(groups, digits=6)
    table(pd.DataFrame(summary), digits=6, sci=("max_item_diff", "max_moment_diff", "loglik_diff", "max_density_TV"))
    st.download_button("Download validation record (JSON)", json.dumps(dict(
        settings=c.as_dict(), dataset=d.name, lab=dict(loglik=res.loglik, cycles=res.cycles, converged=res.converged),
        references=[{k: v for k, v in s.items()} for s in summary], environment=check_r()), default=str, indent=2),
        "validation_record.json", "application/json")
st.subheader("Recorded local validation run")
record_path = Path(__file__).resolve().parents[1] / "docs" / "validation_record.json"
if not record_path.exists():
    st.info("No recorded run found (docs/validation_record.json). Generate it with "
            "`python -m r_reference.run_validation`.")
else:
    rec = json.loads(record_path.read_text())
    env = rec["environment"]
    st.info(f"**Recorded, not live.** These numbers come from the 19 fixed simulated cases run on "
            f"{rec['date']} with {env.get('R')} · TAM {env.get('TAM')} · irtQ {env.get('irtQ')} · mirt {env.get('mirt')} "
            f"({rec['grid']}, tight tolerances). They describe those cases only, not your current data or settings; "
            "use the live check above when R is available.")
    view = pd.DataFrame([dict(package=f"{r['package']} {r.get('version') or ''}".strip(), setup=r["setup"],
                              status=r["status"], max_item_diff=r.get("max_item_diff"),
                              max_moment_diff=r.get("max_moment_diff"), loglik_lab=r.get("loglik_lab"),
                              loglik_ref=r.get("loglik_ref"), max_density_TV=r.get("max_density_TV"), note=r["note"])
                         for r in rec["rows"]])
    sci = ("max_item_diff", "max_moment_diff", "max_density_TV")
    st.markdown("**Agreement cases** (asserted by `tests/test_reference.py`)")
    table(view[[r["kind"] == "agreement" for r in rec["rows"]]], digits=4, sci=sci)
    st.markdown("**Documented differences** (explained in docs/VALIDATION.md, not asserted)")
    table(view[[r["kind"] == "difference" for r in rec["rows"]]], digits=4, sci=sci)
    st.download_button("Download recorded run (JSON)", record_path.read_text(), "validation_record.json",
                       "application/json", key="recorded_dl")
interpret("Agreement to about 1e-5 is expected when both sides solve the same problem to tight tolerance (TAM PCM/RSM, "
          "irtQ GPCM/GRM with fixed normal, estimated normal via posterior moments, FIPC MEM/OEM, MG-FIPC; mirt "
          "single/multigroup). Expected, explained differences: (1) **free histograms** — irtQ stops on item-parameter "
          "change while the histogram keeps creeping towards a spikier shape, so histograms differ even when items agree "
          "to ~1e-3; (2) **exact vs moment normal update** — choose 'Posterior moments' to match TAM/irtQ exactly; "
          "(3) **constrained histogram** — the lab's exact constrained M-step and irtQ's Woods rescaling are different "
          "estimators; choose 'Woods (irtQ)' in the sidebar to compare like with like. docs/VALIDATION.md records the "
          "executed comparisons.")
