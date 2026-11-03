import time
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from components.ui import (config, data, fit_key, four_views, guided, interpret, lines, page_setup, show_plan, table)
from core.data import Dataset
from core.diagnostics import group_recovery, information_criteria, item_recovery
from core.em import ANCHORED, DESIGN_NAMES, DESIGNS, fit, plan
from core.models import MODELS
from core.priors import PRIOR_NAMES, PRIOR_TYPES

page_setup("7 · Calibration Designs", "Single-group, concurrent, fixed-item (FIPC) and concurrent fixed-item (MG-FIPC) "
                                      "calibration — identification, running a calibration and comparing set-ups.")
ss = st.session_state
guided(["Read the identification table: which distribution moments are fixed, free or not updated.",
        "Check the design × prior grid for the active dataset — invalid combinations explain why.",
        "Run the calibration (sidebar settings), then open page 8 to inspect every iteration.",
        "Use the comparison laboratory to run several priors/designs on the same responses."],
       "Why can a GPCM calibration without anchors not estimate the reference group's mean and SD freely?",
       ["Shifting/stretching θ can be undone by changing all a and δ — the likelihood is unchanged",
        "Because the EM algorithm cannot update normal distributions", "Because irtQ does not allow it",
        "Because the posterior would not sum to one"],
       "Shifting/stretching θ can be undone by changing all a and δ — the likelihood is unchanged",
       "θ* = (θ − m)/s with a* = a·s and δ* = (δ − m)/s gives identical probabilities, so the metric must be fixed "
       "by the latent distribution (or by anchors). For PCM/RSM the slope is fixed at 1, so only the location is free.",
       key="design")

d = data()
cfg = config()
tab_id, tab_run, tab_cmp = st.tabs(["Designs & identification", "Run calibration", "Comparison laboratory"])

with tab_id:
    table(pd.DataFrame([
        dict(design="Single-group", data="one population/form", fixed_items="none",
             metric="reference distribution: mean fixed (and SD for GPCM/GRM)", estimates="all items; Rasch SD"),
        dict(design="Concurrent", data="2+ groups/forms", fixed_items="none",
             metric="reference group as above; common items link the forms",
             estimates="all items jointly (common items share parameters); focal-group distributions"),
        dict(design="FIPC", data="new responses + bank anchors", fixed_items="anchors",
             metric="anchor parameters", estimates="new items; the new group's distribution (MEM) "),
        dict(design="MG-FIPC", data="2+ groups/forms + bank anchors", fixed_items="anchors",
             metric="anchor parameters", estimates="free items jointly; every group's distribution")]))
    st.markdown(f"**Current settings:** {cfg.model} · {DESIGN_NAMES[cfg.design]} · {PRIOR_NAMES[cfg.prior]}"
                + (" · pooled prior" if cfg.pooled and cfg.design in ("concurrent", "mgfipc") else "")
                + (f" · {'MWU–MEM' if cfg.fipc_method == 'MEM' else 'NWU–OEM'}" if cfg.anchored else ""))
    pl = plan(d, cfg)
    show_plan(pl, d.group_names)
    if pl["fixed"].any():
        st.caption("Fixed anchors: " + ", ".join(np.array(d.ids)[pl["fixed"]]))
    st.markdown("**Design × prior check for this dataset** (✓ runs · ✗ reason)")
    grid = []
    for design in DESIGNS:
        row = dict(design=DESIGN_NAMES[design])
        for prior in PRIOR_TYPES:
            p = plan(d, replace(cfg, design=design, prior=prior,
                                grid="equal" if prior != "fixed_normal" else cfg.grid))
            row[PRIOR_NAMES[prior]] = "✓" if not p["errors"] else "✗ " + p["errors"][0][:90]
        grid.append(row)
    table(pd.DataFrame(grid))
    interpret("**FIPC methods.** NWU–OEM computes the posterior from the anchors alone under the initial prior, runs one "
              "M-step for the new items and stops — a one-cycle estimate, never a converged solution. MWU–MEM starts the "
              "same way, then cycles with all items and updates the group distribution every cycle until convergence. "
              "In both, anchor parameters never change but their response probabilities enter every posterior.")

with tab_run:
    pl = plan(d, cfg)
    show_plan(pl, d.group_names)
    if st.button("Run MMLE–EM calibration", type="primary", disabled=bool(pl["errors"])):
        bar = st.progress(0.0, text="Starting…")
        status = st.empty()

        def cb(h):
            bar.progress(min(h["cycle"] / max(cfg.max_cycles, 1), 1.0),
                         text=f"Cycle {h['cycle']} · ℓ = {h['loglik']:.4f} · Δitem {h['item_change']:.2e} · "
                              f"Δπ {h['weight_change']:.2e}")
        try:
            res = fit(d, cfg, callback=cb)
            ss.fits["latest"] = res
            ss.fits[fit_key(d, cfg)] = res
            bar.progress(1.0, text="Done")
        except ValueError as e:
            st.error(str(e))
    res = ss.fits.get("latest")
    if res is not None and res.data_id == d.identity:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", "Converged" if res.converged else ("One cycle (OEM)" if "OEM" in res.reason else "Not converged"))
        c2.metric("Cycles", res.cycles)
        c3.metric("Marginal log-likelihood", f"{res.loglik:.4f}")
        c4.metric("Seconds", f"{res.seconds:.2f}")
        st.caption(res.reason)
        for w in res.warnings:
            st.warning(w)
        mm = res.group_moments
        table(pd.DataFrame(dict(group=res.group_names, mean=mm[:, 0], sd=mm[:, 1])))
        table(res.item_table())
        label = st.text_input("Label for comparison", f"{res.config.model} · {res.config.design} · {res.config.prior}"
                              + (f" · {res.config.fipc_method}" if res.config.anchored else ""))
        if st.button("Save this fit for comparison"):
            ss.saved[label] = res
            st.success(f"Saved '{label}'. See the comparison laboratory tab and page 8.")
        st.info("Open **8 · Results** to inspect every iteration: posteriors, expected counts, item and prior updates.")

with tab_cmp:
    st.markdown("Run several set-ups on the **same responses**. Item parameters are compared only between fits of the "
                "same model and the same metric; across models compare log-likelihood, AIC/BIC and group moments.")
    c1, c2, c3 = st.columns(3)
    models = c1.multiselect("Models", MODELS, default=[cfg.model])
    priors = c2.multiselect("Priors", PRIOR_TYPES, default=["fixed_normal", "estimated_normal", "estimated_empirical"],
                            format_func=PRIOR_NAMES.get)
    designs = c3.multiselect("Designs", DESIGNS, default=[cfg.design], format_func=DESIGN_NAMES.get)
    sens = st.checkbox("Anchor sensitivity: also refit anchored designs with every anchor location shifted by +0.2",
                       value=False)
    if st.button("Run comparison", type="primary"):
        out, fits = [], {}
        prog = st.progress(0.0)
        combos = [(m, p, g) for m in models for p in priors for g in designs]
        for k, (m, p, g) in enumerate(combos):
            c = replace(cfg, model=m, prior=p, design=g, grid="equal" if p != "fixed_normal" else cfg.grid,
                        fixed_weights=cfg.fixed_weights, se=False)
            if p == "fixed_empirical" and c.fixed_weights is None:
                from components.ui import empirical_weights
                c.fixed_weights = list(empirical_weights())
            label = f"{m} · {DESIGN_NAMES[g]} · {PRIOR_NAMES[p]}"
            try:
                r = fit(d, c, record=False)
            except ValueError as e:
                out.append(dict(setup=label, status="✗ " + str(e)[:120]))
                continue
            ic = information_criteria(r, d.N)
            mm = r.group_moments
            row = dict(setup=label, status="converged" if r.converged else r.reason[:40], cycles=r.cycles,
                       loglik=r.loglik, parameters=ic["parameters"], AIC=ic["AIC"], BIC=ic["BIC"],
                       **{f"mean {n}": mm[i, 0] for i, n in enumerate(d.group_names)},
                       **{f"SD {n}": mm[i, 1] for i, n in enumerate(d.group_names)},
                       mean_a=np.mean([it.a for it in r.items]),
                       mean_location=np.mean([it.location() for it in r.items]), seconds=r.seconds)
            rec = item_recovery(d, r)
            if len(rec) and m == (d.truth or {}).get("spec", {}).get("model"):
                free = rec[~rec["fixed"]]
                row["RMSE a"] = float(np.sqrt(np.mean(free[free["kind"] == "a"]["error"] ** 2))) if (free["kind"] == "a").any() else np.nan
                row["RMSE location"] = float(np.sqrt(np.mean(free[free["kind"] == "location"]["error"] ** 2)))
                gr = group_recovery(d, r)
                row["max |mean error|"] = float(gr["mean_error"].abs().max())
            if sens and g in ANCHORED:
                from core.simulation import _shift
                d2 = Dataset(d.name, d.Y, d.group, d.group_names, d.specs,
                             anchors={k2: _shift(v, 0.2) for k2, v in d.anchors.items()}, truth=d.truth)
                r2 = fit(d2, c, record=False)
                row["Δ group mean (anchors +0.2)"] = float(np.max(r2.group_moments[:, 0] - mm[:, 0]))
                row["max |Δ free location|"] = float(max(abs(a2.location() - a1.location())
                                                       for a1, a2, f in zip(r.items, r2.items, r.fixed) if not f))
            out.append(row)
            fits[label] = r
            prog.progress((k + 1) / len(combos))
        ss.comparison = (pd.DataFrame(out), fits, d.identity)
    if "comparison" in ss and ss.comparison[2] == d.identity:
        frame, fits, _ = ss.comparison
        table(frame, digits=4)
        if fits:
            x = next(iter(fits.values())).nodes[0]
            g = st.selectbox("Group for distribution overlay", range(d.G), format_func=lambda g: d.group_names[g])
            lines(x, {k: r.weights[g] for k, r in fits.items()}, "Fitted latent distributions", yl="mass", key="cmp_w")
            same = [k for k, r in fits.items()]
            pick = st.multiselect("Compare item parameters of", same, default=same[:2])
            if len(pick) >= 2:
                ms = {fits[k].config.model for k in pick}
                if len(ms) > 1:
                    st.warning("Different models: parameters have different meanings and are not compared.")
                else:
                    rows = []
                    for j, it in enumerate(fits[pick[0]].items):
                        for lab_i, lab in enumerate(it.labels()):
                            if lab == "a" and it.model in ("PCM", "RSM"):
                                continue
                            rows.append(dict(item=it.id, parameter=lab, **{k: fits[k].items[j].vector()[lab_i] for k in pick}))
                    table(pd.DataFrame(rows), digits=4)
    if ss.saved:
        st.markdown("**Saved fits**")
        table(pd.DataFrame([dict(label=k, model=r.config.model, design=r.config.design, prior=r.config.prior,
                                 loglik=r.loglik, converged=r.converged, cycles=r.cycles,
                                 same_data=r.data_id == d.identity) for k, r in ss.saved.items()]))

four_views(
    "A calibration design decides which item parameters are free, which are fixed, how many latent distributions there "
    "are and what fixes the metric. Concurrent calibration estimates every form at once: common items carry one "
    "parameter vector wherever administered. Fixed-item calibration keeps bank parameters unchanged; their responses "
    "still inform each examinee's posterior, which is how the new items land on the bank scale.",
    ["$$\\ell(\\xi,\\pi)=\\sum_g\\sum_{i\\in g}\\log\\Big[\\sum_q\\pi_{gq}\\prod_{j\\in O_{ig}}P_{j,y_{igj}}(\\theta_q;\\xi)\\Big]$$",
     "Shared items: the item M-step maximises Σ_g Σ_q Σ_k C_gjkq log P_jk(θ_q) — counts from every group that saw "
     "item j, one parameter vector.",
     "Reference identification (no anchors): μ₁ = 0, σ₁ = 1 (GPCM/GRM) or μ₁ = 0 with σ₁ free (PCM/RSM). "
     "Anchored designs: all group moments free; the anchors' fixed parameters set location and unit.",
     "Pooled prior: one π for all groups, updated from Σ_g n_gq (sample-size weighting). Group-specific: π_g from n_gq only."],
    None, [plan, fit])
