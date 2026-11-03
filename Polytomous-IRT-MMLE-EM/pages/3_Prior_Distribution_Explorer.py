import json

import numpy as np
import pandas as pd
import streamlit as st

from components.ui import (bars, config, data, empirical_weights, four_views, guided, interpret, keep, latest_fit,
                           lines, page_setup, show_plan, table)
from core import priors as P
from core.diagnostics import information_criteria, item_recovery
from core.em import fit, plan, start_items
from core.quadrature import moments, normal_weights, skewness
from core.walkthrough import cycle

page_setup("3 · Prior Distribution Explorer",
           "The latent ability distribution in MMLE: fixed or estimated, normal or empirical — how it enters the "
           "posterior and how one EM cycle updates it.")
ss = st.session_state
guided(["Choose a prior type in the sidebar (A–D) and read how it is defined on the quadrature grid.",
        "Pick an examinee: compare the conditional likelihood with the posterior — the prior pulls it.",
        "Look at the one-cycle update: unconstrained masses are the average posterior; constrained masses also "
        "satisfy the identification moments.",
        "Run the comparison at the bottom on the bimodal example (dataset 5) and on a normal one."],
       "Without moment constraints, the M-step for a free histogram sets π_q to…",
       ["the average posterior probability at node q", "the normal density at θ_q", "the posterior mode",
        "the observed proportion of examinees with θ near θ_q"], "the average posterior probability at node q",
       "Maximising Σ_q n_q log π_q subject to Σπ_q = 1 gives π_q = n_q/N = (1/N) Σ_i w_iq.", key="prior")

d = data()
cfg = config()
st.info("These are **latent ability distributions** in the marginal likelihood (g(θ) in ∫ L(θ) g(θ) dθ). They are not "
        "Bayesian priors on item parameters; this lab estimates items by pure MMLE.")

st.subheader("Current prior on the quadrature grid")
pl = plan(d, cfg)
show_plan(pl, d.group_names)
if pl["errors"]:
    st.stop()
from core.em import build_grid  # noqa: E402

try:
    nodes, weights = build_grid(cfg, d.G)
except ValueError as e:
    st.error(str(e))
    st.stop()
x = nodes[0]
c1, c2 = st.columns([3, 2])
with c1:
    series = {d.group_names[g]: weights[g] for g in range(d.G)}
    series["N(μ, σ²) for comparison"] = normal_weights(x, cfg.mean, cfg.sd) if cfg.grid != "gauss_hermite" else weights[0]
    bars(np.round(x, 3), series, f"{P.PRIOR_NAMES[cfg.prior]} · {len(x)} nodes", yl="mass π_q", key="prior_plot")
with c2:
    mm = moments(x, weights[0])
    st.metric("Mean", f"{mm[0]:.4f}")
    st.metric("SD", f"{mm[1]:.4f}")
    st.metric("Skewness", f"{skewness(x, weights[0]):.4f}")
    st.caption("π_q ≥ 0 and Σ π_q = " + f"{weights[0].sum():.12f}")

if cfg.prior in ("fixed_empirical", "estimated_empirical"):
    with st.expander("Build, import or edit empirical masses", expanded=True):
        st.caption("Masses live on the same fixed grid as the calibration. Imported historical weights must already be "
                   "on the intended item metric — this lab never silently standardises them.")
        frame = pd.DataFrame(dict(theta=x, weight=empirical_weights(x)))
        edited = st.data_editor(frame, disabled=["theta"], hide_index=True, key="emp_edit", height=260)
        c1, c2 = st.columns(2)
        if c1.button("Use edited masses (normalised)"):
            w = edited["weight"].to_numpy(float)
            if np.any(w < 0) or w.sum() <= 0:
                st.error("Masses must be non-negative with a positive total.")
            else:
                ss.cfg_emp_custom = json.dumps((w / w.sum()).tolist())
                ss.cfg_emp_shape = "custom"
                st.rerun()
        up = c2.file_uploader("Import historical weights (CSV: theta, weight)", type="csv")
        if up is not None:
            h = pd.read_csv(up)
            if not {"theta", "weight"} <= set(h.columns):
                st.error("The file needs columns 'theta' and 'weight'.")
            elif len(h) != len(x) or np.max(np.abs(h["theta"].to_numpy(float) - x)) > 1e-6:
                st.error(f"The imported grid ({len(h)} points, {h['theta'].min():g} … {h['theta'].max():g}) differs "
                         f"from the calibration grid ({len(x)} points, ±{cfg.bound:g}). Set the grid to match — "
                         "re-gridding a historical distribution changes it and is not done silently.")
            else:
                hm = moments(x, h["weight"].to_numpy(float) / h["weight"].sum())
                st.write(f"Imported moments: mean {hm[0]:.4f}, SD {hm[1]:.4f}.")
                if st.button("Use imported weights"):
                    w = h["weight"].to_numpy(float)
                    ss.cfg_emp_custom = json.dumps((w / w.sum()).tolist())
                    ss.cfg_emp_shape = "custom"
                    st.rerun()

st.subheader("One EM cycle through the prior")
res = latest_fit()
use_fit = res is not None and res.config.model == cfg.model and len(res.items) == d.J
items = res.items if use_fit else start_items(d, cfg.model, pl["fixed"])
st.caption("Item parameters used: " + ("the latest calibration of this dataset." if use_fit else
                                       "starting values (run a calibration on page 7 to use estimates)."))
try:
    cyc = cycle(d, cfg, items)
except ValueError as e:
    st.error(str(e))
    st.stop()
i = st.number_input("Examinee", 0, d.N - 1, min(int(ss.get("person", 0)), d.N - 1), key="prior_person")
g = d.group[i]
L = np.exp(cyc["estep"]["logL"][i] - cyc["estep"]["logL"][i].max())
c1, c2 = st.columns(2)
with c1:
    lines(nodes[g], {"conditional likelihood (scaled to max 1)": L, "prior π_q × Q (scaled)": cyc["weights"][g] / cyc["weights"][g].max(),
                     "posterior w_iq (scaled)": cyc["estep"]["post"][i] / cyc["estep"]["post"][i].max()},
          f"Examinee {i} · responses {['·' if np.isnan(v) else int(v) for v in d.Y[i]][:20]}", yl="relative height",
          key="post_plot")
with c2:
    j = st.selectbox("Item for expected counts", range(d.J), format_func=lambda j: d.ids[j])
    C = cyc["counts"][j][g]
    bars(np.round(nodes[g], 2), {f"k={k}": C[:, k] for k in range(C.shape[1])},
         f"Expected category counts C_jkq · item {d.ids[j]} · {d.group_names[g]}", yl="expected count",
         mode="stack", key="counts_plot")
c1, c2 = st.columns(2)
with c1:
    bars(np.round(nodes[0], 2), {"before": cyc["weights"][g], "after this cycle": cyc["new_weights"][g]},
         "Prior masses before and after the M-step", key="upd_plot")
with c2:
    mb, ma = moments(nodes[g], cyc["weights"][g]), moments(nodes[g], cyc["new_weights"][g])
    table(pd.DataFrame([dict(quantity="mean", before=mb[0], after=ma[0]), dict(quantity="SD", before=mb[1], after=ma[1]),
                        dict(quantity="marginal log-likelihood", before=cyc["ll_before"], after=cyc["ll_after"]),
                        dict(quantity="largest |Δπ|", before=np.nan, after=cyc["weight_change"]),
                        dict(quantity="largest |Δ item parameter|", before=np.nan, after=cyc["item_change"])]), digits=5)
    for info in cyc["prior_info"]:
        st.caption(f"{info['method']} · Q_prior {info['q_before']:.4f} → {info['q_after']:.4f}"
                   + (f" · constraint violation {info['violation']:.1e}" if "violation" in info else ""))
    if not cyc["prior_info"]:
        st.caption("This prior is not updated (fixed by assumption).")
with st.expander("Updated item parameters after this cycle"):
    table(pd.DataFrame([dict(item=a.id, **{f"before {l}": v for l, v in zip(a.labels(), a.vector())},
                             **{f"after {l}": v for l, v in zip(b.labels(), b.vector())})
                        for a, b in zip(cyc["items"], cyc["new_items"])]), digits=4)

st.subheader("Normal vs empirical: does the ability distribution change calibration?")
st.caption("Fits fixed normal, estimated normal and estimated empirical priors to the same data. Try the bimodal "
           "example (helps), a normal dataset (negligible difference) and a small sample (instability).")
if st.button("Run the three-prior comparison", type="primary"):
    out, dens = [], {}
    for prior in ("fixed_normal", "estimated_normal", "estimated_empirical"):
        c = config(prior=prior, grid="equal")
        try:
            r = fit(d, c, record=False)
        except ValueError as e:
            st.error(f"{prior}: {e}")
            continue
        ic = information_criteria(r, d.N)
        row = dict(prior=P.PRIOR_NAMES[prior], converged=r.converged, cycles=r.cycles, loglik=r.loglik,
                   parameters=ic["parameters"], AIC=ic["AIC"], BIC=ic["BIC"], mean=r.group_moments[0, 0],
                   sd=r.group_moments[0, 1])
        rec = item_recovery(d, r)
        if len(rec):
            free = rec[~rec["fixed"]]
            row["item RMSE vs truth"] = float(np.sqrt(np.mean(free["error"] ** 2)))
        out.append(row)
        dens[P.PRIOR_NAMES[prior]] = r.weights[0]
        xx = r.nodes[0]
    ss.prior_comparison = (pd.DataFrame(out), dens, xx, d.identity)
if "prior_comparison" in ss and ss.prior_comparison[3] == d.identity:
    frame, dens, xx, _ = ss.prior_comparison
    table(frame, digits=4)
    if d.truth:
        from core.simulation import population_weights
        gt = d.truth["groups"][0]
        dens["generating distribution"] = population_weights(gt["shape"], xx, gt["mean"], gt["sd"])
    lines(xx, dens, "Fitted latent distributions (group 1)", yl="mass", key="cmp_dens")
    interpret("A lower AIC/BIC for the empirical prior, together with a visibly non-normal fitted histogram, signals "
              "latent-distribution misspecification. When the truth is normal the histogram adds Q − 3 parameters for "
              "little likelihood gain (BIC penalises it) and its shape is noisy; with small samples it can become "
              "spiky — nearly empty bins — because the unpenalised histogram MLE tends towards a few support points.")

four_views(
    "MMLE integrates ability out: P(y_i) = Σ_q π_q L_i(θ_q). The masses π_q are the latent distribution. **A · fixed "
    "normal** uses N(μ, σ²) masses that never change. **B · estimated normal** updates μ and σ for groups whose moments "
    "are identified. **C · fixed empirical** uses supplied historical masses. **D · estimated empirical** treats every "
    "π_q as a parameter and updates them from the posteriors. When no anchors fix the metric, the reference "
    "distribution's mean (and, for GPCM/GRM, variance) must be fixed — so a free histogram must satisfy those "
    "moment constraints exactly.",
    ["$$w_{iq}=\\frac{\\pi_qL_i(\\theta_q)}{\\sum_s\\pi_sL_i(\\theta_s)},\\qquad n_q=\\sum_iw_{iq},\\qquad Q_{\\text{prior}}(\\pi)=\\sum_q n_q\\log\\pi_q$$",
     "Unconstrained maximiser: π_q = n_q / N.",
     "Constrained maximiser (Σπ = 1, Σπθ = μ₀, Σπ(θ−μ₀)² = σ₀²): Lagrange conditions give",
     "$$\\pi_q=\\frac{n_q/N}{1+\\lambda_1(\\theta_q-\\mu_0)+\\lambda_2[(\\theta_q-\\mu_0)^2-\\sigma_0^2]}$$",
     "with λ minimising the convex dual −Σ_q (n_q/N) log(1 + λᵀg_q) (Newton steps with backtracking). This is an exact "
     "M-step, so EM stays monotone. irtQ's Woods (2007) EmpHist instead takes n/N and re-standardises by interpolation "
     "with exponential tails — a generalised step whose result satisfies the moments approximately and need not "
     "increase the likelihood.",
     "Estimated normal on a fixed grid: π_q(μ, σ) ∝ φ((θ_q − μ)/σ). Exact M-step: maximise Σ n_q log π_q(μ, σ) "
     "numerically. 'Posterior moments' (TAM/irtQ) sets μ, σ to the posterior moments — exact only for a continuous "
     "normal, a close generalised step on a fine grid."],
    None, [P.empirical_unconstrained, P.empirical_constrained, P.normal_exact, P.normal_moments, P.woods_rescale])
