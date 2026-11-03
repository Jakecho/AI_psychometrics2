import numpy as np
import pandas as pd
import streamlit as st

from components.ui import (data, four_views, full_precision, guided, interpret, lines, page_setup, phase_cards, table)
from core import em
from core import walkthrough as W
from core.data import walkthrough_dataset
from core.models import MODEL_NAMES, MODELS, Item, make_item
from core.priors import PRIOR_NAMES, PRIOR_TYPES

page_setup("6 · EM Walkthrough", "Thirteen clickable steps through one MMLE–EM cycle. Every number is produced by the "
                                 "engine functions that run the full calibrations.")
ss = st.session_state
guided(["Read the 5 × 3 response matrix and the three-point Gauss–Hermite prior.",
        "Step through: probabilities → likelihoods → posterior → expected counts → M-steps → log-likelihood.",
        "At step 13 accept the updates and run the next cycle; the history table shows the climb.",
        "Change the model or prior to see which steps change and which do not."],
       "After the E-step, Σ_q C_jkq (summed over nodes) equals…",
       ["the observed number of category-k responses to item j", "1", "N", "the posterior mean"],
       "the observed number of category-k responses to item j",
       "Each examinee's posterior sums to one, so summing I(Y_ij = k) w_iq over q returns the observed count.", key="wt")

c1, c2, c3 = st.columns(3)
model = c1.selectbox("Model", MODELS, format_func=lambda m: f"{m} · {MODEL_NAMES[m]}", key="wt_model")
prior = c2.selectbox("Ability prior", PRIOR_TYPES, format_func=PRIOR_NAMES.get, key="wt_prior")
source = c3.radio("Data", ["Teaching data (5 × 3)", "Active dataset"], horizontal=True, key="wt_source")
d = walkthrough_dataset() if source.startswith("Teaching") else data()
if d.G > 1 or d.anchors:
    d = d.subset_groups([0]) if d.G > 1 else d
cfg = W.walkthrough_config(model, prior)
if source == "Active dataset":
    st.caption("Active dataset: the walkthrough uses its first group without anchors (single-group design) and shows "
               "the first 10 examinees in per-examinee tables.")
    from core.data import Dataset
    d = Dataset(d.name, d.Y, d.group, d.group_names, d.specs, anchors={}, truth=d.truth)
key = (model, prior, d.identity)
if ss.get("wt_key") != key:
    ss.wt_key = key
    ss.wt_items = W.default_start(d, model)
    ss.wt_weights = None
    ss.wt_hist = []
    ss.wt_step = 1
show_n = min(d.N, 10)

steps = W.STEPS
c = st.columns([1, 1, 1, 1, 2])
if c[0].button("◀ Previous", disabled=ss.wt_step <= 1):
    ss.wt_step -= 1
    st.rerun()
if c[1].button("Next ▶", disabled=ss.wt_step >= len(steps)):
    ss.wt_step += 1
    st.rerun()
if c[2].button("Reset"):
    ss.wt_key = None
    st.rerun()
jump = c[3].selectbox("Go to step", range(1, len(steps) + 1), index=ss.wt_step - 1, label_visibility="collapsed",
                      format_func=lambda k: f"{k}. {steps[k - 1]}")
if jump != ss.wt_step:
    ss.wt_step = jump
    st.rerun()
c[4].markdown(f"**Cycle {len(ss.wt_hist) + 1}** · step {ss.wt_step} of {len(steps)}")
phase_cards([s.replace(" ", " ", 1) for s in steps], ss.wt_step - 1)

try:
    cyc = W.cycle(d, cfg, ss.wt_items, ss.wt_weights, ss.wt_hist[-1]["ll_after"] if ss.wt_hist else None)
except ValueError as e:
    st.error(str(e))
    st.stop()
nodes, weights = cyc["nodes"], cyc["weights"]
step = ss.wt_step
st.subheader(f"Step {step} · {steps[step - 1]}")

if step == 1:
    table(W.table_responses(d).head(show_n), digits=0)
    st.caption(f"{d.N} examinees × {d.J} items, categories 0…{max(d.ncats) - 1}. Every category of every item is "
               "observed at least once, so every step parameter is identified by these data.")
elif step == 2:
    st.caption("Edit the starting values (only before the first cycle). RSM: location b and shared steps τ.")
    frame = W.table_items(ss.wt_items)
    edited = st.data_editor(frame, hide_index=True, disabled=bool(ss.wt_hist), key=f"wt_edit_{key}")
    if not ss.wt_hist and st.button("Apply edited starting values"):
        try:
            new = []
            for _, r in edited.iterrows():
                if model == "RSM":
                    K = d.specs[0].ncat - 1
                    tau = np.array([r[f"τ{h + 1}"] for h in range(K)], float)
                    new.append(Item(r["item"], "RSM", 1.0, (), b=float(r["b (location)"]), tau=tuple(tau - tau.mean())))
                else:
                    K = int(r["categories"]) - 1
                    lab = "b" if model == "GRM" else "δ"
                    new.append(make_item(r["item"], model, float(r["a"]), tuple(float(r[f"{lab}{h + 1}"]) for h in range(K))))
            ss.wt_items = new
            st.rerun()
        except (ValueError, KeyError) as e:
            st.error(f"Invalid starting values: {e}")
elif step == 3:
    table(W.table_prior(nodes, weights, d.group_names), digits=6)
    st.caption({"fixed_normal": "Three-point Gauss–Hermite rule for N(0, 1): θ = (−√3, 0, √3), π = (1/6, 2/3, 1/6). A "
                                "low-order teaching approximation (page 5 shows how the error falls with more nodes).",
                "estimated_normal": "Fixed grid θ = (−√3, 0, √3) with masses ∝ φ(θ_q; μ, σ). One group, no anchors: the "
                                    "mean is fixed at 0; " + ("σ is estimated (Rasch)." if model in ("PCM", "RSM")
                                                              else "σ is fixed at 1 (slopes estimated), so nothing moves."),
                "fixed_empirical": "Fixed masses (1/6, 2/3, 1/6) on the same three nodes, never updated.",
                "estimated_empirical": "Masses on three nodes, updated each cycle subject to identification: mean 0"
                                       + ("." if model in ("PCM", "RSM") else " and variance 1 — with three nodes these "
                                          "two constraints plus Σπ = 1 determine the masses exactly (no freedom).")}[prior])
elif step == 4:
    table(W.table_category_probs(cyc["items"], nodes, cfg.D), digits=6)
    st.caption("Each row sums to one (column 'sum').")
elif step == 5:
    table(W.table_likelihoods(d, cyc["items"], cyc, cfg.D, show_n), digits=6, sci=("likelihood",))
    st.caption("L_i(θ_q) = Π_{j∈O_i} P_{j, y_ij}(θ_q): the product of the probabilities of the categories actually "
               "chosen. Computed on the log scale and exponentiated here for display.")
elif step == 6:
    table(W.table_posterior(d, cyc, show_n).drop(columns=["posterior"]), digits=6,
          sci=("likelihood", "unnormalised", "marginal"))
    st.caption("Unnormalised posterior π_q · L_i(θ_q); 'marginal' = Σ_s π_s L_i(θ_s) = P(y_i), the examinee's "
               "contribution to the marginal likelihood.")
elif step == 7:
    post = cyc["estep"]["post"][:show_n]
    f = pd.DataFrame(post, columns=[f"θ={t:.4f}" for t in nodes[0]])
    f.insert(0, "examinee", [f"P{i + 1}" for i in range(show_n)])
    f["Σ_q w_iq"] = post.sum(axis=1)
    f["EAP"] = post @ nodes[0]
    table(f, digits=6)
    full_precision(f, "posterior.csv", "post")
elif step == 8:
    tc = W.table_counts(d, cyc)
    table(tc, digits=6, height=320)
    st.markdown("**Reconciliation with observed counts**")
    table(W.reconcile_counts(d, cyc), digits=10, sci=("difference",))
    c1, c2 = st.columns(2)
    j = c1.selectbox("Item", range(d.J), format_func=lambda j: d.ids[j], key="wt_j")
    k = c2.selectbox("Category", range(d.specs[j].ncat), key="wt_k")
    st.markdown(f"Contributions I(Y_i{j + 1} = {k}) · w_iq to C_{{{d.ids[j]},{k},q}}")
    table(W.count_contributions(d, cyc, j, k, show_n), digits=6)
elif step == 9:
    rows = []
    for a, b in zip(cyc["items"], cyc["new_items"]):
        for lab, x0, x1 in zip(a.labels(), a.vector(), b.vector()):
            if lab == "a" and model in ("PCM", "RSM"):
                continue
            rows.append(dict(item=a.id, parameter=lab, before=x0, after=x1, change=x1 - x0))
    table(pd.DataFrame(rows), digits=6)
    st.markdown("**Inner optimiser (L-BFGS-B, analytic gradients)**")
    table(pd.DataFrame(cyc["statuses"]).drop(columns=["message"]), digits=6)
    st.caption({"PCM": "PCM: each item's steps maximise Σ_qk C log P separately.",
                "RSM": "RSM: one joint problem — all locations b_j and the shared steps τ (Σ τ = 0) together.",
                "GPCM": "GPCM: log a and steps per item (log a keeps a > 0).",
                "GRM": "GRM: log a, b₁ and log-gaps (keeps b₁ < b₂ < …)."}[model])
elif step == 10:
    if cyc["prior_info"]:
        for info in cyc["prior_info"]:
            st.write(f"**{info['method']}** · Q_prior {info['q_before']:.6f} → {info['q_after']:.6f}")
    else:
        st.info("This prior is fixed: the masses do not change.")
    f = pd.DataFrame(dict(theta=nodes[0], before=weights[0], posterior_mass=cyc["n_g"][0], average_posterior=cyc["n_g"][0] / d.N,
                          after=cyc["new_weights"][0]))
    table(f, digits=6)
    table(W.table_moments(nodes, weights, cyc["new_weights"], d.group_names), digits=6)
elif step == 11:
    table(pd.DataFrame([dict(quantity="ℓ at start of cycle", value=cyc["ll_before"]),
                        dict(quantity="ℓ after both M-steps", value=cyc["ll_after"]),
                        dict(quantity="change", value=cyc["ll_after"] - cyc["ll_before"])]), digits=8)
    st.caption("ℓ = Σ_i log Σ_q π_q L_i(θ_q) (log-sum-exp). EM with exact or improving M-steps never decreases ℓ.")
    if cyc["ll_after"] < cyc["ll_before"] - 1e-9:
        st.error("The log-likelihood decreased — unexpected for exact M-steps.")
elif step == 12:
    tol = em.FitConfig().tol
    est = any(u.estimated for u in cyc["units"])
    ok = cyc["item_change"] < tol and (not est or cyc["weight_change"] < em.FitConfig().prior_tol)
    table(pd.DataFrame([dict(criterion="largest |Δ item parameter|", value=cyc["item_change"], tolerance=tol),
                        dict(criterion="largest |Δ prior mass|", value=cyc["weight_change"],
                             tolerance=em.FitConfig().prior_tol if est else np.nan)]), digits=8, sci=("value", "tolerance"))
    (st.success if ok else st.warning)("Converged." if ok else "Not yet converged — continue to the next cycle.")
elif step == 13:
    st.write("Accept the updated item parameters and prior masses as the starting point of the next cycle.")
    if st.button("Start the next cycle ▶▶", type="primary"):
        ss.wt_hist.append(dict(cycle=len(ss.wt_hist) + 1, ll_before=cyc["ll_before"], ll_after=cyc["ll_after"],
                               item_change=cyc["item_change"], weight_change=cyc["weight_change"]))
        ss.wt_items, ss.wt_weights = cyc["new_items"], cyc["new_weights"]
        ss.wt_step = 4
        st.rerun()
    if st.button("Run 50 more cycles"):
        for _ in range(50):
            c2 = W.cycle(d, cfg, ss.wt_items, ss.wt_weights)
            ss.wt_hist.append(dict(cycle=len(ss.wt_hist) + 1, ll_before=c2["ll_before"], ll_after=c2["ll_after"],
                                   item_change=c2["item_change"], weight_change=c2["weight_change"]))
            ss.wt_items, ss.wt_weights = c2["new_items"], c2["new_weights"]
        st.rerun()

if ss.wt_hist:
    st.subheader("Cycle history")
    h = pd.DataFrame(ss.wt_hist)
    c1, c2 = st.columns([2, 3])
    with c1:
        table(h, digits=6, sci=("item_change", "weight_change"))
    with c2:
        lines(h["cycle"].to_numpy(), {"ℓ after cycle": h["ll_after"].to_numpy()}, "Marginal log-likelihood", xl="cycle",
              yl="ℓ", key="wt_ll")

interpret("The teaching data are tiny, so estimates move a lot and some parameters may approach the optimizer bounds "
          "after many cycles — with three nodes the latent distribution is a crude three-point approximation. The "
          "point is the arithmetic: every table here is generated by `core.em` (estep, expected_counts, mstep_items, "
          "mstep_prior, marginal_loglik), the same code that runs full calibrations on page 7.")

four_views(
    "EM alternates between computing what each examinee's ability probably is given current parameters (E-step) "
    "and re-estimating parameters as if those fractional abilities were observed (M-step).",
    ["$$w_{iq}=\\frac{\\pi_q\\prod_{j\\in O_i}P_{j,y_{ij}}(\\theta_q)}{\\sum_s\\pi_s\\prod_{j\\in O_i}P_{j,y_{ij}}(\\theta_s)}\\qquad C_{jkq}=\\sum_i I(Y_{ij}=k)\\,w_{iq}$$",
     "$$Q_{\\text{item}}(\\xi)=\\sum_{j,k,q}C_{jkq}\\log P_{jk}(\\theta_q;\\xi)\\qquad \\ell=\\sum_i\\log\\sum_q\\pi_q\\prod_{j\\in O_i}P_{j,y_{ij}}(\\theta_q)$$"],
    None, [em.estep, em.expected_counts, em.mstep_items, em.mstep_prior])
