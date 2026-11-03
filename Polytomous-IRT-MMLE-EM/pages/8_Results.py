import numpy as np
import pandas as pd
import streamlit as st

from components.ui import (bars, data, full_precision, guided, interpret, latest_fit, lines, page_setup, person, table)
from core.em import DESIGN_NAMES, inspect_cycle
from core.models import probs
from core.priors import PRIOR_NAMES
from core.quadrature import moments

page_setup("8 · Results", "Item, group and prior estimates, convergence trajectories — and every intermediate quantity "
                          "of any iteration.")
guided(["Check the status and the reason the run stopped (convergence or OEM's single cycle).",
        "Read item estimates: anchors are marked fixed and verified unchanged.",
        "Look at the trajectories: ℓ must rise every cycle for exact M-steps.",
        "Open the iteration inspector and pick a cycle: posteriors, counts, updates, anchors, ℓ."],
       "For empirical priors, convergence should be judged on…",
       ["item changes and changes in the latent distribution", "item changes only", "the log-likelihood only",
        "the number of cycles"], "item changes and changes in the latent distribution",
       "A histogram can keep moving after item parameters have settled; the lab requires both (plus group moments).",
       key="results")
ss = st.session_state
d = data()
res = latest_fit()
if res is None:
    st.info("No calibration for the active dataset yet. Run one on **7 · Calibration Designs**.")
    st.stop()
cfg = res.config
st.markdown(f"**{cfg.model} · {DESIGN_NAMES[cfg.design]} · {PRIOR_NAMES[cfg.prior]}**"
            + (f" · {'MWU–MEM' if cfg.fipc_method == 'MEM' else 'NWU–OEM'}" if cfg.anchored else "")
            + f" · Q = {res.nodes.shape[1]} ({cfg.grid}) · D = {cfg.D}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Status", "Converged" if res.converged else ("One cycle (OEM)" if "OEM" in res.reason else "Not converged"))
c2.metric("Cycles", res.cycles)
c3.metric("Marginal ℓ", f"{res.loglik:.4f}")
c4.metric("Free parameters", res.n_parameters())
c5.metric("Seconds", f"{res.seconds:.2f}")
st.caption(res.reason)
for w in res.warnings:
    st.warning(w)
for n in res.notes:
    st.info(n)

t_items, t_groups, t_conv, t_iter = st.tabs(["Item estimates", "Group & prior estimates", "Convergence",
                                             "Iteration inspector"])
with t_items:
    frame = res.item_table()
    table(frame, digits=4)
    full_precision(frame, "item_estimates.csv", "items")
    if res.fixed.any():
        diffs = [np.max(np.abs(it.vector() - d.anchors[it.id].vector())) for it, f in zip(res.items, res.fixed) if f]
        st.success(f"Anchor check: largest |final − bank| over {len(diffs)} anchors = {max(diffs):.1e} (exactly fixed).")
    if res.se:
        st.caption(f"Standard errors: {res.se.get('status')}.")
    elif cfg.se is False:
        st.caption("Tick 'Compute standard errors' in the sidebar (Convergence) and re-run to add observed-information SEs.")

with t_groups:
    mm = res.group_moments
    im = np.array([moments(res.nodes[g], res.initial_weights[g]) for g in range(len(mm))])
    gt = pd.DataFrame(dict(group=res.group_names, start_mean=im[:, 0], start_sd=im[:, 1], mean=mm[:, 0], sd=mm[:, 1],
                           distribution=[u.describe(res.group_names) for g in range(len(mm)) for u in res.units if g in u.groups]))
    if res.implied_weights is not None:
        imp = np.array([moments(res.nodes[g], res.implied_weights[g]) for g in range(len(mm))])
        gt["OEM implied mean"] = imp[:, 0]
        gt["OEM implied sd"] = imp[:, 1]
    table(gt, digits=4)
    g = st.selectbox("Group", range(len(mm)), format_func=lambda g: res.group_names[g], key="res_g")
    series = {"start": res.initial_weights[g], "final": res.weights[g]}
    if res.implied_weights is not None:
        series["OEM implied (reported, not used)"] = res.implied_weights[g]
    if d.truth:
        from core.simulation import population_weights
        tg = d.truth["groups"][g]
        series["generating"] = population_weights(tg["shape"], res.nodes[g], tg["mean"], tg["sd"],
                                                  d.truth["spec"].get("discrete_nodes"), d.truth["spec"].get("discrete_weights"))
    lines(res.nodes[g], series, f"Latent distribution · {res.group_names[g]}", yl="mass π_q", key="res_w")
    full_precision(pd.DataFrame({"theta": res.nodes[g], **{k: v for k, v in series.items()}}), "weights.csv", "weights")

with t_conv:
    h = pd.DataFrame([{k: v for k, v in e.items() if not isinstance(v, (list, np.ndarray)) and k not in ("prior_info",)}
                      for e in res.history])
    c1, c2 = st.columns(2)
    with c1:
        lines(h["cycle"].to_numpy(), {"ℓ": h["loglik"].to_numpy()}, "Marginal log-likelihood (start of each cycle)",
              xl="cycle", yl="ℓ", key="ll")
    with c2:
        lines(h["cycle"].to_numpy(), {"item": np.log10(np.maximum(h["item_change"], 1e-16)),
                                      "prior mass": np.log10(np.maximum(h["weight_change"], 1e-16)),
                                      "group moments": np.log10(np.maximum(h["group_change"], 1e-16))},
              "log₁₀ largest change per cycle", xl="cycle", yl="log10", key="chg")
    if h["ll_decrease"].any():
        st.error(f"ℓ decreased in cycle(s) {h.loc[h['ll_decrease'], 'cycle'].tolist()}.")
    else:
        st.success("ℓ never decreased (beyond rounding) across cycles.")
    j = st.selectbox("Parameter trajectory for item", range(len(res.items)), format_func=lambda j: res.ids[j])
    if "items_after" in res.history[0]:
        traj = np.array([e["items_after"][j].vector() for e in res.history])
        lines(h["cycle"].to_numpy(), {lab: traj[:, k] for k, lab in enumerate(res.items[j].labels())
                                      if not (lab == "a" and res.items[j].model in ("PCM", "RSM"))},
              f"Estimates by cycle · {res.ids[j]}", xl="cycle", yl="value", key="traj")
    table(h.drop(columns=[c for c in h.columns if c.startswith("q_item")], errors="ignore"), digits=6,
          sci=("item_change", "weight_change", "group_change"), height=300)

with t_iter:
    t = st.slider("Cycle", 1, res.cycles, 1)
    ins = inspect_cycle(d, res, t)
    e = ins["entry"]
    st.markdown(f"**Cycle {t}** · E-step used **{e['estep_items']}** · ℓ at start {e['loglik']:.6f}"
                + (f" (Δ {e['ll_change']:+.3e})" if e["ll_change"] is not None else ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Δ item", f"{e['item_change']:.2e}")
    c2.metric("Δ prior mass", f"{e['weight_change']:.2e}")
    c3.metric("Δ group moments", f"{e['group_change']:.2e}")
    s1, s2, s3, s4, s5, s6 = st.tabs(["Category probabilities", "Nodes & weights", "Examinee posterior",
                                      "Expected counts", "Item & anchor updates", "Prior & group updates"])
    with s1:
        j = st.selectbox("Item", range(d.J), format_func=lambda j: d.ids[j], key="ins_j")
        it = e["items_before"][j]
        P = probs(it, res.nodes[0], cfg.D)
        f = pd.DataFrame(P, columns=[f"P(k={k})" for k in range(it.ncat)])
        f.insert(0, "theta", res.nodes[0])
        table(f, digits=6, height=300)
    with s2:
        f = pd.DataFrame({"theta": res.nodes[0], **{f"π {res.group_names[g]}": e["weights_before"][g]
                                                   for g in range(len(res.group_names))}})
        table(f, digits=6, height=300)
    with s3:
        i = st.number_input("Examinee", 0, d.N - 1, person(), key="ins_i")
        es = ins["estep"]
        g = d.group[i]
        f = pd.DataFrame(dict(theta=res.nodes[g], prior=e["weights_before"][g], log_likelihood=es["logL"][i],
                              posterior=es["post"][i]))
        st.caption(f"Responses: {['·' if np.isnan(v) else int(v) for v in d.Y[i]]} · group {d.group_names[g]} · "
                   f"Σ w = {es['post'][i].sum():.12f} · EAP = {es['post'][i] @ res.nodes[g]:.4f}")
        lines(res.nodes[g], {"prior": e["weights_before"][g], "posterior": es["post"][i]}, "Prior vs posterior",
              yl="mass", key="ins_post")
        table(f, digits=6, height=260)
    with s4:
        j = st.selectbox("Item ", range(d.J), format_func=lambda j: d.ids[j], key="ins_jc")
        g = st.selectbox("Group ", range(d.G), format_func=lambda g: d.group_names[g], key="ins_gc")
        C = ins["counts"][j][g]
        bars(np.round(res.nodes[g], 2), {f"k={k}": C[:, k] for k in range(C.shape[1])},
             f"C_gjkq · {d.ids[j]} · {d.group_names[g]}", mode="stack", key="ins_c")
        st.caption(f"Σ_q C = {np.round(C.sum(axis=0), 6).tolist()} · observed {d.counts(j, g).tolist()}")
    with s5:
        rows = []
        for a, b, f in zip(e["items_before"], e["items_after"], res.fixed):
            for lab, x0, x1 in zip(a.labels(), a.vector(), b.vector()):
                if lab == "a" and a.model in ("PCM", "RSM"):
                    continue
                rows.append(dict(item=a.id, status="fixed anchor" if f else "free", parameter=lab, before=x0,
                                 after=x1, change=x1 - x0))
        table(pd.DataFrame(rows), digits=6, height=300)
        st.markdown("**Inner optimiser**")
        if e["statuses"]:
            table(pd.DataFrame(e["statuses"]), digits=6)
        else:
            st.caption("No free items.")
    with s6:
        mb = np.array([moments(res.nodes[g], e["weights_before"][g]) for g in range(d.G)])
        ma = np.array([moments(res.nodes[g], e["weights_after"][g]) for g in range(d.G)])
        table(pd.DataFrame(dict(group=d.group_names, mean_before=mb[:, 0], mean_after=ma[:, 0], sd_before=mb[:, 1],
                                sd_after=ma[:, 1])), digits=6)
        for info in e["prior_info"]:
            st.caption(f"{', '.join(d.group_names[g] for g in info['groups'])}: {info['method']} · Q_prior "
                       f"{info['q_before']:.5f} → {info['q_after']:.5f}")
        if not e["prior_info"]:
            st.caption("No distribution updated in this cycle (fixed prior, or NWU–OEM).")
        g = st.selectbox("Group  ", range(d.G), format_func=lambda g: d.group_names[g], key="ins_gp")
        bars(np.round(res.nodes[g], 2), {"before": e["weights_before"][g], "average posterior": ins["n_g"][g] / ins["n_g"][g].sum(),
                                         "after": e["weights_after"][g]}, "Prior masses in this cycle", key="ins_w")

interpret("Fixed anchors never change, but they appear in every E-step (and alone in the first E-step of FIPC). "
          "Group means and SDs are reported on the metric fixed either by the reference constraints or by the anchors. "
          "Standard errors, when computed, use the observed information with the latent distribution held at its "
          "estimate, so they are conditional on that distribution.")
