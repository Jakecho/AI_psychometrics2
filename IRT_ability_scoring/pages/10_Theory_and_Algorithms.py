import inspect

import numpy as np
import pandas as pd
import streamlit as st

from importlib import import_module

# Submodules are loaded with import_module: `scoring.wle` etc. are shadowed by the functions of the same name.
base = import_module("irt_models.base")
dich = import_module("irt_models.dichotomous")
grm = import_module("irt_models.grm")
pc = import_module("irt_models.partial_credit")
eap_mod = import_module("scoring.eap")
eapsum_mod = import_module("scoring.eapsum")
solvers = import_module("scoring.solvers")
tcc_mod = import_module("scoring.tcc")
wle_mod = import_module("scoring.wle")
from components.ui import download, interpret, labels, page_setup, settings, table
from irt_models import category_table, item_table
from r_bridge.compare import compare_all
from scoring import METHODS, UNCERTAINTY, mle, node_table, rows, score_all
from scoring.wle import adjustment_curves
from simulation import EXAMPLES, simulate

page_setup("10 · Theory & Algorithms", "Equations, derivations, the executed code, verified numerical examples and references.")
s = settings()
tabs = st.tabs(["Estimator summary", "Derivations", "Algorithms & code", "Numerical examples A–D", "Validation", "References"])

with tabs[0]:
    table([dict(method="MLE", objective="max ℓ(θ)", uses="full pattern", prior="no", uncertainty=UNCERTAINTY["MLE"],
                extreme_patterns="no finite estimate", reference="irtQ ML · mirt ML · TAM MLE"),
           dict(method="EAP", objective="posterior mean", uses="full pattern", prior="yes", uncertainty=UNCERTAINTY["EAP"],
                extreme_patterns="finite", reference="irtQ EAP · mirt EAP"),
           dict(method="MAP", objective="posterior mode", uses="full pattern", prior="yes", uncertainty=UNCERTAINTY["MAP"],
                extreme_patterns="finite", reference="irtQ MAP · mirt MAP"),
           dict(method="WLE", objective="root of ℓ′ + J/(2I)", uses="full pattern", prior="no", uncertainty=UNCERTAINTY["WLE"],
                extreme_patterns="finite (logistic items)", reference="irtQ WL · TAM WLE (mirt: L·√I)"),
           dict(method="TCC", objective="root of T(θ) = r", uses="summed score", prior="no", uncertainty=UNCERTAINTY["TCC"],
                extreme_patterns="no finite estimate", reference="irtQ INV.TCC"),
           dict(method="EAPsum", objective="E(θ | R = r)", uses="summed score", prior="yes", uncertainty=UNCERTAINTY["EAPsum"],
                extreme_patterns="finite", reference="irtQ EAP.SUM · mirt EAPsum")])
    st.markdown("**Notation.** θ ability; j items (O_i: observed by examinee i); k = 0..m_j categories scoring k points; "
                "P_jk(θ) category probability; P′, P″ its θ-derivatives; ℓ log-likelihood; I(θ) = Σ_j Σ_k P′²/P "
                "information; J(θ) = Σ_j Σ_k P′P″/P; T(θ) = Σ_j Σ_k k P_jk test characteristic curve; g(θ) = N(μ, σ²) "
                "prior; (θ_q, w_q) quadrature nodes and weights; D scaling constant.")

with tabs[1]:
    st.markdown("#### Score and information for any item")
    st.latex(r"\frac{d}{d\theta}\log P_{jk}=\frac{P'_{jk}}{P_{jk}},\qquad \frac{d^2}{d\theta^2}\log P_{jk}=\frac{P''_{jk}}{P_{jk}}-\Big(\frac{P'_{jk}}{P_{jk}}\Big)^2,\qquad \sum_k P'_{jk}=\sum_kP''_{jk}=0")
    st.latex(r"E\big[-\ell_j''\big]=\sum_kP_{jk}\Big[\Big(\tfrac{P'_{jk}}{P_{jk}}\Big)^2-\tfrac{P''_{jk}}{P_{jk}}\Big]=\sum_k\frac{P_{jk}'^2}{P_{jk}}=I_j(\theta)")
    st.markdown("#### Model-specific derivatives used in the code")
    st.latex(r"\text{4PL: }P'=(d-c)Da\,\Psi(1-\Psi),\quad P''=(d-c)(Da)^2\Psi(1-\Psi)(1-2\Psi)")
    st.latex(r"\text{PCM/GPCM/RSM: }\frac{d\log P_k}{d\theta}=Da(k-E),\quad\frac{d^2\log P_k}{d\theta^2}=-(Da)^2\mathrm{Var}")
    st.latex(r"\text{GRM: }\frac{d\log P_k}{d\theta}=Da(1-P^*_k-P^*_{k+1}),\quad \frac{d^2\log P_k}{d\theta^2}=-(Da)^2\big[P^*_kQ^*_k+P^*_{k+1}Q^*_{k+1}\big]")
    st.markdown("#### Warm's correction")
    st.latex(r"E(\hat\theta_{ML}-\theta)=\frac{J(\theta)}{2I(\theta)^2}+o(n^{-1})\ \Rightarrow\ U_W(\theta)=\ell'(\theta)+\frac{J(\theta)}{2I(\theta)}")
    st.latex(r"I'(\theta)-J(\theta)=\sum_j\sum_kP_{jk}\,\frac{d\log P_{jk}}{d\theta}\,\frac{d^2\log P_{jk}}{d\theta^2}\quad(=0\text{ for Rasch, 2PL, PCM, GPCM, RSM and GRM items})")
    st.markdown("#### Summed-score methods")
    st.latex(r"\text{Rasch family: }\ell'(\theta)=D\big(r-T(\theta)\big)\Rightarrow\hat\theta_{ML}=\hat\theta_{TCC}")
    st.latex(r"P(R=r\mid\theta)=f_J(r\mid\theta),\quad f_j(s\mid\theta)=\sum_kf_{j-1}(s-k\mid\theta)P_{jk}(\theta)")
    st.markdown("#### Bayesian methods")
    st.latex(r"p(\theta\mid\mathbf y)\propto L(\theta)g(\theta),\quad \hat\theta_{EAP}=E(\theta\mid\mathbf y),\quad \hat\theta_{MAP}=\arg\max p(\theta\mid\mathbf y)")

with tabs[2]:
    st.markdown("Every function below is the code the app executes (shown with `inspect.getsource`).")
    for title, fns in [("Item models", [dich.log_derivs, pc.log_derivs, grm.log_probs, grm.log_derivs, base.pattern_terms]),
                       ("Solvers", [solvers.maximize, solvers.find_root]),
                       ("MLE / MAP / WLE / TCC", [mle, wle_mod.equation, tcc_mod.tcc_inverse]),
                       ("Quadrature", [eap_mod.quadrature, eap_mod.posterior, eapsum_mod.score_distribution])]:
        st.markdown(f"**{title}**")
        for fn in fns:
            with st.expander(f"`{fn.__module__}.{fn.__name__}`"):
                st.code(inspect.getsource(fn), language="python")

with tabs[3]:
    labels("edu", "demo")
    st.caption("Every number below is computed now by the app's functions with the sidebar settings "
               f"(prior N({s.prior_mean:g}, {s.prior_sd:g}²), {s.n_nodes} {s.quad_rule} nodes).")
    for name, fn in EXAMPLES.items():
        ex = fn()
        y = ex.Y[0]
        with st.expander(f"{ex.name} — pattern {ex.person_ids[0]}", expanded=name.startswith("Example A")):
            st.markdown(ex.description)
            st.markdown("**Category probabilities at θ = 0**")
            table(pd.DataFrame(category_table(ex.items, 0.0)), 5, height=min(38 * (sum(i.n_cat for i in ex.items) + 1) + 3, 300))
            r = mle(ex.items, y[None, :], s, record=True)
            st.markdown(f"**MLE Newton–Raphson from θ⁽⁰⁾ = {s.start:g}** (status: {r.status[0]})")
            h = pd.DataFrame(rows(r.history, 0))[["iteration", "theta", "obj", "grad", "hess", "step", "theta_new"]]
            table(h.rename(columns=dict(obj="loglik", grad="score", hess="second_derivative")), 6)
            th = r.theta[0]
            if np.isfinite(th):
                st.markdown(f"**Item terms at the MLE θ̂ = {th:.5f}**")
                table(pd.DataFrame(item_table(ex.items, y, th)), 5)
            tab, m, psd = node_table(ex.items, y, s)
            top = pd.DataFrame(tab).nlargest(5, "posterior_weight")[["node", "prior_weight", "likelihood", "posterior_weight"]]
            st.markdown(f"**EAP**: {m:.5f} (PSD {psd:.5f}); the five nodes with the largest posterior weight:")
            table(top, 6)
            if np.isfinite(th):
                a = adjustment_curves(ex.items, y, np.array([th]))
                st.markdown(f"**WLE adjustment at the MLE**: J/(2I) = {a['warm'][0]:.5f} (I = {a['info'][0]:.5f}, J = {a['J'][0]:.5f})")
            res = score_all(ex.items, ex.Y, s)
            allm = pd.DataFrame(dict(pattern=ex.person_ids, **{m_: res[m_].theta for m_ in METHODS}))
            st.markdown("**All methods, all patterns**")
            table(allm, 4)
    interpret("Example A shows that Rasch patterns with the same raw score share every estimate. Example B is the 2PL "
              "counterexample (±0.4196 from the MLE, 0 from TCC inversion). Example C has a reversed PCM step. Example D "
              "starts from the GRM item whose probabilities at θ = 0 are (0.26894, 0.46212, 0.26894).")

with tabs[4]:
    labels("ref", "edu")
    st.markdown("Runs irtQ, mirt and TAM on examples A–D and a mixed-format simulation (3PL + GRM, 120 examinees) and "
                "compares with the Python results under matched settings. The recorded results of the full validation "
                "are in `docs/VALIDATION.md`.")
    if st.button("Run the validation suite", type="primary"):
        out = []
        sets = [fn() for fn in EXAMPLES.values()] + [simulate(120, seed=3, n_dich=12, dich_model="3PL", n_poly=4, poly_model="GRM")]
        with st.spinner("Running Rscript for each dataset…"):
            for ds in sets:
                _, r = compare_all(ds.items, ds.Y, s)
                for row in r:
                    row["dataset"] = ds.name
                out += r
        st.session_state["theory_val"] = (out, s.as_dict())
    saved = st.session_state.get("theory_val")
    if saved and saved[1] == s.as_dict():
        frame = pd.DataFrame(saved[0])
        frame = frame[["dataset"] + [c for c in frame.columns if c != "dataset"]]
        table(frame, 6, height=460)
        download(frame, "validation.csv")
        counts = frame["result"].value_counts().to_dict()
        st.markdown(" · ".join(f"**{k}**: {v}" for k, v in counts.items()))

with tabs[5]:
    st.markdown("""
- Baker, F. B., & Kim, S.-H. (2004). *Item response theory: Parameter estimation techniques* (2nd ed.). Marcel Dekker.
- Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a microcomputer environment. *Applied Psychological Measurement, 6*, 431–444.
- Lim, H. irtQ: Unidimensional item response theory modeling. R package, version 1.1.0 (used here as the primary reference).
- Chalmers, R. P. (2012). mirt: A multidimensional item response theory package for the R environment. *Journal of Statistical Software, 48*(6).
- Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and equipercentile observed-score "equatings". *Applied Psychological Measurement, 8*, 453–461.
- Masters, G. N. (1982). A Rasch model for partial credit scoring. *Psychometrika, 47*, 149–174.
- Muraki, E. (1992). A generalized partial credit model. *Applied Psychological Measurement, 16*, 159–176.
- Robitzsch, A., Kiefer, T., & Wu, M. (2024). TAM: Test analysis modules. R package.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph No. 17*.
- Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. S. L. (1995). Item response theory for scores on tests including polytomous items with ordered responses. *Applied Psychological Measurement, 19*, 39–49.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika, 54*, 427–450.
- Wright, B. D., & Stone, M. H. (1979). *Best test design*. MESA Press.
""")
