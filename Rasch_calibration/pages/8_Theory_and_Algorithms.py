import numpy as np
import pandas as pd
import streamlit as st

from components.ui import data, download, interpret, labels, page_setup, table
from r_bridge.bridge import check_r, run_reference
from rasch.cmle import cmle
from rasch.jmle import JMLESettings, jmle
from rasch.mml_em import MMLSettings, mml_em
from rasch.scoring import score_persons
from rasch.validation import compare, max_abs

page_setup("8 · Theory & Algorithms", "Equations, identification, standard errors and limitations — and live reference validation.")

st.subheader("1 · The estimators at a glance")
table(pd.DataFrame([
    dict(question="What is estimated?", JMLE="Item b and every person θ", CMLE="Item b only",
         MML_EM="Item b and the ability variance σ²", WLE="Person θ (items fixed)"),
    dict(question="Likelihood maximised", JMLE="Joint L(θ, b)", CMLE="Conditional L(b | raw scores)",
         MML_EM="Marginal L(b, σ²) = ∏∫P(x|θ,b)g(θ)dθ", WLE="Weighted person likelihood"),
    dict(question="Persons handled by", JMLE="Estimated as parameters", CMLE="Conditioned out (raw score sufficient)",
         MML_EM="Integrated out (quadrature)", WLE="Estimated one at a time"),
    dict(question="Assumptions", JMLE="Rasch model", CMLE="Rasch model (sufficiency)",
         MML_EM="Rasch model + normal ability distribution", WLE="Rasch model, known b"),
    dict(question="Identification", JMLE="Mean θ = 0 (TAM 'cases') or mean b = 0", CMLE="Σ b = 0 (eRm sum0)",
         MML_EM="Ability mean = 0", WLE="Inherits the item scale"),
    dict(question="Standard errors", JMLE="1/√Σ P(1−P) per item (conditional on θ̂)", CMLE="Inverse observed conditional information",
         MML_EM="Inverse observed marginal information (TAM reports complete-data)", WLE="1/√I(θ̂)"),
    dict(question="Extreme scores", JMLE="Infinite unless adjusted (adj = 0.3)", CMLE="Uninformative; excluded",
         MML_EM="Used through the distribution", WLE="Finite"),
    dict(question="Convergence", JMLE="max |Δθ|, |Δb| < tol", CMLE="max |conditional score| < tol",
         MML_EM="max |Δb|, |Δσ| < tol; log L non-decreasing", WLE="Root of U_W(θ) (bracketed)"),
    dict(question="Main limitation", JMLE="Inconsistent for fixed J (incidental parameters)", CMLE="Rasch only; no ability distribution",
         MML_EM="Distribution misspecification", WLE="Not an item-calibration method"),
]))

st.subheader("2 · Derivations")
with st.expander("JMLE — score equations and Newton updates"):
    st.latex(r"\ell(\boldsymbol\theta,\mathbf b)=\sum_i\sum_{j\in O_i}\big[x_{ij}(\theta_i-b_j)-\log(1+e^{\theta_i-b_j})\big]")
    st.latex(r"\frac{\partial\ell}{\partial\theta_i}=\sum_{j\in O_i}(x_{ij}-P_{ij}),\quad \frac{\partial^2\ell}{\partial\theta_i^2}=-\sum_{j\in O_i}P_{ij}(1-P_{ij})")
    st.latex(r"\frac{\partial\ell}{\partial b_j}=\sum_{i}(P_{ij}-x_{ij}),\quad \frac{\partial^2\ell}{\partial b_j^2}=-\sum_{i}P_{ij}(1-P_{ij})")
    st.markdown("Newton: parameter ← parameter − score / second derivative, i.e. residual divided by information. "
                "Andersen (1973) and Neyman–Scott (1948) show why estimating one parameter per person makes the item "
                "estimates inconsistent when the number of items stays fixed; for the Rasch model the bias is about "
                "a factor J/(J−1), the basis of TAM's correction.")
with st.expander("CMLE — elementary symmetric functions and the conditional score"):
    st.latex(r"P(\mathbf x_i\mid r_i)=\frac{\prod_j\varepsilon_j^{x_{ij}}}{\gamma_{r_i}},\quad \varepsilon_j=e^{-b_j}")
    st.latex(r"\gamma_r^{(k)}=\gamma_r^{(k-1)}+\varepsilon_k\gamma_{r-1}^{(k-1)},\qquad \gamma_0=1")
    st.latex(r"\frac{\partial\ell_c}{\partial b_j}=\sum_i\Big[\frac{\varepsilon_j\gamma^{(-j)}_{r_i-1}}{\gamma_{r_i}}-x_{ij}\Big]")
    st.markdown("γ⁽⁻ʲ⁾ is the ESF with item j removed; ε_j γ⁽⁻ʲ⁾_{r−1}/γ_r is the probability that item j is correct "
                "given raw score r. Computation is in log space (log-sum-exp), so long tests do not overflow.")
with st.expander("MML-EM — Fisher's identity, E-step and M-step"):
    st.latex(r"\frac{\partial\log L}{\partial b_j}=\sum_i E_{\theta\mid\mathbf x_i}\Big[\frac{\partial\log P(\mathbf x_i\mid\theta)}{\partial b_j}\Big]"
             r"\approx\sum_q\big(N_{jq}P_{jq}-R_{jq}\big)")
    st.latex(r"Q(\mathbf b)=\sum_j\sum_q\big[R_{jq}\log P_{jq}+(N_{jq}-R_{jq})\log(1-P_{jq})\big]")
    st.markdown("The E-step computes posterior node weights; the M-step maximises Q, which separates into one "
                "equation per item. Because Q is a lower bound that touches log L at the current parameters, each "
                "full EM cycle cannot decrease the marginal log-likelihood (Bock & Aitkin, 1981).")
with st.expander("WLE — Warm's correction"):
    st.latex(r"U_W(\theta)=\sum_j(x_j-P_j)+\frac{I'(\theta)}{2I(\theta)},\quad I'(\theta)=\sum_jP_j(1-P_j)(1-2P_j)")
    st.markdown("For the Rasch model the WLE maximises L(θ)·√I(θ); the extra term removes the O(1/J) bias of the MLE "
                "(Warm, 1989). The sign is checked against TAM::tam.wle in the validation below.")

st.subheader("3 · Notation")
table(pd.DataFrame([("X_ij", "response of person i to item j (0/1; missing excluded)"), ("O_i", "items observed for person i"),
                    ("θ_i, b_j", "ability and difficulty (logits)"), ("P_ij", "model probability of a correct response"),
                    ("r_i, s_j", "person raw score, item score"), ("γ_r", "elementary symmetric function of order r"),
                    ("θ_q, w_q", "quadrature node and prior weight"), ("τ_iq", "posterior weight of person i at node q"),
                    ("N_jq, R_jq", "expected respondents / correct answers at node q"), ("I(θ)", "test information Σ P(1−P)")],
                   columns=["symbol", "meaning"]))

st.subheader("4 · Live reference validation on the current dataset")
labels("edu", "ref")
info = check_r()
if not info["available"] or not info.get("TAM") or not info.get("eRm"):
    st.warning(f"Reference validation unavailable: {info['reason']}")
else:
    st.caption(f"{info['R']} · TAM {info['TAM']} · eRm {info['eRm']}")
    if st.button("Run all reference comparisons", type="primary"):
        d = data()
        rows = []
        with st.spinner("Fitting Python and R estimators on the current data…"):
            j = jmle(d.X, JMLESettings())
            rj = run_reference("JMLE", d.X, dict(adj=.3, bias=True, constraint="cases", tol=1e-4, max_iter=1000), d.item_ids)
            rows += [dict(method="JMLE vs TAM::tam.jml", **r) for r in compare("JMLE", dict(b=j.b, se_b=j.se_b, theta=j.theta), rj)]
            try:
                c = cmle(d.X)
                rc = run_reference("CMLE", d.X, {}, d.item_ids)
                rows += [dict(method="CMLE vs eRm::RM", **r) for r in compare("CMLE", dict(b=c.b, se_b=c.se_b, cond_loglik=c.cond_loglik), rc, d.n_persons)]
            except ValueError as e:
                rows.append(dict(method="CMLE vs eRm::RM", check="Python CMLE", status=f"Not run: {e}"))
            m = mml_em(d.X, MMLSettings())
            s = dict(n_nodes=21, extent=6.0, estimate_variance=True, fixed_sd=1.0, tol=1e-4, max_iter=1000)
            rm = run_reference("MML", d.X, s, d.item_ids)
            rows += [dict(method="MML-EM vs TAM::tam.mml", **r) for r in compare("MML", dict(b=m.b, se_b=m.se_b_complete, se_label=" (complete-data)", variance=m.variance, loglik=m.loglik_tam), rm, d.n_persons)]
            if rm.get("status") == "Executed":
                w, wse = score_persons(d.X, rm["b"], "WLE")
                for label, a, b2 in (("WLE abilities", w, rm["wle"]), ("WLE standard errors", wse, rm["wle_se"])):
                    diff = max_abs(a, b2)
                    rows.append(dict(method="WLE vs TAM::tam.wle", check=label, max_abs_difference=diff, tolerance=1e-3,
                                     status="Pass" if diff <= 1e-3 else "Fail"))
        st.session_state.validation_rows = (d.identity, rows)
    saved = st.session_state.get("validation_rows")
    if saved and saved[0] == data().identity:
        frame = pd.DataFrame(saved[1])
        table(frame, 7)
        download(frame, "reference_validation.csv")
        n_pass = int((frame.status == "Pass").sum())
        st.markdown(f"**{n_pass} of {len(frame)} checks pass** (absolute tolerance 1e-3; per-person 1e-5 for log-likelihoods).")
interpret("Agreement means the transparent Python algorithms reproduce the R packages on these data. It is a check of "
          "implementation, not a proof that any estimator is unbiased for your data — that is what the simulation "
          "studies on page 7 address.")

st.subheader("5 · References")
st.markdown("""
- Andersen, E. B. (1970). Asymptotic properties of conditional maximum-likelihood estimators. *JRSS B, 32*, 283–301.
- Andersen, E. B. (1973). Conditional inference and models for measuring. Copenhagen: Mentalhygiejnisk Forlag.
- Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*, 443–459.
- Fischer, G. H., & Molenaar, I. W. (Eds.) (1995). *Rasch models: Foundations, recent developments, and applications*. Springer.
- Neyman, J., & Scott, E. L. (1948). Consistent estimates based on partially consistent observations. *Econometrica, 16*, 1–32.
- Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory. *Psychometrika, 54*, 427–450.
- Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press.
- Robitzsch, A., Kiefer, T., & Wu, M. — TAM package: https://cran.r-project.org/package=TAM
- Mair, P., & Hatzinger, R. — eRm package: https://cran.r-project.org/package=eRm
""")
