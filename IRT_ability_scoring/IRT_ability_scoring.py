"""IRT Ability Scoring Learning Lab — home page.

Run:  streamlit run IRT_ability_scoring.py      (or: streamlit run app.py)
Pages in ./pages share one dataset, one selected examinee and one set of scoring settings.
"""
import streamlit as st

from components.ui import data, interpret, page_setup, person, settings, table
from r_bridge.bridge import check_r

page_setup("IRT Ability Scoring Learning Lab",
           "How MLE, EAP, MAP, WLE, TCC inversion and EAPsum turn item responses into an ability estimate — "
           "with known item parameters, for dichotomous, polytomous and mixed-format tests, step by step.")


@st.cache_data(ttl=300, show_spinner=False)
def r_status():
    return check_r()


st.subheader("Start here · recommended route")
route = [("1 · IRT Model Explorer", "Category curves, expected scores and information for 8 models (Rasch–4PL, PCM, GPCM, RSM, GRM)."),
         ("2 · Data & Simulation", "Load examples A–D or simulate a mixed-format test; edit items and response patterns."),
         ("3 · MLE Lab", "Newton–Raphson on the log-likelihood, iteration by iteration; extreme patterns."),
         ("4 · EAP Lab", "Quadrature explorer: likelihood × prior → posterior at every node."),
         ("5 · MAP Lab", "Posterior mode: likelihood score + prior score; prior sensitivity."),
         ("6 · WLE Lab", "Warm's bias correction J/(2I); why it differs from mirt's L·√I for 3PL/4PL."),
         ("7 · TCC & EAPsum Lab", "Scoring from the summed score: TCC inversion and the Lord–Wingersky recursion."),
         ("8 · Estimator Comparison", "All methods on the same examinees; uncertainty; R reference checks."),
         ("9 · Monte Carlo Simulation", "Bias, RMSE, MAD, coverage and failures against known θ."),
         ("10 · Theory & Algorithms", "Equations, derivations, code, numerical examples A–D, references.")]
cols = st.columns(3)
for k, (name, text) in enumerate(route):
    with cols[k % 3]:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(text)
st.caption("Open the pages from the sidebar. Every page uses the dataset, selected examinee and scoring settings "
           "chosen in the sidebar, so you can switch methods without losing the current responses.")

d = data()
s = settings()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", d.name.split(" · ")[0])
c2.metric("Examinees × items", f"{d.n_persons:,} × {d.n_items}")
c3.metric("Selected examinee", d.person_ids[person()])
c4.metric("Prior / nodes", f"N({s.prior_mean:g}, {s.prior_sd:g}²) · {s.n_nodes}")

st.subheader("Three result types, always labelled")
table([dict(label="Reference result (R)", meaning="Output of an installed R package: irtQ::est_score() (primary: ML, WL, "
                                                  "MAP, EAP, EAP.SUM, INV.TCC), mirt::fscores(), TAM::tam.wle(). Never "
                                                  "re-created in Python."),
       dict(label="Educational algorithm (Python)", meaning="Transparent implementation in this app; iterations and "
                                                            "quadrature quantities are recorded and replayable."),
       dict(label="Numerical teaching demonstration", meaning="One step recomputed from recorded inputs (e.g. one Newton "
                                                              "update, one node of the posterior) to show the arithmetic.")])

info = r_status()
st.subheader("Reference software status")
if info["available"]:
    st.success(f"{info['R']} · irtQ {info.get('irtQ') or 'missing'} · mirt {info.get('mirt') or 'missing'} · "
               f"TAM {info.get('TAM') or 'missing'}. Reference checks can run on each lab page.")
else:
    st.warning(f"Reference validation unavailable: {info['reason']} The Python algorithms still run. "
               "Install R and run install_packages.R to enable irtQ / mirt / TAM checks.")

interpret("Item parameters are **known and fixed** throughout this lab: the task is to estimate one examinee's θ "
          "from their responses. The six methods differ in what they use and what they target. **MLE** maximises the "
          "likelihood of the full response pattern. **MAP** and **EAP** combine it with a prior (mode vs mean of the "
          "posterior). **WLE** corrects the MLE's first-order bias. **TCC inversion** and **EAPsum** use only the "
          "summed score. Item calibration (how the parameters were obtained) is the subject of the MMLE-EM and Rasch "
          "Calibration labs.")
