"""Rasch Calibration Learning Lab — home page.

Run:  streamlit run Rasch_calibration.py      (or: streamlit run app.py)
Pages in ./pages share one dataset through st.session_state.
"""
import streamlit as st

from components.ui import data, interpret, page_setup, table
from r_bridge.bridge import check_r

page_setup("Rasch Calibration Learning Lab",
           "How JMLE, CMLE and MML-EM calibrate Rasch item difficulties, how WLE scores persons, and how "
           "anchors link test forms — shown step by step on your own data.")


@st.cache_data(ttl=120, show_spinner=False)
def r_status():
    return check_r()


st.subheader("Start here · recommended route")
route = [("1 · Rasch Model Explorer", "See how ability and difficulty set the probability of a correct answer."),
         ("2 · Data & Simulation", "Generate responses with known truth, or upload your own 0/1 data."),
         ("3 · JMLE Lab", "Estimate persons and items jointly; watch the alternating updates."),
         ("4 · CMLE Lab", "Condition on raw scores so ability drops out; see the symmetric functions."),
         ("5 · MML-EM Lab", "Integrate ability out with quadrature; step through E- and M-steps."),
         ("6 · Person Scoring", "Score persons by MLE and Warm's WLE with calibrated items."),
         ("7 · Estimator Comparison", "Compare recovery, standard errors, runtime and convergence."),
         ("8 · Theory & Algorithms", "Equations, identification, standard errors, references; run all reference checks."),
         ("9 · Linking & Anchoring", "Put two forms on one bank scale: separate + linking, concurrent, fixed anchor, "
                                     "fixed anchor + concurrent.")]
cols = st.columns(3)
for k, (name, text) in enumerate(route):
    with cols[k % 3]:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(text)
st.caption("Open the pages from the sidebar navigation. Pages 1–8 use the same saved dataset; changing the data "
           "on page 2 invalidates calibrations made on the old data. Page 9 builds its own two-form dataset.")

d = data()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Persons", f"{d.n_persons:,}")
c2.metric("Items", d.n_items)
c3.metric("Missing responses", f"{1 - d.mask.mean():.1%}")
c4.metric("Data source", d.source)

st.subheader("Three result types, always labelled")
table([dict(label="Reference result", meaning="Output of an R package: TAM::tam.jml (JMLE), eRm::RM (CMLE), "
                                              "TAM::tam.mml (MML-EM), TAM::tam.wle (WLE). Never re-created in Python."),
       dict(label="Educational algorithm", meaning="Transparent Python implementation run on the same data; "
                                                   "its iterations are recorded and replayable."),
       dict(label="Intermediate demonstration", meaning="A single step recomputed from recorded inputs "
                                                        "(e.g. one person's E-step) to show the arithmetic.")])

info = r_status()
st.subheader("Reference software status")
if info["available"]:
    st.success(f"{info['R']} · TAM {info.get('TAM') or 'missing'} · eRm {info.get('eRm') or 'missing'}. "
               "Reference comparisons can run on each lab page.")
else:
    st.warning(f"Reference validation unavailable: {info['reason']} The Python educational algorithms still run. "
               "Install R and run install_packages.R to enable TAM/eRm comparisons.")
interpret("The Rasch model has one item parameter (difficulty b) and one person parameter (ability θ). "
          "The three calibration methods differ in **what they do with the person parameters**: JMLE estimates "
          "them alongside the items, CMLE removes them by conditioning on raw scores, and MML-EM integrates them "
          "out over an assumed ability distribution. Person scoring (MLE, WLE) is a separate step that uses the "
          "calibrated difficulties.")
