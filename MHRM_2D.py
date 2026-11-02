"""MHRM_2D Streamlit App
Three-stage Metropolis-Hastings Robbins-Monro (MHRM) calibration for a
confirmatory binary M2PL model with fixed N(0,I) latent distribution.
The independently testable estimator is implemented in mhrm_engine.py.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import streamlit.components.v1 as components
from mhrm_engine import MHRM_MIRT, simulate_data, marginal_loglikelihood
from mhrm_animation import item_convergence_figure, item_history_table, item_path
from mirt_validation import check_mirt, fit_mirt, compare_estimates
import hashlib
import json
from pathlib import Path
import time
import math

# One text scale for every matplotlib chart so panels placed side by side line up.
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
# Figure sizes matched to column widths so text renders at the same size everywhere.
GRID3 = (3.3, 2.7)   # charts in a 3-column row
GRID2 = (4.8, 3.3)   # charts in a 2-column row
INK, MUTED, BLUE, GREEN, ORANGE = '#0b0b0b', '#52514e', '#2a78d6', '#1baf7a', '#eb6834'


@st.cache_data(ttl=60, show_spinner=False)
def _mirt_available():
    return check_mirt()


@st.cache_data(show_spinner=False)
def _fit_r_reference(data, q_matrix, settings, reference_version, source_signature):
    return fit_mirt(data, q_matrix, **settings)

# ==========================================
# 1. CORE MATH & STATISTICS FUNCTIONS
# ==========================================

def summarize_recovery(true_values, est_values):
    """Return recovery diagnostics for two aligned parameter vectors."""
    true_values = np.asarray(true_values, dtype=float)
    est_values = np.asarray(est_values, dtype=float)
    bias = est_values - true_values

    if true_values.size < 2 or np.allclose(true_values.std(), 0) or np.allclose(est_values.std(), 0):
        corr = np.nan
    else:
        corr = np.corrcoef(true_values, est_values)[0, 1]

    return {
        'Mean True': np.mean(true_values),
        'Mean Est': np.mean(est_values),
        'Mean Bias': np.mean(bias),
        'MAE': np.mean(np.abs(bias)),
        'RMSE': np.sqrt(np.mean(bias ** 2)),
        'Corr': corr
    }

def plot_path_diagram(q_matrix, structure_type):
    """
    Visualizes the MIRT model structure as a path diagram.
    Vertical format: Factors on Left, Items on Right.
    For Bifactor: General Factor Left, Items Middle, Specific Factors Right.
    """
    J, D = q_matrix.shape
    # Adjust height based on J
    fig, ax = plt.subplots(figsize=(4.4, max(3.2, J * 0.13)))
    
    if structure_type == "Bifactor":
        # G on Left, S on Right, Items in Middle
        factor_x = np.zeros(D)
        factor_y = np.zeros(D)
        
        # General Factor (Index 0)
        factor_x[0] = 0.1
        factor_y[0] = 0.5
        
        # Specific Factors (Indices 1..D-1)
        if D > 1:
            factor_x[1:] = 0.9
            factor_y[1:] = np.linspace(0.8, 0.2, D-1)
            
        # Items in Middle
        item_x = np.ones(J) * 0.5
        item_y = np.linspace(0.95, 0.05, J)
        
    else:
        # Standard: Factors Left, Items Right
        factor_x = np.ones(D) * 0.2
        factor_y = np.linspace(0.8, 0.2, D)
        
        item_x = np.ones(J) * 0.8
        item_y = np.linspace(0.95, 0.05, J)
    
    # Draw Edges
    for j in range(J):
        for d in range(D):
            if q_matrix[j, d] == 1:
                ax.plot([factor_x[d], item_x[j]], [factor_y[d], item_y[j]], 'k-', alpha=0.3, lw=1)
                
    # Draw Nodes
    # Factors
    ax.scatter(factor_x, factor_y, s=520, c='#b7d3f6', edgecolors=INK, zorder=10)
    for d in range(D):
        if structure_type == "Bifactor":
            label = "G" if d == 0 else f"S_{d}"
        else:
            label = f"F{d+1}"
        ax.text(factor_x[d], factor_y[d], label, ha='center', va='center', fontweight='bold', fontsize=8, zorder=11)
        
    # Items
    ax.scatter(item_x, item_y, s=90, c='white', edgecolors=INK, linewidths=.8, zorder=10)
    # Label all items
    for j in range(J):
        ax.text(item_x[j], item_y[j], str(j+1), ha='center', va='center', fontsize=6, zorder=11)
            
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Path diagram: {structure_type}")
    fig.tight_layout()
    return fig

def interpret(markdown_text):
    """Educational 'how to read this' box shown under each step."""
    with st.container(border=True):
        st.markdown("📘 **How to read this**\n\n" + markdown_text)


def show_table(df, digits=3):
    """Hide the row index and show every float column with the same number of decimals."""
    floats = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    st.dataframe(df, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format=f"%.{digits}f") for c in floats})


def chart_grid(n, per_row=3):
    """Yield n equal-width columns, per_row at a time, so charts line up in a grid."""
    for start in range(0, n, per_row):
        cols = st.columns(per_row)
        for k in range(min(per_row, n - start)):
            yield cols[k]


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def quality(rmse):
    """Rough rule of thumb for simulation studies (not a formal standard)."""
    return "excellent" if rmse < .10 else "good" if rmse < .20 else "fair" if rmse < .30 else "poor"


def recovery_scatter(true_values, est_values, label, color=BLUE):
    m = summarize_recovery(true_values, est_values)
    fig, ax = plt.subplots(figsize=GRID3)
    lo = min(np.min(true_values), np.min(est_values)); hi = max(np.max(true_values), np.max(est_values))
    pad = .06 * ((hi - lo) or 1)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls='--', color=MUTED, lw=1, zorder=1)
    ax.scatter(true_values, est_values, s=26, color=color, edgecolors='white', linewidths=.6, zorder=2)
    ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), xlabel='True', ylabel='Estimated')
    ax.set_title(f"{label}\nbias {m['Mean Bias']:+.3f} · RMSE {m['RMSE']:.3f} · r {m['Corr']:.3f}")
    ax.grid(alpha=.3)
    fig.tight_layout()
    show_fig(fig)


def loading_heatmap(values, mask, title, column_labels, vmax):
    """Loading matrix with fixed-zero cells in light gray so they are not mistaken for small slopes."""
    rows = values.shape[0]
    fig, ax = plt.subplots(figsize=(GRID2[0], max(GRID2[1], rows * .12)))
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#e4e3df')
    image = ax.imshow(np.ma.masked_where(~mask, values), aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(values.shape[1])); ax.set_xticklabels(column_labels)
    step = 1 if rows <= 30 else 2 if rows <= 60 else 5
    ax.set_yticks(range(0, rows, step)); ax.set_yticklabels([str(j+1) for j in range(0, rows, step)], fontsize=7)
    ax.set(xlabel='Dimension', ylabel='Item', title=title)
    fig.colorbar(image, ax=ax, shrink=.8, label='Slope (a)')
    fig.tight_layout()
    show_fig(fig)


def _render_live(model, diagnostic):
    started = time.perf_counter()
    with plot_placeholder.container():
        ncols = 2 if model.D == 4 else min(3, model.D)   # 4 panels as 2x2 instead of 3+1
        nrows = math.ceil(model.D / ncols)
        height = 2.0 * nrows + .6
        left, right = st.columns(2)
        with left:
            fig, axes = plt.subplots(nrows, ncols, figsize=(GRID2[0], height), squeeze=False)
            for k, ax in enumerate(axes.flat):
                if k < model.D:
                    ax.hist(model.current_theta[:, k], bins=25, density=True, color=BLUE, alpha=.75)
                    name = f"dim {k+1}" if factor_labels is None else factor_labels[k]
                    ax.set_title(f"θ draws: {name}")
                    ax.set_xlim(-4, 4)
                else:
                    ax.axis('off')
            fig.tight_layout()
            show_fig(fig)
        with right:
            fig, ax = plt.subplots(figsize=(GRID2[0], height))
            ax.plot(model.iteration_history, model.loss_history, color=BLUE, lw=1.6)
            ax.set(title='Sampled conditional log-likelihood', xlabel='Total cycle', ylabel='Conditional LL')
            ax.grid(alpha=.3)
            fig.tight_layout()
            show_fig(fig)
        if show_live_rm:
            st.markdown(f"**Live parameter paths — Item {inspection_item+1}**")
            available = model.parameter_history if playback_stages == 'all' else model.rm_parameter_history
            if available:
                figure = item_convergence_figure(model, inspection_item, true_a=true_a,
                                                 true_d=true_d, animate=False,
                                                 factor_labels=factor_labels, stages=playback_stages)
                st.plotly_chart(figure, width='stretch',
                    key=f"rm_live_{diagnostic['iteration']}_{inspection_item}")
            else:
                st.caption("RM parameter updates start after burn-in and SEM averaging.")
    # Drawing time is excluded from the reported estimation time.
    render_seconds[0] += time.perf_counter() - started


# ==========================================
# 3. STREAMLIT APP LAYOUT
# ==========================================

st.set_page_config(layout="wide", page_title="MIRT Calibration with MHRM")

st.title("🧩 MHRM Algorithm for MIRT Calibration")
st.markdown(r"""
This application demonstrates the **Metropolis-Hastings Robbins-Monro (MHRM)** algorithm
for calibrating a Multidimensional Item Response Theory (MIRT) model.
Three-stage MH-RM for a binary compensatory 2PL with Q-matrix zeros, positive
slopes bounded by 10, and a fixed $N(0,I)$ latent distribution. Uses full item
curvature blocks, SEM averaging, and shrinking Robbins–Monro updates.
Identification and adequate sampling are required; convergence is not a guarantee of a global maximum.
""")

# --- Sidebar Controls ---
st.sidebar.header("1. Data Generation")
N = st.sidebar.slider("Sample Size (N)", 500, 5000, 2000, step=500)
J = st.sidebar.slider("Number of Items (J)", 10, 60, 30, step=5)

st.sidebar.header("Structure")
structure = st.sidebar.radio(
    "Loading Structure",
    ["Simple", "Cross-load", "Bifactor"],
    index=0,
    help="Simple: one loading per item. Cross-load: items may load on multiple dims. Bifactor: general factor + specific factors."
)
# Bifactor needs at least three specific factors (see the check below), so its slider starts at 3.
if structure == "Bifactor":
    D = st.sidebar.slider("Dimensions (Specific Factors)", 3, 5, 3,
                          help="Bifactor models here need at least three specific factors.")
else:
    D = st.sidebar.slider("Dimensions (Specific Factors)", 1, 5, 2)

if structure == "Cross-load":
    cross_p = st.sidebar.slider(
        "Cross-loading probability per dimension",
        0.05, 0.80, 0.30, step=0.05,
        help="Probability an item loads on a given dimension (independent). Ensures at least one loading per item."
    )
    min_loads = 1 if D == 1 else st.sidebar.slider(
        "Minimum loadings per non-anchor item",
        1, D, 1,
        help="Applies to non-anchor items. Three pure anchors per dimension are reserved to fix the axes."
    )
elif structure == "Bifactor":
    general_strength = st.sidebar.slider(
        "General factor discrimination range high",
        1.0, 3.0, 2.0, step=0.25,
        help="Upper bound for Uniform(0.5, high) for general factor a parameters."
    )
    specific_scale = st.sidebar.slider(
        "Specific factor max discrimination",
        1.0, 2.5, 2.0, step=0.25,
        help="Upper bound for Uniform(0.5, max) for specific factors."
    )

st.sidebar.header("2. MHRM Settings")
max_iters = st.sidebar.slider("Maximum RM Cycles (Stage 3)", 50, 2000, 500, step=50)
burn_in = st.sidebar.slider("Burn-in Cycles (Stage 1)", 10, 300, 150)
sem_cycles = st.sidebar.slider("SEM Averaging Cycles (Stage 2)", 20, 300, 100)
decay_rate = st.sidebar.slider("Gain Exponent", 0.51, 1.0, 0.75,
    help="RM gain is (gain scale / RM cycle)^exponent. Exponents must exceed 0.5 and be at most 1.")
gain_scale = st.sidebar.number_input("Gain Scale", .01, 1.0, .1, .01)
mh_sweeps = st.sidebar.slider("MH Sweeps per Cycle", 1, 30, 5)
convergence_tol = st.sidebar.number_input("Maximum-change Tolerance", .0001, .01, .001, .0001, format="%.4f")
random_seed = st.sidebar.number_input("Random Seed", 0, 1000000, 42)
compute_marginal = st.sidebar.checkbox("Estimate Final Marginal Log-Likelihood", value=True)
st.sidebar.caption("The total cycle limit is burn-in + SEM averaging + RM. Convergence additionally requires a small 30-cycle mean projected score. The MH proposal is tuned during burn-in and then frozen.")

st.sidebar.header("3. R mirt Cross-check")
r_available, r_status = _mirt_available()
compare_r = st.sidebar.checkbox("Cross-check with R mirt", value=r_available, disabled=not r_available)
if r_available:
    st.sidebar.caption(f"R mirt {r_status}. Compares fits on the same data; this is not held-out predictive cross-validation.")
else:
    st.sidebar.info(r_status + ' To enable the cross-check, install R and run install.packages("mirt") in R.')

st.sidebar.header("4. Parameter Convergence Animation")
inspection_item = st.sidebar.selectbox("Item to inspect", range(1, J+1),
                                     format_func=lambda value: f"Item {value}") - 1
playback_range = st.sidebar.radio("Playback range", ["All stages (burn-in → SEM → RM)", "RM stage only"],
    help="All stages starts from the random starting values, so the movement toward the estimate is visible. "
         "RM stage only starts from the SEM-averaged values, where paths are usually already close to final.")
playback_stages = 'all' if playback_range.startswith('All') else 'rm'
show_live_rm = st.sidebar.checkbox("Show live item updates", value=True)
playback_speed = st.sidebar.selectbox("Playback speed", ['Normal', 'Slow', 'Fast'])
st.sidebar.caption("Inspect every active discrimination and the intercept for any item. Item selection and playback settings reuse the recorded calibration.")

# Retain fitted Python results across downloads and display-only option changes.
run_settings = [N, J, D, structure, cross_p if structure == 'Cross-load' else None,
                min_loads if structure == 'Cross-load' else None,
                general_strength if structure == 'Bifactor' else None,
                specific_scale if structure == 'Bifactor' else None,
                max_iters, burn_in, sem_cycles, decay_rate, gain_scale,
                mh_sweeps, convergence_tol, random_seed]
engine_signature = Path(__file__).with_name('mhrm_engine.py').read_bytes()
run_fingerprint = hashlib.sha256(json.dumps(run_settings).encode()+engine_signature).hexdigest()

# --- Main Logic ---

run_requested = st.button("🚀 Run Simulation & Calibration", type="primary")
saved_run = st.session_state.get('mhrm_saved_run')
if run_requested or (saved_run is not None and saved_run['fingerprint'] == run_fingerprint):

    # Supported designs use conservative structural safeguards, not a general
    # proof of identification for arbitrary confirmatory loading patterns.
    if J < 3 * D:
        st.error("Use at least three items per dimension/specific factor (increase J or reduce D).")
        st.stop()
    if structure == "Bifactor" and D < 3:
        st.error("This demonstration supports bifactor models with at least three specific factors. Two-group models need additional identification checks; one group leaves the axes unidentified.")
        st.stop()
    q_rng = np.random.default_rng(int(random_seed) + 1)
    if structure == "Simple":
        q_matrix = np.zeros((J, D))
        for k, indices in enumerate(np.array_split(np.arange(J), D)):
            q_matrix[indices, k] = 1
    elif structure == "Cross-load":
        q_matrix = np.zeros((J, D))
        # Three pure indicators per factor fix the orientation; additional
        # items may cross-load. Do not treat an all-free mask as identified.
        for k in range(D):
            q_matrix[3*k:3*k+3, k] = 1
        for j in range(3*D, J):
            loads = q_rng.random(D) < cross_p
            if loads.sum() < min_loads:
                loads[q_rng.choice(D, size=min_loads, replace=False)] = True
            q_matrix[j, loads] = 1
    else:
        D_effective = D + 1
        q_matrix = np.zeros((J, D_effective))
        q_matrix[:, 0] = 1
        for k, indices in enumerate(np.array_split(np.arange(J), D)):
            q_matrix[indices, k+1] = 1

    # Effective dimension count for model objects
    if structure == "Bifactor":
        dims_for_model = D_effective
    else:
        dims_for_model = D
    # Bifactor tables and charts name factors G, S_1, S_2, ...; other structures keep a1, a2, ...
    factor_labels = (["G"] + [f"S_{k}" for k in range(1, dims_for_model)]
                     if structure == "Bifactor" else None)

    # 2. Simulate Data
    with st.spinner("Simulating Response Data..."):
        # Simulate data based on effective dimension count
        if structure == "Bifactor":
            # Generate parameters respecting the slider settings
            simulation_rng = np.random.default_rng(int(random_seed) + 2)
            # General Factor (Col 0)
            a_gen = simulation_rng.uniform(0.5, general_strength, size=J)
            # Specific Factors (Cols 1..D)
            a_spec = simulation_rng.uniform(0.5, specific_scale, size=(J, D))
            
            # Combine into true_a matrix
            true_a_sim = np.zeros((J, D_effective))
            true_a_sim[:, 0] = a_gen
            true_a_sim[:, 1:] = a_spec
            
            # Apply Q-matrix mask
            true_a_sim = true_a_sim * q_matrix
            
            response_data, true_theta, true_a, true_d = simulate_data(N, J, dims_for_model, q_matrix, true_a=true_a_sim, seed=int(random_seed))
        else:
            response_data, true_theta, true_a, true_d = simulate_data(N, J, D, q_matrix, seed=int(random_seed))

    # Display names used throughout the steps below.
    def dim_name(k):
        if structure == "Bifactor":
            return "General (G)" if k == 0 else f"Specific (S_{k})"
        return f"Dim {k+1}"

    def dim_short(k):
        return f"D{k+1}" if factor_labels is None else factor_labels[k]

    # ==========================================================
    # Step 1 · Model structure and simulated data
    # ==========================================================
    st.subheader("Step 1 · Model structure and simulated data")
    p_correct = response_data.mean(axis=0)
    items_per_dim = q_matrix.sum(axis=0).astype(int)
    left, right = st.columns([1, 1.15])
    with left:
        show_fig(plot_path_diagram(q_matrix, structure))
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Examinees (N)", f"{N:,}")
        c2.metric("Items (J)", J)
        c3.metric("Latent dimensions", dims_for_model)
        c4, c5, c6 = st.columns(3)
        c4.metric("Avg. proportion correct", f"{p_correct.mean():.2f}")
        c5.metric("Hardest item", f"{p_correct.min():.2f}")
        c6.metric("Easiest item", f"{p_correct.max():.2f}")
        if structure == "Simple":
            design = (f"Each item measures exactly **one** dimension "
                      f"({', '.join(f'{dim_short(k)}: {n} items' for k, n in enumerate(items_per_dim))}). "
                      "This is the easiest design to estimate because every slope is informed by its own set of items.")
        elif structure == "Cross-load":
            n_cross = int(np.sum(q_matrix.sum(axis=1) > 1))
            design = (f"Items 1–{3*D} are **anchors**: three pure items per dimension that fix what each dimension means. "
                      f"**{n_cross}** of the other {J-3*D} items load on two or more dimensions; their slopes have to be "
                      "separated from each other, which makes them harder to estimate.")
            if n_cross == 0:
                design += " With these settings no item cross-loads, so this design is effectively Simple structure."
        else:
            design = (f"Every item loads on the **general factor G** (the common trait) and on **one specific factor** "
                      f"(S_1…S_{D}, {', '.join(str(n) for n in items_per_dim[1:])} items each). Specific slopes are usually "
                      "estimated less precisely: each specific factor has few items and shares them with G. Estimates can be "
                      "unstable when an item's general and specific slopes are nearly proportional.")
        slope_range = (f"Uniform(0.5, {general_strength:g}) for G and Uniform(0.5, {specific_scale:g}) for specific factors"
                       if structure == "Bifactor" else "Uniform(0.5, 2)")
        interpret(
            f"{design}\n\n"
            "Responses were **simulated** from the compensatory multidimensional 2PL model, "
            "P(correct) = logistic(**a**·**θ** + d), with abilities **θ** ~ N(0, I). "
            f"True slopes were drawn from {slope_range}; intercepts from Uniform(−2, 2). Because the true values are "
            "known, later steps can measure how well MHRM recovers them.\n\n"
            f"On average {p_correct.mean():.0%} of examinees answered an item correctly (range {p_correct.min():.0%}–"
            f"{p_correct.max():.0%}). Items that almost everyone gets right or wrong carry little information, "
            "so their parameters are estimated less precisely.")

    # ==========================================================
    # Step 2 · Calibrating with MHRM
    # ==========================================================
    st.subheader("Step 2 · Calibrating with MHRM")
    interpret(
        "Each cycle of MHRM makes two moves: **(1) Metropolis–Hastings (MH)** draws plausible abilities θ for every "
        "examinee given the current item parameters, then **(2) Robbins–Monro (RM)** nudges the item parameters "
        "toward values that fit those draws better. The run has three stages:\n\n"
        f"- **Burn-in** ({burn_in} cycles): large steps (gain 0.2) away from random starting values; the MH sampler "
        "also tunes its step size.\n"
        f"- **SEM averaging** ({sem_cycles} cycles): the same large steps, but the results are averaged to give a "
        "stable starting point.\n"
        f"- **RM** (up to {max_iters} cycles): the step size shrinks every cycle, so the random noise from sampling "
        "averages out and the estimates settle.\n\n"
        "While it runs: the **left** chart shows the current θ draws (they should look like a bell curve centred "
        "at 0); the **right** chart shows the conditional log-likelihood rising quickly and then levelling off; "
        "the chart below follows the selected item's parameters.")
    prog_bar = st.progress(0, text="Initializing MHRM...")
    plot_placeholder = st.empty()
    render_seconds = [0.]
    if run_requested:
        mhrm = MHRM_MIRT(response_data, dims_for_model, q_matrix, seed=int(random_seed) + 3)
        start_time = time.perf_counter()
        mhrm.fit(max_iter=max_iters, burn_in=burn_in, sem_cycles=sem_cycles,
                 decay=decay_rate, gain_scale=gain_scale, mh_steps=mh_sweeps,
                 tolerance=convergence_tol, progress_bar=prog_bar, callback=_render_live)
        # Estimation time only: time spent drawing the live charts is excluded.
        elapsed_seconds = time.perf_counter() - start_time - render_seconds[0]
        st.session_state['mhrm_saved_run'] = dict(fingerprint=run_fingerprint,
                                                 model=mhrm, seconds=elapsed_seconds)
    else:
        mhrm, elapsed_seconds = saved_run['model'], saved_run['seconds']
        prog_bar.progress(1., text=mhrm.stop_reason_)
        _render_live(mhrm, mhrm.diagnostics[-1])
    if mhrm.converged:
        st.success(f"Convergence criteria met after {mhrm.rm_iterations_} RM cycles ({mhrm.n_iter_} total). "
                   f"Estimation time: {elapsed_seconds:.2f} seconds (live-chart drawing excluded).")
    else:
        st.warning(f"{mhrm.stop_reason_}. Estimates are provisional. "
                   f"Estimation time: {elapsed_seconds:.2f} seconds (live-chart drawing excluded).")
    diagnostics = pd.DataFrame(mhrm.diagnostics)
    after_burn = diagnostics[diagnostics.stage != 'Burn-in']
    last = mhrm.diagnostics[-1]
    acceptance = float(after_burn.acceptance.mean()) if len(after_burn) else float(diagnostics.acceptance.iloc[-1])
    acc_note = ("in the healthy 0.2–0.4 range, so the sampler explores well" if .2 <= acceptance <= .4 else
                "below 0.2, so draws change slowly — consider more MH sweeps" if acceptance < .2 else
                "above 0.4, so steps are small — mixing may be slow")
    status = (f"**Converged** after {mhrm.rm_iterations_} RM cycles ({mhrm.n_iter_} cycles in total)."
              if mhrm.converged else
              f"**Did not converge** within {max_iters} RM cycles. Treat the estimates as provisional; "
              "increase *Maximum RM Cycles* (at least 52 are needed, because the rule must hold for three "
              "consecutive cycles after cycle 50) or add MH sweeps.")
    projected = last.get('mean_projected_score', float('nan'))
    interpret(
        f"- {status}\n"
        f"- **MH acceptance rate** after burn-in ≈ {acceptance:.2f} (proposal SD {mhrm.proposal_sd:.2f}): {acc_note}.\n"
        f"- **Largest parameter change** in the final cycle: {last['max_change']:.4f} (tolerance {convergence_tol:g}). "
        "Small changes alone are not enough, because the shrinking step size forces changes to become small.\n"
        f"- **Average gradient** over the last 30 cycles: {projected:.4f} (threshold 0.005). A value near zero means "
        "the likelihood has no remaining uphill direction — the real sign that the estimates have settled.\n\n"
        "Convergence means the algorithm has stopped moving, not that the estimates equal the true values. "
        "Steps 3–4 show how close they are.")
    with st.expander("Cycle-by-cycle diagnostics table"):
        show_table(diagnostics, digits=4)
    st.markdown("[Jump to the parameter convergence playback](#convergence-playback)")

    est_a, est_d = mhrm.est_a, mhrm.est_d
    python_marginal = None
    if compute_marginal:
        python_marginal = marginal_loglikelihood(response_data, est_a, est_d)

    # The R reference is fitted here so the playback (Step 3) can show its endpoints.
    reference, comparison, r_error = None, None, None
    if compare_r:
        r_settings = dict(cycles=int(max_iters), burn_in=int(burn_in), sem_cycles=int(sem_cycles),
                          seed=int(random_seed)+3, gain_scale=float(gain_scale), decay=float(decay_rate),
                          mh_steps=int(mh_sweeps), tolerance=float(convergence_tol))
        try:
            with st.spinner("Fitting the same model with R mirt for the cross-check (Step 7)..."):
                reference_source = b''.join(Path(__file__).with_name(name).read_bytes()
                    for name in ('mirt_validation.py', 'mirt_validation.R'))
                reference = _fit_r_reference(response_data, q_matrix, r_settings, r_status,
                                             hashlib.sha256(reference_source).hexdigest())
                comparison = compare_estimates(response_data, q_matrix, est_a, est_d, reference,
                                               true_a, true_d, python_marginal,
                                               factor_labels=factor_labels)
        except (RuntimeError, ValueError, OSError) as exc:
            r_error, reference, comparison = exc, None, None

    # ==========================================================
    # Step 3 · Parameter convergence playback
    # ==========================================================
    st.subheader("Step 3 · Parameter convergence playback", anchor="convergence-playback")
    duration = {'Slow': 500, 'Normal': 180, 'Fast': 60}[playback_speed]
    playback = item_convergence_figure(mhrm, inspection_item, true_a=true_a, true_d=true_d,
        reference_a=None if reference is None else reference['a'],
        reference_d=None if reference is None else reference['d'], frame_duration=duration,
        factor_labels=factor_labels, stages=playback_stages)
    # Embed Plotly's full animation player: native static charts do not reliably
    # retain frame playback across Streamlit versions. No external CDN is needed.
    player_html = pio.to_html(playback, include_plotlyjs=True, full_html=False,
                              auto_play=False, config={'responsive': True, 'displaylogo': False})
    components.html(player_html, height=int(playback.layout.height)+35, scrolling=False)
    st.caption("Use Play, Pause, Restart, or the slider (choose the item and range in the sidebar). Long runs sample "
               "playback frames; the trace and table keep every recorded cycle. Fixed-zero loadings are omitted.")
    path = item_path(mhrm, inspection_item, factor_labels, playback_stages)
    lines = []
    for k, (parameter, label) in enumerate(zip(path['parameters'], path['labels'])):
        truth = true_a[inspection_item, parameter] if parameter < mhrm.D else true_d[inspection_item]
        values = path['values'][:, k]
        wobble = np.ptp(values[-30:])
        lines.append(f"- **{label}**: started at {values[0]:.3f}, ended at {values[-1]:.3f}; true value {truth:.3f} "
                     f"(error {values[-1]-truth:+.3f}). Over the last 30 recorded cycles it moved by only {wobble:.4f}.")
    stage_story = ("Read the shaded bands left to right: in **burn-in** the parameters climb quickly from random "
                   "starting values; in **SEM** they jitter around their target because the step size is still "
                   "large; in **RM** the shrinking steps smooth the path until it is flat — that flat line is what "
                   "convergence looks like." if playback_stages == 'all' else
                   "This view starts at the SEM-averaged values, so most of the movement has already happened; "
                   "switch *Playback range* to *All stages* to see the path from the random start.")
    interpret(
        f"{stage_story}\n\n**Item {inspection_item+1}:**\n" + "\n".join(lines) + "\n\n"
        "The green dotted line is the true (simulated) value. A flat path that stops a little away from it is "
        "normal: with a finite sample even the best possible estimate differs from the truth. "
        + ("The purple dashed line is R mirt's estimate; when both estimators end in the same place, the gap to "
           "the truth comes from the sample, not from the algorithm." if reference is not None else ""))
    selected_history = item_history_table(mhrm, inspection_item, factor_labels=factor_labels,
                                          stages=playback_stages)
    with st.expander(f"Recorded values and changes for Item {inspection_item+1}"):
        show_table(selected_history, digits=4)
        st.download_button("Download this item's trajectory CSV", selected_history.to_csv(index=False),
                           f'mhrm_item_{inspection_item+1}_{playback_stages}_trajectory.csv', 'text/csv')

    # ==========================================================
    # Step 4 · Parameter recovery
    # ==========================================================
    st.subheader("Step 4 · Parameter recovery")
    overall_a_metrics = summarize_recovery(true_a[q_matrix == 1], est_a[q_matrix == 1])
    d_metrics = summarize_recovery(true_d, est_d)
    dim_rows, dim_metrics = [], {}
    for dim in range(mhrm.D):
        mask_dim = q_matrix[:, dim] == 1
        if mask_dim.any():
            dim_metrics[dim] = summarize_recovery(true_a[mask_dim, dim], est_a[mask_dim, dim])
            dim_rows.append({'Parameter': 'Discrimination (a)', 'Dimension': dim_name(dim),
                             **dim_metrics[dim], 'N Items': int(mask_dim.sum())})
    dim_rows.append({'Parameter': 'Intercept (d)', 'Dimension': '–', **d_metrics, 'N Items': len(true_d)})
    df_bias_rmse = pd.DataFrame(dim_rows)

    panels = [(f"Slope a — {dim_name(dim)}", true_a[q_matrix[:, dim] == 1, dim], est_a[q_matrix[:, dim] == 1, dim], BLUE)
              for dim in dim_metrics] + [("Intercept d", true_d, est_d, GREEN)]
    for column, (label, truth, estimate, color) in zip(chart_grid(len(panels)), panels):
        with column:
            recovery_scatter(truth, estimate, label, color)
    worst = max(dim_metrics, key=lambda k: dim_metrics[k]['RMSE'])
    best = min(dim_metrics, key=lambda k: dim_metrics[k]['RMSE'])
    extra = ""
    if structure == "Bifactor":
        spec = [dim_metrics[k]['RMSE'] for k in dim_metrics if k > 0]
        extra = (f"\n\nIn this bifactor model the general factor's slopes have RMSE {dim_metrics[0]['RMSE']:.3f}, the "
                 f"specific factors' average {np.mean(spec):.3f}. Specific slopes are usually harder to recover because "
                 "each specific factor has fewer items and shares them with G.")
    elif structure == "Cross-load":
        cross = q_matrix.sum(axis=1) > 1
        errs_cross = (est_a - true_a)[(q_matrix == 1) & cross[:, None]]
        errs_pure = (est_a - true_a)[(q_matrix == 1) & ~cross[:, None]]
        if errs_cross.size:
            extra = (f"\n\nSlopes of cross-loading items have RMSE {np.sqrt(np.mean(errs_cross**2)):.3f} versus "
                     f"{np.sqrt(np.mean(errs_pure**2)):.3f} for single-dimension items: when an item measures two "
                     "dimensions, the data must be split between two slopes.")
    interpret(
        "Each dot is one item: **true** value (x) against **estimate** (y). Dots on the dashed line are recovered "
        "perfectly. **Bias** is the average error (positive = overestimated), **RMSE** is the typical size of an "
        "error in the parameter's own units, and **r** tells whether items are ranked in the right order. "
        "(Rough guide for simulations: RMSE below 0.10 is excellent, below 0.20 good.)\n\n"
        f"- **Slopes (all dimensions):** RMSE {overall_a_metrics['RMSE']:.3f} — {quality(overall_a_metrics['RMSE'])}; "
        f"bias {overall_a_metrics['Mean Bias']:+.3f}; r {overall_a_metrics['Corr']:.3f}.\n"
        f"- **Intercepts:** RMSE {d_metrics['RMSE']:.3f} — {quality(d_metrics['RMSE'])}; r {d_metrics['Corr']:.3f}. "
        "Intercepts are usually recovered better than slopes because every response informs them.\n"
        f"- Best-recovered dimension: **{dim_name(best)}** (RMSE {dim_metrics[best]['RMSE']:.3f}); hardest: "
        f"**{dim_name(worst)}** (RMSE {dim_metrics[worst]['RMSE']:.3f})." + extra +
        "\n\nLarger samples (N) and more items per dimension reduce these errors.")
    st.markdown("**Recovery summary**")
    show_table(df_bias_rmse)

    st.markdown("**Loading matrix: true vs estimated** (gray cells are fixed at zero by the design)")
    vmax = float(max(true_a.max(), est_a.max()))
    column_labels = [dim_short(k) for k in range(mhrm.D)]
    left, right = st.columns(2)
    with left:
        loading_heatmap(true_a, q_matrix == 1, "True slopes", column_labels, vmax)
    with right:
        loading_heatmap(est_a, q_matrix == 1, "Estimated slopes", column_labels, vmax)
    st.markdown("**Distribution of item parameters**")
    left, right = st.columns(2)
    for column, (label, truth, estimate) in zip((left, right), [("Slopes (a), active only", true_a[q_matrix == 1], est_a[q_matrix == 1]),
                                                                 ("Intercepts (d)", true_d, est_d)]):
        with column:
            fig, ax = plt.subplots(figsize=GRID2)
            bins = np.histogram_bin_edges(np.r_[truth, estimate], bins=12)
            ax.hist(truth, bins=bins, alpha=.55, color=MUTED, label='True')
            ax.hist(estimate, bins=bins, alpha=.55, color=BLUE, label='Estimated')
            ax.set(title=label, xlabel='Value', ylabel='Items')
            ax.legend(frameon=False)
            fig.tight_layout()
            show_fig(fig)
    interpret("The loading heatmaps should show the same pattern: bright cells where the design allows a slope and "
              "similar colours in both panels. The histograms compare the overall spread of true and estimated "
              "parameters; with few items per dimension, individual bars differ even when recovery is good.")

    with st.expander("Item-level tables (true, estimated, and bias for every item)"):
        item_rows = []
        for j in range(mhrm.J):
            row = {'Item': j + 1, 'True d': true_d[j], 'Est d': est_d[j], 'Bias d': est_d[j] - true_d[j]}
            for dim in range(mhrm.D):
                if q_matrix[j, dim] == 1:
                    tag = dim_short(dim)
                    row[f'True a ({tag})'] = true_a[j, dim]
                    row[f'Est a ({tag})'] = est_a[j, dim]
                    row[f'Bias a ({tag})'] = est_a[j, dim] - true_a[j, dim]
            item_rows.append(row)
        st.caption("Blank cells are slopes fixed at zero by the design.")
        show_table(pd.DataFrame(item_rows))

    # ==========================================================
    # Step 5 · Latent traits
    # ==========================================================
    st.subheader("Step 5 · Latent traits (θ)")
    theta_stats = []
    for column, dim in zip(chart_grid(mhrm.D, per_row=2 if mhrm.D == 4 else 3), range(mhrm.D)):
        draws = mhrm.current_theta[:, dim]
        theta_stats.append((dim, draws.mean(), draws.std(), np.corrcoef(true_theta[:, dim], draws)[0, 1]))
        with column:
            fig, ax = plt.subplots(figsize=GRID2 if mhrm.D == 4 else GRID3)
            bins = np.linspace(-4, 4, 33)
            ax.hist(true_theta[:, dim], bins=bins, alpha=.55, color=MUTED, label='True θ')
            ax.hist(draws, bins=bins, alpha=.55, color=BLUE, label='Posterior draw')
            ax.set(title=f"θ — {dim_name(dim)}", xlabel='θ', ylabel='Examinees')
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)   # headroom so the legend never covers bars
            ax.legend(frameon=False, loc='upper left')
            fig.tight_layout()
            show_fig(fig)
    interpret(
        "Gray bars are the examinees' true abilities; blue bars are **one** posterior draw per examinee from the "
        "final MH step. A draw is a plausible value, not a score estimate: it includes random noise, so it is not "
        "the same as an EAP or MAP ability estimate.\n\n" +
        "\n".join(f"- **{dim_name(k)}:** draws have mean {m:+.2f} and SD {s:.2f} (the model assumes 0 and 1); "
                  f"correlation with true θ is {r:.2f}." for k, m, s, r in theta_stats) +
        "\n\nDraws should look like a standard normal bell curve. Correlations are well below 1 because a single "
        "draw is noisy, and dimensions measured by fewer items are noisier still.")

    # ==========================================================
    # Step 6 · Model fit
    # ==========================================================
    st.subheader("Step 6 · Model fit: log-likelihood")
    left, right = st.columns([1, 2.2])
    with left:
        if python_marginal is not None:
            final_ll, integration_method = python_marginal
            st.metric("Final marginal log-likelihood", f"{final_ll:,.1f}")
            st.caption(f"{integration_method}.")
        else:
            st.caption("Turn on *Estimate Final Marginal Log-Likelihood* in the sidebar to compute it.")
    with right:
        fig, ax = plt.subplots(figsize=(6.6, 2.6))
        ax.plot(mhrm.iteration_history, mhrm.loss_history, color=BLUE, lw=1.6)
        for edge, name in ((burn_in, 'SEM starts'), (burn_in + sem_cycles, 'RM starts')):
            if edge < mhrm.n_iter_:
                ax.axvline(edge, color=MUTED, lw=.8, ls=':')
                ax.text(edge, ax.get_ylim()[0], f' {name}', color=MUTED, fontsize=8, va='bottom')
        ax.set(title='Sampled conditional log-likelihood', xlabel='Total cycle', ylabel='Conditional LL')
        ax.grid(alpha=.3)
        fig.tight_layout()
        show_fig(fig)
    interpret(
        "The **conditional log-likelihood** (right) measures how well the item parameters fit the sampled "
        "abilities. It climbs steeply during burn-in and then levels off; small wiggles afterwards are sampling "
        "noise, so it does not have to increase every cycle.\n\n"
        "The **marginal log-likelihood** (left) is the quantity MHRM actually maximizes: abilities are averaged "
        "out by numerical integration. It is always negative; higher (closer to zero) means better fit. On its "
        "own the number has no fixed scale — it is used to **compare** models fitted to the same data (for "
        "example Simple vs Cross-load, or Python vs R in Step 7).")

    # ==========================================================
    # Step 7 · Cross-check against R mirt
    # ==========================================================
    if compare_r:
        st.subheader("Step 7 · Cross-check against R mirt output")
        st.caption("Same response matrix, Q-matrix, positive slope bounds, and fixed N(0,I) distribution. "
                   "Initialization, proposal tuning, and stopping rules can differ.")
        if r_error is not None:
            st.error(f"R cross-check could not finish: {r_error}")
            st.caption("The Python calibration and its diagnostics above remain available.")
        else:
            convergence_table = pd.DataFrame([
                dict(Estimator='Python', Converged=mhrm.converged, RM_cycles=mhrm.rm_iterations_,
                     Total_cycles=mhrm.n_iter_, Seconds=elapsed_seconds),
                dict(Estimator=f"R mirt {reference['mirt_version']}", Converged=reference['converged'],
                     RM_cycles=reference['rm_cycles'],
                     Total_cycles=burn_in+sem_cycles+reference['rm_cycles'], Seconds=reference['seconds'])])
            show_table(convergence_table, digits=1)
            st.caption("Python seconds exclude live-chart drawing; R seconds include starting R and loading mirt.")
            if not (mhrm.converged and reference['converged']):
                st.warning("At least one estimator did not converge. All agreement statistics below are provisional.")
            summary = comparison['summary']
            slopes_row = summary.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Slope difference (RMSE)", f"{slopes_row.RMSE:.4f}")
            m2.metric("Intercept difference (RMSE)", f"{summary.iloc[1].RMSE:.4f}")
            m3.metric("Integrated log-likelihood difference (Python − R)", f"{comparison['loglik_difference']:.4f}")
            left, right = st.columns(2)
            for column, (label, py, r) in zip((left, right), [
                    ('Active slopes', est_a[q_matrix == 1], reference['a'][q_matrix == 1]),
                    ('Intercepts', est_d, reference['d'])]):
                with column:
                    fig, ax = plt.subplots(figsize=GRID2)
                    lo, hi = min(np.min(r), np.min(py)), max(np.max(r), np.max(py))
                    ax.plot([lo, hi], [lo, hi], ls='--', color=MUTED, lw=1)
                    ax.scatter(r, py, s=26, color=BLUE, edgecolors='white', linewidths=.6)
                    ax.set(title=label, xlabel='R mirt', ylabel='Python'); ax.grid(alpha=.3)
                    fig.tight_layout()
                    show_fig(fig)
            ll_gap = abs(comparison['loglik_difference'])
            interpret(
                "R's **mirt** package is the standard implementation of MHRM, so it serves as an independent check "
                "of this Python code on the **same data**.\n\n"
                f"- Slopes differ by RMSE {slopes_row.RMSE:.4f} (largest single difference "
                f"{slopes_row.Max_absolute_difference:.3f}) and intercepts by {summary.iloc[1].RMSE:.4f}. "
                + ("These differences are much smaller than either estimator's error against the true values "
                   f"(slope RMSE {overall_a_metrics['RMSE']:.3f}), so both land in essentially the same place."
                   if slopes_row.RMSE < overall_a_metrics['RMSE'] / 2 else
                   "These differences are comparable to the recovery error, so the two fits disagree noticeably — "
                   "check convergence and try more cycles.") + "\n"
                f"- The integrated log-likelihoods differ by {comparison['loglik_difference']:+.3f} "
                + ("— practically identical fit." if ll_gap < 1 else
                   "— a noticeable difference; the higher value fits these data better.") + "\n\n"
                "Small differences are expected: the two programs use different random draws, starting values, and "
                "stopping rules. Agreement shows the Python implementation behaves like the reference; it does not "
                "prove either one found the global maximum.")
            st.markdown("**Parameter agreement: Python minus R**")
            show_table(summary, digits=4)
            st.markdown("**Recovery against the simulated true parameters**")
            show_table(comparison['recovery'], digits=4)
            with st.expander("Item coefficients from both estimators, downloads, and R details"):
                show_table(comparison['per_item'], digits=4)
                b1, b2 = st.columns(2)
                with b1:
                    st.download_button("Download Python–R item comparison CSV", comparison['per_item'].to_csv(index=False),
                                       'mhrm_python_r_items.csv', 'text/csv')
                with b2:
                    st.download_button("Download agreement summary CSV", summary.to_csv(index=False),
                                       'mhrm_python_r_agreement.csv', 'text/csv')
                st.markdown("**Marginal likelihood using common integration nodes**")
                show_table(comparison['likelihood'], digits=3)
                st.caption(comparison['integration_method'] + ". Both fits use identical nodes and weights.")
                st.code(reference['model_specification'], language='r')
                st.json(dict(R=reference['r_version'], mirt=reference['mirt_version'], **reference['settings']))
                st.write("R-native marginal log-likelihood (separate numerical approximation):", reference['native_loglik'])
                st.code(reference['messages'] or 'No R warnings or messages.')
    plt.close("all")

else:
    st.info("Adjust settings in the sidebar and click Run to start.")

# --- Explanation Section ---
with st.expander("ℹ️ How does this algorithm work?"):
    st.markdown(r"""
    **1. Model, notation, and estimation target**

    Let $Y_{ij}\in\{0,1\}$ denote the response of person $i=1,\ldots,N$
    to item $j=1,\ldots,J$. Each person has a $D$-dimensional latent ability
    vector. This app fixes the latent distribution rather than estimating its
    mean or covariance:
    """)
    st.latex(r"\boldsymbol{\theta}_i\overset{\mathrm{iid}}{\sim}\mathcal{N}_D(\mathbf{0},\mathbf{I}_D).")
    st.markdown(r"The compensatory multidimensional 2PL model is")
    st.latex(r"""
        \begin{aligned}
        \eta_{ij}&=\mathbf{a}_j^{\mathsf T}\boldsymbol{\theta}_i+d_j,\\
        p_{ij}(\boldsymbol{\theta}_i;\boldsymbol{\xi})
        &=\Pr(Y_{ij}=1\mid\boldsymbol{\theta}_i,\boldsymbol{\xi})
        =\frac{1}{1+\exp(-\eta_{ij})}.
        \end{aligned}
    """)
    st.markdown(r"""
    Responses are conditionally independent given ability. The vector
    $\boldsymbol{\xi}$ collects all free item coefficients. The loading mask
    $Q_{jr}$ fixes inactive slopes to zero; active slopes are constrained to
    $[0,10]$, while intercepts $d_j$ are unrestricted. Define
    $\mathcal{S}_j=\{r:Q_{jr}=1\}$ and the item-specific coefficient and design vectors:
    """)
    st.latex(r"""
        \boldsymbol{\beta}_j=
        \begin{pmatrix}\mathbf{a}_{j,\mathcal{S}_j}\\d_j\end{pmatrix},
        \qquad
        \mathbf{x}_{ij}=
        \begin{pmatrix}\boldsymbol{\theta}_{i,\mathcal{S}_j}\\1\end{pmatrix},
        \qquad \eta_{ij}=\mathbf{x}_{ij}^{\mathsf T}\boldsymbol{\beta}_j.
    """)
    st.markdown(r"""
    The intercept is not an item difficulty parameter. In one dimension,
    $b_j=-d_j/a_j$ for $a_j\ne0$. In multiple dimensions, the 50% success
    contour is $\mathbf{a}_j^{\mathsf T}\boldsymbol{\theta}+d_j=0$.

    The target is the constrained **marginal maximum-likelihood** estimate:
    """)
    st.latex(r"""
        \begin{aligned}
        \ell(\boldsymbol{\xi};\mathbf{Y})
        &=\sum_{i=1}^{N}\log\!\int_{\mathbb{R}^{D}}
        \left\{\prod_{j=1}^{J}
        p_{ij}(\boldsymbol{\theta};\boldsymbol{\xi})^{Y_{ij}}
        [1-p_{ij}(\boldsymbol{\theta};\boldsymbol{\xi})]^{1-Y_{ij}}
        \right\}\phi_D(\boldsymbol{\theta})\,d\boldsymbol{\theta},\\
        \widehat{\boldsymbol{\xi}}
        &\in\underset{\boldsymbol{\xi}\in\mathcal{C}}{\operatorname{arg\,max}}
        \;\ell(\boldsymbol{\xi};\mathbf{Y}).
        \end{aligned}
    """)
    st.markdown(r"""
    Here $\phi_D$ is the standard multivariate normal density and $\mathcal{C}$
    contains the loading-mask and slope-bound constraints. Ordinary product
    quadrature becomes expensive as $D$ increases; there is no universal
    dimension at which integration becomes mathematically impossible.

    **2. Metropolis–Hastings: sample abilities at the current item coefficients**

    During each outer cycle, hold the current item coefficients fixed while
    running MH sweeps. For person $i$, the posterior log density, up to a
    constant independent of $\boldsymbol{\theta}$, is
    """)
    st.latex(r"""
        \log\pi_i(\boldsymbol{\theta}\mid\mathbf{y}_i,\boldsymbol{\xi})
        =\sum_{j=1}^{J}\!\left[
        y_{ij}\eta_{ij}(\boldsymbol{\theta})
        -\log\{1+\exp(\eta_{ij}(\boldsymbol{\theta}))\}\right]
        -\frac{1}{2}\boldsymbol{\theta}^{\mathsf T}\boldsymbol{\theta}+C_i.
    """)
    st.markdown(r"The app uses a symmetric Gaussian random-walk proposal:")
    st.latex(r"""
        \begin{aligned}
        \boldsymbol{\theta}_i^*\mid\boldsymbol{\theta}_i
        &\sim\mathcal{N}_D(\boldsymbol{\theta}_i,s^2\mathbf{I}_D),\\
        \Delta_i
        &=\log\pi_i(\boldsymbol{\theta}_i^*)
          -\log\pi_i(\boldsymbol{\theta}_i),\\
        \alpha_i&=\min\{1,\exp(\Delta_i)\},\qquad U_i\sim\operatorname{Uniform}(0,1),\\
        \boldsymbol{\theta}_i^{\mathrm{new}}
        &=\begin{cases}
          \boldsymbol{\theta}_i^*,&\log U_i<\min(0,\Delta_i),\\
          \boldsymbol{\theta}_i,&\text{otherwise}.
        \end{cases}
        \end{aligned}
    """)
    st.markdown(r"""
    The proposal-density terms cancel because the proposal is symmetric.
    The normal-prior term must remain in the acceptance ratio. Proposal
    standard deviation $s$ is tuned during stage 1 and frozen afterward.
    After the configured sweeps, the last draw for each person is used in
    that cycle's item update. A posterior draw is not an EAP ability estimate.

    **3. Stochastic score and full item curvature**

    At the current item coefficients and sampled abilities, the normalized
    complete-data score and positive curvature for item $j$ are
    """)
    st.latex(r"""
        \begin{aligned}
        \mathbf{g}_{j,t}
        &=\frac{1}{N}\sum_{i=1}^{N}
          \mathbf{x}_{ij}^{(t)}\bigl(y_{ij}-p_{ij}^{(t)}\bigr),\\
        \mathbf{H}_{j,t}
        &=\frac{1}{N}\sum_{i=1}^{N}
          p_{ij}^{(t)}\bigl(1-p_{ij}^{(t)}\bigr)
          \mathbf{x}_{ij}^{(t)}\mathbf{x}_{ij}^{(t)\mathsf T}
        =-\frac{1}{N}\nabla_{\boldsymbol{\beta}_j}^{2}
          \ell_{\mathrm{cond}}(\boldsymbol{\xi};\mathbf{Y},\boldsymbol{\Theta}^{(t)}).
        \end{aligned}
    """)
    st.markdown(r"""
    $\mathbf{H}_{j,t}$ is the **negative** conditional log-likelihood Hessian,
    so the maximization update uses a plus sign. It contains all slope–slope
    and slope–intercept terms. Because the latent distribution is fixed,
    different items have zero cross-item complete-data Hessian blocks.
    For nonsingular curvature, normalizing both score and curvature by $N$
    leaves the unstabilized Newton correction unchanged. This curvature is not the observed marginal information matrix
    used to calculate standard errors.

    The reason posterior sampling can guide marginal-likelihood estimation
    is Fisher's identity. With an exact posterior expectation,
    """)
    st.latex(r"""
        \nabla_{\boldsymbol{\xi}}\frac{\ell(\boldsymbol{\xi};\mathbf{Y})}{N}
        =\operatorname{E}_{\boldsymbol{\Theta}\mid\mathbf{Y},\boldsymbol{\xi}}
        \!\left[\nabla_{\boldsymbol{\xi}}
        \frac{\ell_{\mathrm{cond}}(\boldsymbol{\xi};\mathbf{Y},\boldsymbol{\Theta})}{N}
        \right].
    """)
    st.markdown(r"""
    The latent prior has no item-parameter derivatives in this model.
    Finite MH sweeps provide a Monte Carlo approximation; adequate mixing
    matters, and individual updates need not increase the marginal likelihood.

    **4. Stages 1 and 2: constant-gain updates and SEM averaging**

    Stage 1 is burn-in; stage 2 is stochastic-EM (SEM) averaging. In both stages,
    the implementation uses a gain of $0.2$ and clips each component of the
    Newton correction to $[-1,1]$:
    """)
    st.latex(r"""
        \begin{aligned}
        \mathbf{v}_{j,t}&=\widetilde{\mathbf{H}}_{j,t}^{-1}\mathbf{g}_{j,t},\\
        \boldsymbol{\beta}_{j,t}^{+}
        &=\mathcal{B}_{\boldsymbol{\beta}_{j,t}}\!\left(
        \boldsymbol{\beta}_{j,t}
        +0.2\,\operatorname{clip}(\mathbf{v}_{j,t},-1,1)\right).
        \end{aligned}
    """)
    st.markdown(r"""
    A tilde denotes the numerically stabilized matrix used for the solve.
    The operator $\mathcal{B}$ applies the bound safeguard defined below.
    Over the $M$ stage-2 cycles, the app averages the **updated** item
    coefficients and the curvature computed before each corresponding update:
    """)
    st.latex(r"""
        \boldsymbol{\beta}_{j,0}^{\mathrm{RM}}
        =\frac{1}{M}\sum_{s=1}^{M}\boldsymbol{\beta}_{j,s}^{\mathrm{SEM},+},
        \qquad
        \boldsymbol{\Gamma}_{j,0}
        =\frac{1}{M}\sum_{s=1}^{M}\mathbf{H}_{j,s}^{\mathrm{SEM}}.
    """)
    st.markdown(r"""
    These averages initialize stage 3. The SEM average is not an average of
    the later RM estimates, and the app does not average all MH draws within
    a cycle.

    **5. Stage 3: Robbins–Monro with smoothed curvature**

    Let $k=1,2,\ldots,K$ count **RM cycles only**. Each cycle samples abilities
    at $\boldsymbol{\beta}_{k-1}$, evaluates the score and curvature there, and updates
    """)
    st.latex(r"""
        \begin{aligned}
        \gamma_k&=\left(\frac{c}{k}\right)^{\alpha},
          &0<c\le1,\quad\frac{1}{2}<\alpha\le1,\\
        \boldsymbol{\Gamma}_{j,k}
        &=(1-\gamma_k)\boldsymbol{\Gamma}_{j,k-1}
          +\gamma_k\mathbf{H}_{j,k},\\
        \boldsymbol{\beta}_{j,k}
        &=\mathcal{B}_{\boldsymbol{\beta}_{j,k-1}}\!\left(
          \boldsymbol{\beta}_{j,k-1}
          +\gamma_k\widetilde{\boldsymbol{\Gamma}}_{j,k}^{-1}
          \mathbf{g}_{j,k}\right).
        \end{aligned}
    """)
    st.markdown(r"""
    The default values are $c=0.1$ and $\alpha=0.75$. This gain sequence satisfies
    """)
    st.latex(r"\sum_{k=1}^{\infty}\gamma_k=\infty,\qquad \sum_{k=1}^{\infty}\gamma_k^2<\infty.")
    st.markdown(r"""
    The app smooths curvature across RM cycles and uses the current sampled
    score. It does not clip the RM correction to $[-1,1]$; that clipping is
    restricted to stages 1 and 2.

    **6. Numerical stabilization and slope bounds**

    For a curvature matrix $\mathbf{A}$, the solve uses
    """)
    st.latex(r"""
        \widetilde{\mathbf{A}}=\mathbf{A}+\delta\mathbf{I},\qquad
        \delta=\max\!\left(0,10^{-6}-\lambda_{\min}(\mathbf{A})\right).
    """)
    st.markdown(r"""
    This stabilizes the linear solve; it does not add an item-parameter prior
    to the likelihood. For each active slope, with old value $a$ and proposed
    value $z$, the implementation uses a halfway-to-bound safeguard:
    """)
    st.latex(r"""
        \mathcal{B}_{a}(z)=
        \begin{cases}
        a/2,&z<0,\\
        (a+10)/2,&z>10,\\
        z,&0\le z\le10.
        \end{cases}
    """)
    st.markdown(r"""
    Intercepts are unchanged by this bound operator. Inactive slopes remain
    exactly zero throughout estimation.

    **7. Stopping rule and interpretation of diagnostics**

    Stack all free coefficients into $\boldsymbol{\beta}_k$ and all item scores
    into $\mathbf{g}_k$. Once 30 RM scores are available, define
    """)
    st.latex(r"""
        \overline{\mathbf{g}}_k
        =\frac{1}{30}\sum_{r=k-29}^{k}\mathbf{g}_r,
        \qquad
        \Delta_k=\left\|\boldsymbol{\beta}_k-
        \boldsymbol{\beta}_{k-1}\right\|_{\infty}.
    """)
    st.markdown(r"""
    For the projected score $\mathbf{g}_k^{\mathrm{proj}}$, an outward-pointing
    component of $\overline{\mathbf{g}}_k$ is set to zero when its slope is within
    $10^{-6}$ of the relevant bound. Other components, including intercept
    components, are retained. Under the app's default minimum of 50 RM cycles,
    all of the following must hold in **three consecutive cycles**:
    """)
    st.latex(r"""
        k\ge50,\qquad \Delta_k<\tau,\qquad
        \left\|\mathbf{g}_k^{\mathrm{proj}}\right\|_{\infty}<0.005.
    """)
    st.markdown(r"""
    $\tau$ is the selected maximum-change tolerance, defaulting to $0.001$.
    This is the app's numerical stopping rule, including an additional
    averaged-score check; it is not a universal convergence theorem or an
    exact reproduction of every stopping option in R mirt. Reaching the RM
    cycle limit is reported as nonconvergence.

    The live trace is the **conditional response log-likelihood at sampled
    abilities**:
    """)
    st.latex(r"""
        \ell_{\mathrm{cond}}(\boldsymbol{\xi};\mathbf{Y},\boldsymbol{\Theta})
        =\sum_{i=1}^{N}\sum_{j=1}^{J}
          \left[y_{ij}\eta_{ij}-\log\{1+\exp(\eta_{ij})\}\right].
    """)
    st.markdown(r"""
    It excludes the ability-prior term, is not the marginal log-likelihood,
    and need not increase monotonically. The optional final marginal
    log-likelihood integrates out abilities separately: Gauss–Hermite
    quadrature for one/two dimensions and a labeled Sobol QMC approximation
    for higher dimensions. Neither diagnostic proves a global optimum,
    identification, or adequate mixing. This app does not compute standard errors.

    **References:** [Cai (2010)](https://doi.org/10.1007/s11336-009-9136-x),
    [mirt documentation](https://philchalmers.github.io/mirt/reference/mirt.html),
    [mirt MHRM source](https://github.com/philchalmers/mirt/blob/main/R/MHRM.group.R),
    [mirt derivative/update source](https://github.com/philchalmers/mirt/blob/main/R/MHRM.utils.R).
    """)
