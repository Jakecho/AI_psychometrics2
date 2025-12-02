"""MHRM_2D Streamlit App
Demonstrates Metropolis-Hastings Robbins-Monro (MHRM) calibration for a
confirmatory Multidimensional 2PL (M2PL) IRT model.
Converted from a notebook: removed Jupyter cell magic.
"""
# Cache refresh marker
# Updated on 2025-11-23 to clear residual notebook magic syntax error.
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import math

# ==========================================
# 1. CORE MATH & STATISTICS FUNCTIONS
# ==========================================

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def prob_m2pl(theta, a, d):
    """
    Calculate P(Y=1) for Multidimensional 2PL.
    theta: (N, D) matrix of latent traits
    a: (J, D) matrix of discrimination parameters
    d: (J,) vector of intercepts
    Returns: (N, J) matrix of probabilities
    """
    # Logit = theta * a^T + d
    logits = np.dot(theta, a.T) + d
    return sigmoid(logits)

def log_likelihood(y, theta, a, d):
    """Compute complete data log-likelihood."""
    P = prob_m2pl(theta, a, d)
    epsilon = 1e-9
    P = np.clip(P, epsilon, 1 - epsilon)
    return np.sum(y * np.log(P) + (1 - y) * np.log(1 - P))

def simulate_data(N, J, D, q_matrix, true_a=None, true_d=None):
    """
    Simulate response data for M2PL.
    N: Number of examinees
    J: Number of items
    D: Number of dimensions
    q_matrix: (J, D) binary mask for confirmatory structure
    true_a: Optional pre-defined discrimination parameters
    true_d: Optional pre-defined intercept parameters
    """
    np.random.seed(42)

    # True Theta ~ N(0, I) (Scale Identification)
    true_theta = np.random.normal(0, 1, size=(N, D))

    # True Parameters
    if true_a is None:
        # Discriminations (a): Uniform(0.5, 2.0) masked by Q-matrix
        true_a = np.random.uniform(0.5, 2.0, size=(J, D)) * q_matrix

    if true_d is None:
        # Intercepts (d): Uniform(-2, 2)
        true_d = np.random.uniform(-2, 2, size=J)

    # Generate Probabilities
    P = prob_m2pl(true_theta, true_a, true_d)

    # Generate Responses (Bernoulli)
    data = (np.random.random(size=(N, J)) < P).astype(int)

    return data, true_theta, true_a, true_d

def plot_path_diagram(q_matrix, structure_type):
    """
    Visualizes the MIRT model structure as a path diagram.
    Vertical format: Factors on Left, Items on Right.
    For Bifactor: General Factor Left, Items Middle, Specific Factors Right.
    """
    J, D = q_matrix.shape
    # Adjust height based on J
    fig, ax = plt.subplots(figsize=(8, max(6, J * 0.3)))
    
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
    ax.scatter(factor_x, factor_y, s=600, c='skyblue', edgecolors='black', zorder=10)
    for d in range(D):
        if structure_type == "Bifactor":
            label = "G" if d == 0 else f"S_{d}"
        else:
            label = f"F{d+1}"
        ax.text(factor_x[d], factor_y[d], label, ha='center', va='center', fontweight='bold')
        
    # Items
    ax.scatter(item_x, item_y, s=100, c='white', edgecolors='black', zorder=10)
    # Label all items
    for j in range(J):
        ax.text(item_x[j], item_y[j], str(j+1), ha='center', va='center', fontsize=7)
            
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Path Diagram: {structure_type}")
    return fig

# ==========================================
# 2. MHRM ALGORITHM CLASS
# ==========================================

class MHRM_MIRT:
    def __init__(self, data, dims, q_matrix):
        self.Y = data
        self.N, self.J = data.shape
        self.D = dims
        self.Q = q_matrix

        # Initialize Parameters (Random starts)
        np.random.seed(101)
        self.est_a = np.random.uniform(0.5, 1.0, size=(self.J, self.D)) * self.Q
        self.est_d = np.zeros(self.J)

        # Initial Theta guesses (Start at 0)
        self.current_theta = np.zeros((self.N, self.D))

        # History for plotting
        self.loss_history = []

    def mh_step(self, n_steps=5, candidate_sd=0.5):
        """
        Metropolis-Hastings Step:
        Impute missing theta values from the posterior P(theta | Y, params).
        Using a Random Walk Metropolis sampler.
        Prior P(theta) is N(0, I) for identification.
        """
        accepted_count = 0
        total_count = 0

        for _ in range(n_steps):
            # 1. Propose new theta: theta_new ~ N(theta_curr, sd)
            proposal = self.current_theta + np.random.normal(0, candidate_sd, size=(self.N, self.D))

            # 2. Calculate Acceptance Ratio
            # Log Posterior ~ Log Likelihood + Log Prior

            # Current LogL
            P_curr = prob_m2pl(self.current_theta, self.est_a, self.est_d)
            epsilon = 1e-9
            P_curr = np.clip(P_curr, epsilon, 1-epsilon)
            ll_curr = np.sum(self.Y * np.log(P_curr) + (1-self.Y) * np.log(1-P_curr), axis=1)
            # Log Prior (Normal) - dropping constants
            lprior_curr = -0.5 * np.sum(self.current_theta**2, axis=1)
            log_post_curr = ll_curr + lprior_curr

            # Proposed LogL
            P_prop = prob_m2pl(proposal, self.est_a, self.est_d)
            P_prop = np.clip(P_prop, epsilon, 1-epsilon)
            ll_prop = np.sum(self.Y * np.log(P_prop) + (1-self.Y) * np.log(1-P_prop), axis=1)
            lprior_prop = -0.5 * np.sum(proposal**2, axis=1)
            log_post_prop = ll_prop + lprior_prop

            # 3. Accept/Reject (Vectorized for all N)
            log_ratio = log_post_prop - log_post_curr
            u = np.log(np.random.random(self.N))

            accept_mask = u < log_ratio
            self.current_theta[accept_mask] = proposal[accept_mask]

            accepted_count += np.sum(accept_mask)
            total_count += self.N

        return accepted_count / total_count

    def rm_step(self, learning_rate):
        """
        Robbins-Monro Step:
        Update item parameters (a, d) using stochastic gradients.
        """
        # Calculate probabilities with current imputed theta
        P = prob_m2pl(self.current_theta, self.est_a, self.est_d) # (N, J)
        residual = self.Y - P # (N, J)
        W = P * (1 - P) # (N, J)

        # Gradients
        # Grad_d: Sum over N of (Y - P)
        grad_d = np.sum(residual, axis=0) # (J,)

        # Grad_a: Sum over N of (Y - P) * theta
        # Reshape for broadcasting: (N, J, 1) * (N, 1, D) -> (N, J, D)
        grad_a = np.dot(residual.T, self.current_theta) # (J, D)

        # Hessian Approximations (Diagonal)
        # Hess_d: -Sum W
        hess_d = -np.sum(W, axis=0) # (J,)
        
        # Hess_a: -Sum W * theta^2
        # (N, J) -> (J, N) dot (N, D) -> (J, D)
        hess_a = -np.dot(W.T, self.current_theta**2) # (J, D)

        # Clamp Hessians to avoid instability (division by near-zero)
        hess_d = np.minimum(hess_d, -0.1)
        hess_a = np.minimum(hess_a, -0.1)

        # Update d (Newton-Raphson step)
        step_d = - (grad_d / hess_d) * learning_rate
        self.est_d += step_d

        # Update a (Newton-Raphson step)
        step_a = - (grad_a / hess_a) * learning_rate

        # Apply Q-matrix mask to gradient/step
        step_a = step_a * self.Q

        self.est_a += step_a

        # Constraints: Prevent 'a' from going negative to resolve sign indeterminacy
        self.est_a = np.maximum(self.est_a, 0)
        self.est_a = np.minimum(self.est_a, 10)

    def fit(self, max_iter=100, burn_in=20, decay=0.99, progress_bar=None, plot_placeholder=None, structure_type="Simple"):

        # Gain sequence
        gamma = 1.0

        for iteration in range(max_iter):
            # 1. Decay Learning Rate after burn-in
            if iteration > burn_in:
                gamma = 1.0 / (iteration - burn_in + 1)**decay
            else:
                gamma = 1.0 # Constant gain during burn-in

            # 2. MH Step (Impute Theta)
            # More steps during burn-in to ensure theta chain mixes well
            mh_steps = 5 if iteration > 5 else 20
            acc_rate = self.mh_step(n_steps=mh_steps)

            # 3. RM Step (Update Parameters)
            self.rm_step(learning_rate=gamma)

            # Logging
            if iteration % 5 == 0:
                current_ll = log_likelihood(self.Y, self.current_theta, self.est_a, self.est_d)
                self.loss_history.append(current_ll)
                if progress_bar:
                    progress_bar.progress((iteration + 1) / max_iter,
                                        text=f"Iter {iteration}: LL={current_ll:.0f}, MH Accept={acc_rate:.2f}, Gamma={gamma:.4f}")
                
                # Dynamic Visualization
                if plot_placeholder:
                    with plot_placeholder.container():
                        col1, col2 = st.columns(2)
                        
                        # Plot 1: MH Sampling (Theta Distribution)
                        with col1:
                            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
                            n_cols = min(3, max(1, self.D))
                            n_rows = math.ceil(self.D / n_cols)
                            fig_mh, axes = plt.subplots(n_rows, n_cols,
                                                        figsize=(n_cols * 3.2, n_rows * 2.5))
                            axes = np.array(axes).reshape(-1)

                            for d in range(self.D):
                                ax = axes[d]
                                if structure_type == "Bifactor":
                                    label = "General (G)" if d == 0 else f"Specific (S_{d})"
                                else:
                                    label = f"Dim {d+1}"

                                ax.hist(self.current_theta[:, d], bins=20,
                                        color=colors[d % len(colors)],
                                        alpha=0.5, density=True)
                                ax.set_title(label)
                                ax.set_xlabel("Theta")
                                ax.grid(alpha=0.2)

                            # Hide unused axes
                            for idx in range(self.D, len(axes)):
                                axes[idx].axis('off')

                            fig_mh.suptitle(f"MH Step: Theta Sampling (Iter {iteration})", fontsize=12)
                            fig_mh.tight_layout()
                            st.pyplot(fig_mh)
                            plt.close(fig_mh)

                        # Plot 2: RM Convergence (Log-Likelihood Trace)
                        with col2:
                            fig_rm, ax_rm = plt.subplots(figsize=(5, 3))
                            if len(self.loss_history) > 1:
                                ax_rm.plot(range(0, len(self.loss_history)*5, 5), self.loss_history, 'b-', lw=2)
                            ax_rm.set_title("RM Step: Convergence (Log-Likelihood)")
                            ax_rm.set_xlabel("Iteration")
                            ax_rm.set_ylabel("LL")
                            ax_rm.grid(True, alpha=0.3)
                            st.pyplot(fig_rm)
                            plt.close(fig_rm)

# ==========================================
# 3. STREAMLIT APP LAYOUT
# ==========================================

st.set_page_config(layout="wide", page_title="MIRT Calibration with MHRM")

st.title("🧩 MHRM Algorithm for MIRT Calibration")
st.markdown(r"""
This application demonstrates the **Metropolis-Hastings Robbins-Monro (MHRM)** algorithm
for calibrating a Multidimensional Item Response Theory (MIRT) model.
We use a **Confirmatory MIRT** structure to ensure parameter recoverability and identification.
""")

# --- Sidebar Controls ---
st.sidebar.header("1. Data Generation")
N = st.sidebar.slider("Sample Size (N)", 500, 5000, 2000, step=500)
J = st.sidebar.slider("Number of Items (J)", 10, 60, 30, step=5)
D = st.sidebar.slider("Dimensions (Specific Factors)", 1, 5, 2)

st.sidebar.header("Structure")
structure = st.sidebar.radio(
    "Loading Structure",
    ["Simple", "Cross-load", "Bifactor"],
    index=0,
    help="Simple: one loading per item. Cross-load: items may load on multiple dims. Bifactor: general factor + specific factors."
)

if structure == "Cross-load":
    cross_p = st.sidebar.slider(
        "Cross-loading probability per dimension",
        0.05, 0.80, 0.30, step=0.05,
        help="Probability an item loads on a given dimension (independent). Ensures at least one loading per item."
    )
    min_loads = st.sidebar.slider(
        "Minimum loadings per item",
        1, D, 1,
        help="Guarantee each item has at least this many non-zero discriminations."
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
max_iters = st.sidebar.slider("Max Iterations", 50, 1000, 500, step=50)
burn_in = st.sidebar.slider("Burn-in Period", 10, 200, 50)
decay_rate = st.sidebar.slider("Gain Decay Rate", 0.5, 1.0, 0.75, help="Controls how fast step size shrinks. Higher = faster decay.")

# --- Main Logic ---

if st.button("🚀 Run Simulation & Calibration", type="primary"):

    # 1. Setup Structure (Q-Matrix)
    if structure == "Simple":
        q_matrix = np.zeros((J, D))
        items_per_dim = J // D
        for d in range(D):
            start = d * items_per_dim
            end = (d + 1) * items_per_dim if d < D - 1 else J
            q_matrix[start:end, d] = 1.0
    elif structure == "Cross-load":
        q_matrix = np.zeros((J, D))
        for j in range(J):
            # Sample loadings independently
            loads = np.random.rand(D) < cross_p
            # Ensure minimum number of loadings
            if loads.sum() < min_loads:
                # Force highest random draws to be on until min_loads achieved
                idx_sorted = np.argsort(np.random.rand(D))
                loads[idx_sorted[:min_loads]] = True
            q_matrix[j, loads] = 1.0
    elif structure == "Bifactor":
        # Bifactor: add one general factor column (all ones) + D specific factors
        # Represent general as extra column at index 0, specifics 1..D
        D_total = D + 1
        q_matrix = np.zeros((J, D_total))
        # General factor loads on all items
        q_matrix[:, 0] = 1.0
        # Assign each item to exactly one specific factor for clarity
        items_per_dim = J // D
        for d in range(D):
            start = d * items_per_dim
            end = (d + 1) * items_per_dim if d < D - 1 else J
            q_matrix[start:end, d + 1] = 1.0
        # Adjust D to include general for downstream code
        D_effective = D_total
    else:
        q_matrix = np.zeros((J, D))  # fallback

    # Effective dimension count for model objects
    if structure == "Bifactor":
        dims_for_model = D_effective
    else:
        dims_for_model = D

    # 2. Simulate Data
    with st.spinner("Simulating Response Data..."):
        # Simulate data based on effective dimension count
        if structure == "Bifactor":
            # Generate parameters respecting the slider settings
            np.random.seed(42)
            # General Factor (Col 0)
            a_gen = np.random.uniform(0.5, general_strength, size=J)
            # Specific Factors (Cols 1..D)
            a_spec = np.random.uniform(0.5, specific_scale, size=(J, D))
            
            # Combine into true_a matrix
            true_a_sim = np.zeros((J, D_effective))
            true_a_sim[:, 0] = a_gen
            true_a_sim[:, 1:] = a_spec
            
            # Apply Q-matrix mask
            true_a_sim = true_a_sim * q_matrix
            
            response_data, true_theta, true_a, true_d = simulate_data(N, J, dims_for_model, q_matrix, true_a=true_a_sim)
        else:
            response_data, true_theta, true_a, true_d = simulate_data(N, J, D, q_matrix)

    st.success(f"Data Generated! Shape: {response_data.shape}. True scale fixed to N(0, I).")

    # 3. Run MHRM
    mhrm = MHRM_MIRT(response_data, dims_for_model, q_matrix)

    prog_bar = st.progress(0, text="Initializing MHRM...")
    plot_placeholder = st.empty() # Create placeholder for dynamic plots
    start_time = time.time()

    mhrm.fit(max_iter=max_iters, burn_in=burn_in, decay=decay_rate, progress_bar=prog_bar,
             plot_placeholder=plot_placeholder, structure_type=structure)

    end_time = time.time()
    st.info(f"Calibration finished in {end_time - start_time:.2f} seconds.")

    # 4. Evaluation
    est_a = mhrm.est_a
    est_d = mhrm.est_d

    # Bias and RMSE
    # Overall discrimination metrics (all non-zero loadings)
    bias_a = np.mean(est_a[q_matrix==1] - true_a[q_matrix==1])
    rmse_a = np.sqrt(np.mean((est_a[q_matrix==1] - true_a[q_matrix==1])**2))

    # Per-dimension discrimination metrics
    dim_rows = []
    for dim in range(mhrm.D):
        mask_dim = q_matrix[:, dim] == 1
        if np.sum(mask_dim) == 0:
            continue
        true_dim = true_a[mask_dim, dim]
        est_dim = est_a[mask_dim, dim]
        bias_dim = np.mean(est_dim - true_dim)
        rmse_dim = np.sqrt(np.mean((est_dim - true_dim)**2))
        
        if structure == "Bifactor":
            d_name = "General (G)" if dim == 0 else f"Specific (S_{dim})"
        else:
            d_name = f"Dim {dim+1}"

        dim_rows.append({
            'Parameter': 'Discrimination (a)',
            'Dimension': d_name,
            'Mean True': np.mean(true_dim),
            'Mean Est': np.mean(est_dim),
            'Mean Bias': bias_dim,
            'RMSE': rmse_dim,
            'N Items': np.sum(mask_dim)
        })

    bias_d = np.mean(est_d - true_d)
    rmse_d = np.sqrt(np.mean((est_d - true_d)**2))

    dim_rows.append({
        'Parameter': 'Intercept (d)',
        'Dimension': '-',
        'Mean True': np.mean(true_d),
        'Mean Est': np.mean(est_d),
        'Mean Bias': bias_d,
        'RMSE': rmse_d,
        'N Items': len(true_d)
    })

    df_bias_rmse = pd.DataFrame(dim_rows)

    # 5. Visualizations
    st.subheader("Model Structure Diagram")
    fig_diagram = plot_path_diagram(q_matrix, structure)
    st.pyplot(fig_diagram)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parameter Recovery: Discrimination (a) by Dimension")
        # Create per-dimension scatter plots
        for dim in range(mhrm.D):
            mask_dim = q_matrix[:, dim] == 1
            if np.sum(mask_dim) == 0:
                continue
            
            # Determine Label
            if structure == "Bifactor":
                dim_label = "General (G)" if dim == 0 else f"Specific (S_{dim})"
            else:
                dim_label = f"Dim {dim+1}"

            fig_dim, ax_dim = plt.subplots(figsize=(4,3))
            ax_dim.scatter(true_a[mask_dim, dim], est_a[mask_dim, dim], alpha=0.6, c='blue', edgecolors='k')
            min_val = min(np.min(true_a[mask_dim, dim]), np.min(est_a[mask_dim, dim]))
            max_val = max(np.max(true_a[mask_dim, dim]), np.max(est_a[mask_dim, dim]))
            ax_dim.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
            bias_dim = np.mean(est_a[mask_dim, dim] - true_a[mask_dim, dim])
            rmse_dim = np.sqrt(np.mean((est_a[mask_dim, dim] - true_a[mask_dim, dim])**2))
            ax_dim.set_xlabel("True a")
            ax_dim.set_ylabel("Est a")
            ax_dim.set_title(f"{dim_label}\nBias {bias_dim:.3f} RMSE {rmse_dim:.3f}")
            ax_dim.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig_dim)
        st.markdown(f"**Overall (All D) Bias:** {bias_a:.3f} | **RMSE:** {rmse_a:.3f}")

    with col2:
        st.subheader("Parameter Recovery: Intercept (d)")
        fig, ax = plt.subplots()
        ax.scatter(true_d, est_d, alpha=0.6, c='green', edgecolors='k')
        min_val = min(np.min(true_d), np.min(est_d))
        max_val = max(np.max(true_d), np.max(est_d))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel("True d")
        ax.set_ylabel("Estimated d")
        ax.set_title(f"Bias: {bias_d:.3f} | RMSE: {rmse_d:.3f}")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    st.subheader("Convergence Trace (Complete Data Log-Likelihood)")
    fig_conv, ax_conv = plt.subplots(figsize=(10, 3))
    ax_conv.plot(mhrm.loss_history, lw=2)
    ax_conv.set_xlabel("Iteration (Every 5th)")
    ax_conv.set_ylabel("Approx Log-Likelihood")
    ax_conv.set_title("Optimization Trajectory")
    st.pyplot(fig_conv)

    # Distribution plots for item parameters
    st.subheader("Distribution of Item Parameters")
    dist_cols = st.columns(2)
    with dist_cols[0]:
        fig_hist_a, ax_hist_a = plt.subplots(figsize=(5,3))
        ax_hist_a.hist(true_a[q_matrix==1], bins=20, alpha=0.5, label='True a')
        ax_hist_a.hist(est_a[q_matrix==1], bins=20, alpha=0.5, label='Est a')
        ax_hist_a.set_title("Discrimination (Aggregated)")
        ax_hist_a.legend()
        st.pyplot(fig_hist_a)
    with dist_cols[1]:
        fig_hist_d, ax_hist_d = plt.subplots(figsize=(5,3))
        ax_hist_d.hist(true_d, bins=20, alpha=0.5, label='True d')
        ax_hist_d.hist(est_d, bins=20, alpha=0.5, label='Est d')
        ax_hist_d.set_title("Intercepts")
        ax_hist_d.legend()
        st.pyplot(fig_hist_d)

    # Theta distribution per dimension
    st.subheader("Distribution of Latent Traits (Theta)")
    theta_cols = st.columns(min(3, mhrm.D))
    for dim in range(mhrm.D):
        with theta_cols[dim % len(theta_cols)]:
            # Determine Label
            if structure == "Bifactor":
                dim_label = "General (G)" if dim == 0 else f"Specific (S_{dim})"
            else:
                dim_label = f"Dim {dim+1}"
                
            fig_th, ax_th = plt.subplots(figsize=(4,3))
            ax_th.hist(true_theta[:, dim], bins=25, alpha=0.5, label='True θ')
            ax_th.hist(mhrm.current_theta[:, dim], bins=25, alpha=0.5, label='Imputed θ')
            ax_th.set_title(f"Theta {dim_label}")
            ax_th.legend(fontsize='small')
            st.pyplot(fig_th)

    # Bias / RMSE Summary Table
    st.subheader("Mean Bias & RMSE Summary")
    st.dataframe(df_bias_rmse.style.format({
        'Mean True': '{:.3f}',
        'Mean Est': '{:.3f}',
        'Mean Bias': '{:.3f}',
        'RMSE': '{:.3f}'
    }))

    # -------------------------------------------------
    # Per-Item Detailed Bias / RMSE Table
    # -------------------------------------------------
    st.subheader("Per-Item Parameter Biases")
    item_rows = []
    for j in range(mhrm.J):
        row = {
            'Item': j + 1,
            'True_d': true_d[j],
            'Est_d': est_d[j],
            'Bias_d': est_d[j] - true_d[j]
        }
        # Add discriminations for each dimension
        for dim in range(mhrm.D):
            if structure == "Bifactor":
                col_label = "G" if dim == 0 else f"S_{dim}"
            else:
                col_label = f"D{dim+1}"
                
            row[f'True_a_{col_label}'] = true_a[j, dim]
            row[f'Est_a_{col_label}'] = est_a[j, dim]
            row[f'Bias_a_{col_label}'] = est_a[j, dim] - true_a[j, dim]
        item_rows.append(row)
    df_items = pd.DataFrame(item_rows)
    # Formatting
    bias_format = {col: '{:.3f}' for col in df_items.columns if col not in ['Item']}
    st.dataframe(df_items.style.format(bias_format))

    # -------------------------------------------------
    # Bifactor Specific Displays
    # -------------------------------------------------
    if structure == "Bifactor":
        st.subheader("Bifactor: General vs Specific Factors")
        # General factor is column 0
        general_true = true_a[:, 0]
        general_est = est_a[:, 0]
        general_bias = general_est - general_true
        general_rmse = np.sqrt(np.mean((general_bias)**2))
        # Specifics are columns 1..D_effective-1
        specific_rows = []
        for dim in range(1, mhrm.D):
            spec_true = true_a[:, dim]
            # Only consider items that load (q_matrix[:,dim]==1)
            mask = q_matrix[:, dim] == 1
            if np.sum(mask) == 0:
                continue
            spec_true_masked = spec_true[mask]
            spec_est_masked = est_a[mask, dim]
            bias = np.mean(spec_est_masked - spec_true_masked)
            rmse = np.sqrt(np.mean((spec_est_masked - spec_true_masked)**2))
            specific_rows.append({
                'Specific Factor': f'SF {dim}',
                'Mean True a': np.mean(spec_true_masked),
                'Mean Est a': np.mean(spec_est_masked),
                'Mean Bias': bias,
                'RMSE': rmse,
                'Items': int(np.sum(mask))
            })
        df_general = pd.DataFrame({
            'Metric': ['Mean True a', 'Mean Est a', 'Mean Bias', 'RMSE'],
            'General Factor': [np.mean(general_true), np.mean(general_est), np.mean(general_bias), general_rmse]
        })
        st.markdown("**General Factor Summary**")
        st.dataframe(df_general.style.format({'General Factor': '{:.3f}'}))
        if specific_rows:
            st.markdown("**Specific Factor Summaries**")
            df_specific = pd.DataFrame(specific_rows)
            st.dataframe(df_specific.style.format({'Mean True a': '{:.3f}', 'Mean Est a': '{:.3f}', 'Mean Bias': '{:.3f}', 'RMSE': '{:.3f}'}))

    # DataFrame Display
    st.subheader("Parameter Table (All Items)")
    df_res = pd.DataFrame({
        "Item": [f"Item {i+1}" for i in range(J)],
        "True d": true_d,
        "Est d": est_d
    })
    # Add all discrimination columns
    for dim in range(mhrm.D):
        if structure == "Bifactor":
            col_label = "G" if dim == 0 else f"S_{dim}"
        else:
            col_label = f"Dim {dim+1}"
            
        df_res[f"True a ({col_label})"] = true_a[:, dim]
        df_res[f"Est a ({col_label})"] = est_a[:, dim]
        
    st.dataframe(df_res.style.background_gradient(cmap="Blues"))

    # Heatmap for discrimination matrix
    st.subheader("Discrimination Matrix Heatmap")
    import matplotlib.cm as cm
    # Dynamic height based on number of items
    fig_height = max(4, J * 0.25)
    fig_hm, ax_hm = plt.subplots(figsize=(6, fig_height))
    cax = ax_hm.imshow(est_a, aspect='auto', cmap='viridis')
    ax_hm.set_xlabel('Dimensions')
    ax_hm.set_ylabel('Items')
    ax_hm.set_title('Estimated Discriminations (a)')
    
    # Set ticks for items
    if J <= 50:
        ax_hm.set_yticks(np.arange(J))
        ax_hm.set_yticklabels([f"Item {i+1}" for i in range(J)], fontsize=8)
        
    # Set ticks for dimensions
    ax_hm.set_xticks(np.arange(mhrm.D))
    if structure == "Bifactor":
        dim_labels = ["G"] + [f"S_{d}" for d in range(1, mhrm.D)]
    else:
        dim_labels = [f"D{d+1}" for d in range(mhrm.D)]
    ax_hm.set_xticklabels(dim_labels)
    
    fig_hm.colorbar(cax, ax=ax_hm, shrink=0.7)
    st.pyplot(fig_hm)

else:
    st.info("Adjust settings in the sidebar and click Run to start.")

# --- Explanation Section ---
with st.expander("ℹ️ How does this Algorithm work?"):
    st.markdown(r"""
    ### The Metropolis-Hastings Robbins-Monro (MHRM) Algorithm

    The MHRM algorithm (Cai, 2010) combines Markov Chain Monte Carlo (MCMC) sampling with Stochastic Approximation to estimate item parameters in high-dimensional IRT models.

    #### 1. The Model (Multidimensional 2PL)
    The probability of examinee $i$ answering item $j$ correctly is given by the logistic function:
    $$
    P(Y_{ij}=1 | \boldsymbol{\theta}_i, \mathbf{a}_j, d_j) = \frac{1}{1 + \exp[-(\mathbf{a}_j^T \boldsymbol{\theta}_i + d_j)]}
    $$
    *   $\boldsymbol{\theta}_i$: Latent trait vector for person $i$ (Dimension $D \times 1$).
    *   $\mathbf{a}_j$: Discrimination vector for item $j$ (Dimension $D \times 1$).
    *   $d_j$: Intercept parameter for item $j$ (Scalar).

    #### 2. The Objective
    We aim to maximize the **Marginal Log-Likelihood** of the observed data $\mathbf{Y}$, integrating out the unknown $\boldsymbol{\theta}$:
    $$
    \ell(\xi) = \sum_{i=1}^N \log \int P(\mathbf{Y}_i | \boldsymbol{\theta}, \xi) f(\boldsymbol{\theta}) d\boldsymbol{\theta}
    $$
    where $\xi$ represents all item parameters $(\mathbf{a}, \mathbf{d})$.
    *   **Challenge:** This integral is intractable for high dimensions ($D > 3$) using traditional quadrature.

    #### 3. The MHRM Solution
    Instead of calculating the integral directly, we use **Fisher's Identity**, which states that the gradient of the marginal log-likelihood is the expected value of the complete-data gradient:
    $$
    \nabla_{\xi} \ell(\xi) = E_{\boldsymbol{\theta} | \mathbf{Y}, \xi} \left[ \nabla_{\xi} \log P(\mathbf{Y}, \boldsymbol{\theta} | \xi) \right]
    $$
    
    #### 4. Algorithm Steps (Iterative)

    **Step 1: Metropolis-Hastings (Imputation)**
    We cannot calculate the expectation exactly, so we approximate it by sampling.
    *   Draw samples of $\boldsymbol{\theta}$ from the posterior distribution $P(\boldsymbol{\theta} | \mathbf{Y}, \xi^{(k)})$ using a Random Walk Metropolis sampler.
    *   Propose $\boldsymbol{\theta}^* \sim N(\boldsymbol{\theta}^{(t)}, \sigma^2)$.
    *   Accept/Reject based on the ratio of posterior probabilities.

    **Step 2: Robbins-Monro (Approximation & Update)**
    Update parameters using a stochastic gradient ascent rule:
    $$
    \xi^{(k+1)} = \xi^{(k)} + \gamma_k (\mathbf{H}^{(k)})^{-1} \mathbf{g}^{(k)}
    $$
    *   $\mathbf{g}^{(k)}$: Gradient of the complete data log-likelihood using the imputed $\boldsymbol{\theta}$.
    *   $\mathbf{H}^{(k)}$: Approximation of the Hessian (Information Matrix).
    *   $\gamma_k$: Gain sequence (Step size), satisfying $\sum \gamma_k = \infty$ and $\sum \gamma_k^2 < \infty$ (e.g., $\gamma_k = 1/k$).

    This allows the estimates to converge to the maximum likelihood solution despite the noise introduced by sampling $\boldsymbol{\theta}$.
    """)
