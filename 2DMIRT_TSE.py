import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import multivariate_normal, norm, beta
from scipy.optimize import brentq
import matplotlib.patches as patches

# ==========================================
# 1. DATA GENERATION & MODEL LOGIC
# ==========================================

def mirt_3pl_prob(theta, a, d, g, D=1.702):
    """
    Calculate P(theta) for 2D 3PL MIRT.
    theta: array-like of shape (N, 2) or (2,)
    a: array-like of shape (2,)
    d: scalar
    g: scalar
    """
    theta = np.atleast_2d(theta)
    a = np.array(a)
    z = D * (np.dot(theta, a) + d)
    prob = g + (1 - g) / (1 + np.exp(-z))
    return prob.flatten()

def uirt_3pl_prob(theta, a, b, g, D=1.702):
    """Standard 1D 3PL IRT"""
    z = D * a * (theta - b)
    return g + (1 - g) / (1 + np.exp(-z))

def generate_form_parameters(form_type="base", n_items=50):
    """
    Generates item parameters based on MC-I simulation study conditions (Table 2).
    """
    np.random.seed(42)
    
    # g is common: Beta(5, 17)
    g = np.random.beta(5, 17, n_items)
    
    if form_type == "base":
        # Base Form (Y)
        # a1, a2 ~ U(0.57, 1.14)
        # d ~ U(-1.5, 1.5)
        a1 = np.random.uniform(0.57, 1.14, n_items)
        a2 = np.random.uniform(0.57, 1.14, n_items)
        d = np.random.uniform(-1.5, 1.5, n_items)
        
    elif form_type == "same_rc":
        # Same Reference Composite (New Form X - Condition 2)
        # d ~ U(-0.95, 1.5) (More difficult)
        d = np.random.uniform(-0.95, 1.5, n_items)
        
        n_half = n_items // 2
        # Cluster 1 (1-25): a1 ~ U(0.57, 1.14), a2 ~ U(0.98, 1.95)
        # Note: Table 2 values.
        a1_c1 = np.random.uniform(0.57, 1.14, n_half)
        a2_c1 = np.random.uniform(0.98, 1.95, n_half)
        
        # Cluster 2 (26-50): a1 ~ U(0.98, 1.95), a2 ~ U(0.57, 1.14)
        a1_c2 = np.random.uniform(0.98, 1.95, n_items - n_half)
        a2_c2 = np.random.uniform(0.57, 1.14, n_items - n_half)
        
        a1 = np.concatenate([a1_c1, a1_c2])
        a2 = np.concatenate([a2_c1, a2_c2])
        
    elif form_type == "diff_rc":
        # Different Reference Composite (New Form X - Condition 3)
        # d ~ U(-0.95, 1.5)
        d = np.random.uniform(-0.95, 1.5, n_items)
        
        n_half = n_items // 2
        # Cluster 1 (1-25): a1 ~ U(0.26, 0.57), a2 ~ U(0.78, 1.57)
        a1_c1 = np.random.uniform(0.26, 0.57, n_half)
        a2_c1 = np.random.uniform(0.78, 1.57, n_half)
        
        # Cluster 2 (26-50): a1 ~ U(0.78, 1.57), a2 ~ U(0.66, 1.31)
        a1_c2 = np.random.uniform(0.78, 1.57, n_items - n_half)
        a2_c2 = np.random.uniform(0.66, 1.31, n_items - n_half)
        
        a1 = np.concatenate([a1_c1, a1_c2])
        a2 = np.concatenate([a2_c1, a2_c2])
    
    return pd.DataFrame({'a1': a1, 'a2': a2, 'd': d, 'g': g})

def get_latent_params(group="reference"):
    """
    Returns Mean and Covariance for latent distributions.
    """
    if group == "reference":
        mu = np.array([0, 0])
        cov = np.array([[1, 0.3], [0.3, 1]])
    elif group == "target":
        # Using the (0.5, 0.5) mean shift and covariance shift condition from image
        mu = np.array([0.5, 0.5])
        cov = np.array([[1, 0.5], [0.5, 1.2]])
    return mu, cov

# ... (rest of projection and equating functions remain similar)


def project_to_uirt(items):
    """
    Projects MIRT parameters to UIRT Linear Composite parameters.
    Reference: Zhang & Wang (1998), Dissertation equations 2.2.6-2.2.10.
    Assumes standard normal population covariance matrix Sigma = I.
    """
    A = items[['a1', 'a2']].values
    
    # 1. Find Reference Composite (Eigenvector of A^T A)
    ata = np.dot(A.T, A)
    eigvals, eigvecs = np.linalg.eigh(ata)
    # Sort to get largest eigenvalue
    idx = eigvals.argsort()[::-1]
    alpha = eigvecs[:, idx][:, 0] # Principal direction vector
    
    # Ensure positive orientation
    if np.sum(alpha) < 0: alpha = -alpha
    
    # 2. Project parameters
    # Sigma is Identity for standard normal population assumption
    # a_star = (1 + var_star)^(-0.5) * (a . alpha)
    # var_star represents "noise" perpendicular to composite. 
    # Simplified projection often used: a_uni ~ projection length
    
    a_star = []
    b_star = []
    
    for i in range(len(items)):
        a_vec = A[i]
        d_val = items.iloc[i]['d']
        
        # Projection of a onto alpha
        proj_len = np.dot(a_vec, alpha)
        
        # Variance of specific factor (perpendicular)
        # sigma_i_sq = a_vec^T Sigma a_vec - (a_vec^T Sigma alpha)^2
        # With Sigma=I: |a|^2 - (a.alpha)^2
        sigma_i_sq = np.dot(a_vec, a_vec) - proj_len**2
        
        # Formulas
        scaling = (1 + sigma_i_sq)**(-0.5)
        a_uni = scaling * proj_len
        d_uni = scaling * d_val
        
        # Convert intercept d to difficulty b: b = -d / a
        b_uni = -d_uni / a_uni
        
        a_star.append(a_uni)
        b_star.append(b_uni)
        
    return pd.DataFrame({
        'a': a_star,
        'b': b_star,
        'g': items['g']
    })

# ==========================================
# 2. EQUATING METHODS
# ==========================================

# --- Helper: Lord-Wingersky 1D ---
def lord_wingersky_1d(probs):
    """
    Recursive algorithm for 1D observed score distribution.
    probs: (n_items, n_quad)
    Returns: (n_items + 1, n_quad) distribution of scores 0..N
    """
    n_items, n_quad = probs.shape
    dist = np.zeros((n_items + 1, n_quad))
    dist[0, :] = 1.0
    
    for i in range(n_items):
        p = probs[i, :]
        q = 1 - p
        new_dist = np.zeros_like(dist)
        
        # Score x: came from x (got wrong) or x-1 (got right)
        # f_i(x) = f_{i-1}(x)*q + f_{i-1}(x-1)*p
        
        # Case x=0 to i
        new_dist[0:i+2, :] = dist[0:i+2, :] * q + np.vstack([np.zeros((1, n_quad)), dist[0:i+1, :] * p])
        dist = new_dist
        
    return dist

# --- Helper: Lord-Wingersky 2D (Generalized) ---
# We simply calculate probs for every theta pair, then run 1D LW on that column.
# The "2D" aspect is in how probabilities are generated.

def get_percentile_ranks(dist):
    """
    Calculate percentile ranks for a discrete distribution.
    PR(x) = P(X < x) + 0.5 * P(X = x)
    """
    cdf = np.cumsum(dist)
    # P(X < x) is cdf shifted right
    p_less = np.roll(cdf, 1)
    p_less[0] = 0
    pr = p_less + 0.5 * dist
    return pr

def run_mirt_ose(base_items, new_items, mu=None, cov=None):
    """
    MIRT Observed Score Equating.
    1. Grid of theta points.
    2. Calculate score dist P(X|theta) for Base and New.
    3. Marginalize over bivariate normal.
    4. Equipercentile equate with interpolation.
    """
    if mu is None: mu = [0, 0]
    if cov is None: cov = [[1, 0], [0, 1]]

    # Quadrature Grid
    nodes = 41
    x = np.linspace(-4, 4, nodes)
    X, Y = np.meshgrid(x, x)
    flat_X, flat_Y = X.flatten(), Y.flatten()
    points = np.vstack([flat_X, flat_Y]).T
    
    # Weights (Bivariate Normal)
    weights = multivariate_normal.pdf(points, mean=mu, cov=cov)
    weights /= np.sum(weights)
    
    # Calculate Probabilities
    # Shape: (n_items, n_points)
    def get_probs(items):
        p_mat = []
        for _, item in items.iterrows():
            a = [item['a1'], item['a2']]
            p = mirt_3pl_prob(points, a, item['d'], item['g'])
            p_mat.append(p)
        return np.array(p_mat)
    
    probs_base = get_probs(base_items)
    probs_new = get_probs(new_items)
    
    # Conditional Score Distributions P(X | theta)
    # Run LW for each theta point (column)
    dist_base_cond = lord_wingersky_1d(probs_base) # (N+1, n_points)
    dist_new_cond = lord_wingersky_1d(probs_new)
    
    # Marginalize: Sum (P(X|theta) * weight(theta))
    dist_base_marg = np.dot(dist_base_cond, weights)
    dist_new_marg = np.dot(dist_new_cond, weights)
    
    # Equipercentile Equating using interpolation
    pr_base = get_percentile_ranks(dist_base_marg)
    pr_new = get_percentile_ranks(dist_new_marg)
    
    scores = np.arange(len(dist_new_marg))
    
    # Map New Score -> PR -> Base Score (via interpolation)
    equated_scores = np.interp(pr_new, pr_base, scores)
        
    return scores, equated_scores

def run_uirt_ose(base_uirt, new_uirt):
    """Standard 1D OSE"""
    nodes = 61
    theta = np.linspace(-4, 4, nodes)
    weights = norm.pdf(theta)
    weights /= np.sum(weights)
    
    def get_probs(items):
        p_mat = []
        for _, item in items.iterrows():
            p = uirt_3pl_prob(theta, item['a'], item['b'], item['g'])
            p_mat.append(p)
        return np.array(p_mat)
    
    dist_base = np.dot(lord_wingersky_1d(get_probs(base_uirt)), weights)
    dist_new = np.dot(lord_wingersky_1d(get_probs(new_uirt)), weights)
    
    pr_base = get_percentile_ranks(dist_base)
    pr_new = get_percentile_ranks(dist_new)
    
    scores = np.arange(len(dist_new))
    equated_scores = np.interp(pr_new, pr_base, scores)
        
    return scores, equated_scores

def run_uirt_tse(base_uirt, new_uirt):
    """Standard UIRT True Score Equating"""
    scores = np.arange(0, len(new_uirt)+1)
    equated = []
    
    # Define TCC functions
    def get_tcc(theta, items):
        t = 0
        for _, i in items.iterrows():
            t += uirt_3pl_prob(theta, i['a'], i['b'], i['g'])
        return t
    
    for s in scores:
        # 1. Find theta equivalent to score s on New Form
        # TCC_new(theta) - s = 0
        try:
            theta_s = brentq(lambda t: get_tcc(t, new_uirt) - s, -6, 6)
            # 2. Project theta to Base Form
            s_y = get_tcc(theta_s, base_uirt)
        except ValueError:
            # Score outside possible range (e.g. sum of guessings)
            s_y = s # Fallback or min/max
            if s < get_tcc(-6, new_uirt): s_y = get_tcc(-6, base_uirt)
            if s > get_tcc(6, new_uirt): s_y = get_tcc(6, base_uirt)
            
        equated.append(s_y)
    return scores, np.array(equated)

def run_mirt_tse_approx(base_items, new_items, use_weights=True, mu=None, cov=None):
    """
    Approximate MIRT True Score Equating (Dissertation Appendix C).
    1. For each score S on New Form:
    2. Find iso-score contour: E(X|t1, t2) = S.
    3. Select equi-distant points on contour.
    4. Map these points to Base Form Expected Score.
    5. Average (weighted by conditional probability P(X=S|theta) * Density).
    """
    if mu is None: mu = [0, 0]
    if cov is None: cov = [[1, 0], [0, 1]]

    scores = np.arange(0, len(new_items)+1)
    equated = []
    
    # Vectorized TCC calculation for speed
    a_new = new_items[['a1', 'a2']].values
    d_new = new_items['d'].values
    g_new = new_items['g'].values
    
    a_base = base_items[['a1', 'a2']].values
    d_base = base_items['d'].values
    g_base = base_items['g'].values

    def get_tcc_new(theta):
        # theta: (N, 2)
        z = 1.702 * (theta @ a_new.T + d_new)
        prob = g_new + (1 - g_new) / (1 + np.exp(-z))
        return prob.sum(axis=1)

    def get_tcc_base(theta):
        # theta: (N, 2)
        z = 1.702 * (theta @ a_base.T + d_base)
        prob = g_base + (1 - g_base) / (1 + np.exp(-z))
        return prob.sum(axis=1)

    # Grid for contour search
    theta1_grid = np.linspace(-4, 4, 100) 
    
    for s in scores:
        contour_points = []
        
        # Find contour points
        for t1 in theta1_grid:
            func = lambda t2: get_tcc_new(np.array([[t1, t2]]))[0] - s
            try:
                low, high = -5, 5
                fa, fb = func(low), func(high)
                if np.sign(fa) != np.sign(fb):
                    t2 = brentq(func, low, high)
                    contour_points.append([t1, t2])
            except ValueError:
                pass
        
        if len(contour_points) > 1:
            contour_points = np.array(contour_points)
            
            # Resample to be equidistant along the curve
            # Calculate cumulative distance
            dists = np.sqrt(np.sum(np.diff(contour_points, axis=0)**2, axis=1))
            cum_dist = np.insert(np.cumsum(dists), 0, 0)
            total_len = cum_dist[-1]
            
            # Select 20 equidistant points (or more for accuracy)
            n_samples = 20
            target_dists = np.linspace(0, total_len, n_samples)
            
            # Interpolate
            equi_points = np.zeros((n_samples, 2))
            equi_points[:, 0] = np.interp(target_dists, cum_dist, contour_points[:, 0])
            equi_points[:, 1] = np.interp(target_dists, cum_dist, contour_points[:, 1])
            
            # 1. Get score on Base Form
            s_base = get_tcc_base(equi_points)
            
            # 2. Get Weights
            if use_weights:
                # Weight = P(X=s | theta) * Density(theta)
                # This represents the posterior density along the contour
                cond_prob = get_conditional_prob_2d(equi_points, new_items, s)
                density = multivariate_normal.pdf(equi_points, mean=mu, cov=cov)
                weights = cond_prob * density
            else:
                weights = np.ones(len(equi_points))
            
            if np.sum(weights) > 0:
                avg_score = np.average(s_base, weights=weights)
            else:
                avg_score = np.mean(s_base)
                
            equated.append(avg_score)
        else:
            # Fallback
            if s < len(new_items) / 2:
                 equated.append(get_tcc_base(np.array([[-4, -4]]))[0])
            else:
                 equated.append(get_tcc_base(np.array([[4, 4]]))[0])
            
    return scores, np.array(equated)

def get_conditional_prob_2d(theta, items, score_k):
    """
    Calculates P(X=k | theta) for a set of theta points using Vectorized LW recursion.
    theta: (N, 2)
    items: DataFrame
    """
    a = items[['a1', 'a2']].values
    d = items['d'].values
    g = items['g'].values
    
    # Calculate P matrix (N_theta, n_items)
    z = 1.702 * (theta @ a.T + d)
    probs = g + (1 - g) / (1 + np.exp(-z))
    
    # Vectorized Lord-Wingersky
    n_points = probs.shape[0]
    n_items = probs.shape[1]
    
    # dist[p, s] is probability of score s for point p
    dist = np.zeros((n_points, n_items + 1))
    dist[:, 0] = 1.0
    
    for i in range(n_items):
        p = probs[:, i:i+1] # (N, 1)
        new_dist = np.zeros_like(dist)
        new_dist[:, 1:] += dist[:, :-1] * p
        new_dist[:, :-1] += dist[:, :-1] * (1 - p)
        dist = new_dist
        
    return dist[:, score_k]

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================

def plot_vectors(df, title):
    """
    Vector plot for MIRT items.
    Vectors start at the point of difficulty (closest to origin on L(theta)=0)
    and point in the direction of discrimination.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    a1 = df['a1'].values
    a2 = df['a2'].values
    d = df['d'].values
    
    # Calculate arrow properties
    arrow_len = np.sqrt(a1**2 + a2**2)
    # arrow_tail is the distance from origin to the start of the vector (difficulty location)
    arrow_tail = -d / arrow_len
    
    # Calculate angle
    arrow_angle = np.arctan2(a2, a1)
    
    # Calculate start points (x0, y0)
    x0 = arrow_tail * np.cos(arrow_angle)
    y0 = arrow_tail * np.sin(arrow_angle)
    
    # Plot setup
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title)
    
    # Grid lines
    ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.3)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Reference lines
    # 1. 45 degree line (b=1)
    ax.plot([-4, 4], [-4, 4], 'k--', alpha=0.5, label='45° Line')
    
    # 2. Reference Composite (First Eigenvector of A'A)
    A = np.column_stack([a1, a2])
    ATA = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(ATA)
    # Sort by eigenvalue descending
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    v1 = eigvecs[:, 0]
    
    # Slope of reference composite
    slope = v1[1] / v1[0]
    
    # Plot Reference Composite
    x_vals = np.array([-4, 4])
    y_vals = slope * x_vals
    ax.plot(x_vals, y_vals, 'r-', linewidth=1.5, label='Ref Composite')
    
    # Plot Vectors
    # X, Y: start points (x0, y0)
    # U, V: components (a1, a2)
    ax.quiver(x0, y0, a1, a2, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.6, width=0.005)
    
    # Add legend
    ax.legend(loc='upper left', fontsize='small')
    
    return fig

def plot_latent_distributions(mu_ref, cov_ref, mu_tar, cov_tar, label1='Reference', label2='Target', title="Latent Ability Distributions"):
    """
    Plots 95% confidence ellipses for Reference and Target groups.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def plot_ellipse(mu, cov, ax, color, label):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(5.991 * vals) # 95% CI
        ell = patches.Ellipse(xy=mu, width=width, height=height, angle=theta, 
                              edgecolor=color, facecolor='none', lw=2, label=label)
        ax.add_patch(ell)
        ax.plot(mu[0], mu[1], 'x', color=color)

    plot_ellipse(mu_ref, cov_ref, ax, 'red', label1)
    plot_ellipse(mu_tar, cov_tar, ax, 'blue', label2)
    
    # Add marginal distributions (schematic)
    x = np.linspace(-4, 4, 100)
    # Ref Marginals
    ax.plot(x, norm.pdf(x, mu_ref[0], np.sqrt(cov_ref[0,0])) - 4, 'r-', alpha=0.5)
    ax.plot(norm.pdf(x, mu_ref[1], np.sqrt(cov_ref[1,1])) - 4, x, 'r-', alpha=0.5)
    
    # Tar Marginals
    ax.plot(x, norm.pdf(x, mu_tar[0], np.sqrt(cov_tar[0,0])) - 4, 'b-', alpha=0.5)
    ax.plot(norm.pdf(x, mu_tar[1], np.sqrt(cov_tar[1,1])) - 4, x, 'b-', alpha=0.5)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--')
    return fig

def plot_tcc_surface(items, title="Test Characteristic Surface", items2=None, label2="New Form"):
    """
    3D Surface plot of TCC.
    """
    x = np.linspace(-3, 3, 40)
    y = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(x, y)
    
    # Vectorized TCC
    def get_z(df):
        a = df[['a1', 'a2']].values
        d = df['d'].values
        g = df['g'].values
        
        # Flatten for calculation
        flat_X = X.flatten()
        flat_Y = Y.flatten()
        theta = np.vstack([flat_X, flat_Y]).T
        
        # Calculate Prob Sum
        z = 1.702 * (theta @ a.T + d)
        probs = g + (1 - g) / (1 + np.exp(-z))
        tcc = np.sum(probs, axis=1).reshape(X.shape)
        return tcc

    z1 = get_z(items)
    
    data = [go.Surface(z=z1, x=X, y=Y, colorscale='Blues', opacity=0.8, name="Base Form", showscale=False)]
    
    if items2 is not None:
        z2 = get_z(items2)
        data.append(go.Surface(z=z2, x=X, y=Y, colorscale='Reds', opacity=0.7, name=label2, showscale=False))
    
    fig = go.Figure(data=data)
    
    # Add dummy traces for legend
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                               marker=dict(size=10, color='blue'), name='Base Form'))
    
    if items2 is not None:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                                   marker=dict(size=10, color='red'), name=label2))

    fig.update_layout(title=title, scene=dict(
        xaxis_title='Theta 1',
        yaxis_title='Theta 2',
        zaxis_title='Expected Score'
    ), width=700, height=600, showlegend=True)
    return fig

def plot_contour_optimal_line(items, score_level, title="TCC Contour & Optimal Line"):
    """
    Contour plot with optimal line (Reference Composite) and equidistant points.
    """
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    a = items[['a1', 'a2']].values
    d = items['d'].values
    g = items['g'].values
    
    theta = np.vstack([X.flatten(), Y.flatten()]).T
    z = 1.702 * (theta @ a.T + d)
    probs = g + (1 - g) / (1 + np.exp(-z))
    tcc = np.sum(probs, axis=1).reshape(X.shape)
    
    n_items = len(items)
    levels = np.arange(0, n_items + 1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cp = ax.contourf(X, Y, tcc, levels=levels, cmap='viridis')
    fig.colorbar(cp, label='Expected Score')
    
    # Add faint lines for all integer scores
    ax.contour(X, Y, tcc, levels=levels, colors='black', alpha=0.2, linewidths=0.5)
    
    # Contour line for specific score
    ax.contour(X, Y, tcc, levels=[score_level], colors='white', linewidths=2)
    
    # Optimal Line (Reference Composite)
    ata = np.dot(a.T, a)
    eigvals, eigvecs = np.linalg.eigh(ata)
    alpha = eigvecs[:, eigvals.argsort()[::-1]][:, 0]
    if np.sum(alpha) < 0: alpha = -alpha
    
    # Plot line through origin
    line_x = np.array([-3, 3])
    line_y = line_x * (alpha[1] / alpha[0])
    ax.plot(line_x, line_y, 'k-', lw=2, label='Optimal Line (Ref Composite)')
    
    # Find and plot equidistant points on the contour
    # 1. Find contour points
    def get_tcc_val(t):
        z = 1.702 * (t @ a.T + d)
        p = g + (1 - g) / (1 + np.exp(-z))
        return p.sum()

    contour_points = []
    t1_grid = np.linspace(-3, 3, 50)
    for t1 in t1_grid:
        func = lambda t2: get_tcc_val(np.array([t1, t2])) - score_level
        try:
            t2 = brentq(func, -4, 4)
            contour_points.append([t1, t2])
        except ValueError:
            pass
            
    if len(contour_points) > 1:
        contour_points = np.array(contour_points)
        # Resample to be equidistant
        dists = np.sqrt(np.sum(np.diff(contour_points, axis=0)**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dist[-1]
        
        n_samples = 10
        target_dists = np.linspace(0, total_len, n_samples)
        
        equi_points = np.zeros((n_samples, 2))
        equi_points[:, 0] = np.interp(target_dists, cum_dist, contour_points[:, 0])
        equi_points[:, 1] = np.interp(target_dists, cum_dist, contour_points[:, 1])
        
        ax.plot(equi_points[:, 0], equi_points[:, 1], 'wo', markersize=6, markeredgecolor='k', label='Equidistant Points')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Theta 1")
    ax.set_ylabel("Theta 2")
    ax.set_title(f"{title} (Score = {score_level})")
    ax.legend()
    return fig

def plot_conditional_score_prob(items, title="2D MIRT Conditional Observed Score Probability Visualization in 3D Space"):
    """
    3D Scatter plot of P(X=k | Theta) for all k and Theta grid.
    Z-axis is Score k.
    Color is Probability.
    """
    # Resolution
    n_grid = 20
    x = np.linspace(-3, 3, n_grid)
    y = np.linspace(-3, 3, n_grid)
    X, Y = np.meshgrid(x, y)
    
    a = items[['a1', 'a2']].values
    d = items['d'].values
    g = items['g'].values
    
    theta = np.vstack([X.flatten(), Y.flatten()]).T
    
    # Calculate P matrix (N_grid, n_items)
    z = 1.702 * (theta @ a.T + d)
    probs = g + (1 - g) / (1 + np.exp(-z))
    
    # Vectorized Lord-Wingersky
    n_points = probs.shape[0]
    n_items = probs.shape[1]
    
    # dist[p, s] is probability of score s for point p
    dist = np.zeros((n_points, n_items + 1))
    dist[:, 0] = 1.0
    
    for i in range(n_items):
        p = probs[:, i:i+1] # (N, 1)
        new_dist = np.zeros_like(dist)
        new_dist[:, 1:] += dist[:, :-1] * p
        new_dist[:, :-1] += dist[:, :-1] * (1 - p)
        dist = new_dist
        
    # Prepare data for plotting
    # We want (theta1, theta2, score, prob)
    
    # Repeat theta for each score
    theta1_rep = np.repeat(theta[:, 0], n_items + 1)
    theta2_rep = np.repeat(theta[:, 1], n_items + 1)
    
    # Tile scores for each theta
    scores_rep = np.tile(np.arange(n_items + 1), n_points)
    
    # Flatten probs
    probs_flat = dist.flatten()
    
    # Filter low probabilities to reduce clutter
    mask = probs_flat > 0.001
    
    fig = go.Figure(data=[go.Scatter3d(
        x=theta1_rep[mask],
        y=theta2_rep[mask],
        z=scores_rep[mask],
        mode='markers',
        marker=dict(
            size=2,
            color=probs_flat[mask],
            colorscale='Reds',
            opacity=0.5,
            colorbar=dict(title="Probability")
        )
    )])
    
    fig.update_layout(title=title, scene=dict(
        xaxis_title='Theta 1',
        yaxis_title='Theta 2',
        zaxis_title='Number Correct Score'
    ), width=800, height=700)
    return fig

# ==========================================
# 4. STREAMLIT APP UI
# ==========================================

st.set_page_config(layout="wide", page_title="MC-I Simulation Study")
st.title("MC-I Simulation Study: Multidimensional Equating")
st.markdown("""
This application replicates the **MC-I** simulation condition.
It investigates the impact of multidimensional structure on equating under three conditions:
1.  **Base Form**: Uniform structure (Reference Composite ~45°).
2.  **Same RC**: Clusters at 30°/60° but same Reference Composite.
3.  **Diff RC**: Different Reference Composite (Angle ~51°).
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
n_items = st.sidebar.slider("Number of Items", 10, 50, 50, step=10)
seed = st.sidebar.number_input("Random Seed", 42)
use_weights_mirt = st.sidebar.checkbox("Use Weights for MIRT TSE", value=False, help="If checked, MIRT TSE uses density-weighted averaging along the iso-score contour. If unchecked, it uses a simple average.")
use_target_group = st.sidebar.checkbox("Use Target Group Distribution", value=False, help="If checked, Target Group has mean (0.5, 0.5). If unchecked, Target Group has same distribution as Reference.")

# Show Generation Info
with st.expander("📋 View Generation Parameters & Distributions", expanded=True):
    st.markdown("### Item Parameter Generation (MC-I)")
    st.markdown("""
    | Form | Items | $a_1$ | $a_2$ | $d$ | $g$ |
    |---|---|---|---|---|---|
    | **Base** | 1-50 | $U(0.57, 1.14)$ | $U(0.57, 1.14)$ | $U(-1.5, 1.5)$ | $Beta(5, 17)$ |
    | **Same RC** | 1-25 | $U(0.57, 1.14)$ | $U(0.98, 1.95)$ | $U(-0.95, 1.5)$ | $Beta(5, 17)$ |
    | | 26-50 | $U(0.98, 1.95)$ | $U(0.57, 1.14)$ | | |
    | **Diff RC** | 1-25 | $U(0.26, 0.57)$ | $U(0.78, 1.57)$ | $U(-0.95, 1.5)$ | $Beta(5, 17)$ |
    | | 26-50 | $U(0.78, 1.57)$ | $U(0.66, 1.31)$ | | |
    """)
    
    st.markdown("### Latent Ability Distributions")
    
    md_text = r"""
    **Reference Group**: $\mu = (0,0)$, $\Sigma = \begin{pmatrix} 1 & 0.3 \\ 0.3 & 1 \end{pmatrix}$
    
    **Target Group**: $\mu = (0.5, 0.5)$, $\Sigma = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1.2 \end{pmatrix}$
    """
    
    st.markdown(md_text, unsafe_allow_html=True)

if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Generating data and performing equating..."):
        progress_bar = st.progress(0, text="Initializing...")
        
        # Define Population Parameters
        mu_ref = np.array([0, 0])
        cov_ref = np.array([[1, 0.3], [0.3, 1]])
        
        mu_tar = np.array([0.5, 0.5])
        cov_tar = np.array([[1, 0.5], [0.5, 1.2]])

        # Determine Run Parameters based on Sidebar
        if use_target_group:
            mu_run = mu_tar
            cov_run = cov_tar
            label_run = "Target Group"
        else:
            mu_run = mu_ref
            cov_run = cov_ref
            label_run = "Equivalent Group"

        # 1. Generate Parameters
        df_base = generate_form_parameters("base", n_items)
        df_same_rc = generate_form_parameters("same_rc", n_items)
        df_diff_rc = generate_form_parameters("diff_rc", n_items)
        
        progress_bar.progress(10, text="Projecting to UIRT...")
        # 2. Project to UIRT
        uirt_base = project_to_uirt(df_base)
        uirt_same_rc = project_to_uirt(df_same_rc)
        uirt_diff_rc = project_to_uirt(df_diff_rc)
        
        # 3. Visualizations
        st.header("1. Visualizations")
        
        # A. Latent Distributions
        st.subheader("A. Latent Ability Distributions")
        # mu_ref, cov_ref = get_latent_params("reference") # Use defined vars instead
        # mu_tar, cov_tar = get_latent_params("target")
        
        fig_latent = plot_latent_distributions(mu_ref, cov_ref, mu_run, cov_run, label2=label_run)
        st.pyplot(fig_latent)
        
        # B. Vector Space
        st.subheader("B. Item Vector Space")
        cols = st.columns(3)
        
        with cols[0]:
            st.pyplot(plot_vectors(df_base, "Base Form"))
        with cols[1]:
            st.pyplot(plot_vectors(df_same_rc, "Same RC"))
        with cols[2]:
            st.pyplot(plot_vectors(df_diff_rc, "Diff RC"))
        # Equiv Group plot removed from here as requested
        
        # C. TCC Surface & Contour
        st.subheader("C. Test Characteristic Surface & Contour")
        tab1, tab2, tab3 = st.tabs(["3D Surface", "Contour & Optimal Line", "Conditional Prob P(X=k)"])
        
        with tab1:
            col_surf1, col_surf2 = st.columns(2)
            with col_surf1:
                fig_surf1 = plot_tcc_surface(df_base, "Base vs Same RC", items2=df_same_rc, label2="Same RC")
                st.plotly_chart(fig_surf1)
            with col_surf2:
                fig_surf2 = plot_tcc_surface(df_base, "Base vs Diff RC", items2=df_diff_rc, label2="Diff RC")
                st.plotly_chart(fig_surf2)
            
        with tab2:
            target_score = n_items // 2
            fig_cont = plot_contour_optimal_line(df_base, target_score, "Base Form Contour")
            st.pyplot(fig_cont)
            
        with tab3:
            st.markdown("Visualization of P(X=k | Theta) for all scores k.")
            fig_cond = plot_conditional_score_prob(df_base, "Base Form")
            st.plotly_chart(fig_cond)

        # 4. Equating
        st.header("2. Equating Results")
        st.markdown("Comparing **Same RC** and **Diff RC** forms to **Base Form**.")
        
        progress_bar.progress(30, text="Running Equating (Same RC)...")
        # Same RC (Target Group)
        x_mirt_tse_1, y_mirt_tse_1 = run_mirt_tse_approx(df_base, df_same_rc, use_weights=use_weights_mirt, mu=mu_run, cov=cov_run)
        x_mirt_ose_1, y_mirt_ose_1 = run_mirt_ose(df_base, df_same_rc, mu=mu_run, cov=cov_run)
        x_uirt_tse_1, y_uirt_tse_1 = run_uirt_tse(uirt_base, uirt_same_rc)
        x_uirt_ose_1, y_uirt_ose_1 = run_uirt_ose(uirt_base, uirt_same_rc)
        
        progress_bar.progress(60, text="Running Equating (Diff RC)...")
        # Diff RC (Target Group)
        x_mirt_tse_2, y_mirt_tse_2 = run_mirt_tse_approx(df_base, df_diff_rc, use_weights=use_weights_mirt, mu=mu_run, cov=cov_run)
        x_mirt_ose_2, y_mirt_ose_2 = run_mirt_ose(df_base, df_diff_rc, mu=mu_run, cov=cov_run)
        x_uirt_tse_2, y_uirt_tse_2 = run_uirt_tse(uirt_base, uirt_diff_rc)
        x_uirt_ose_2, y_uirt_ose_2 = run_uirt_ose(uirt_base, uirt_diff_rc)
        
        progress_bar.progress(100, text="Done!")
        
        # Calculate Sum of Guessing
        sum_g_1 = df_same_rc['g'].sum()
        sum_g_2 = df_diff_rc['g'].sum()
        
        # Plot Equating
        n_plots = 2
        fig_eq, axes_eq = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        if n_plots == 1: axes_eq = [axes_eq] # Handle single plot case if ever needed
        
        # Plot 1: Same RC
        mask1 = x_mirt_tse_1 >= sum_g_1
        axes_eq[0].plot([0, n_items], [0, n_items], 'k--', alpha=0.3, label="Identity")
        axes_eq[0].plot(x_mirt_tse_1[mask1], y_mirt_tse_1[mask1], label=f"MIRT TSE (w={use_weights_mirt})", lw=2, color='red')
        axes_eq[0].plot(x_mirt_ose_1[mask1], y_mirt_ose_1[mask1], label="MIRT OSE", ls='--', color='orange')
        axes_eq[0].plot(x_uirt_tse_1[mask1], y_uirt_tse_1[mask1], label="UIRT TSE", ls='-.', color='blue')
        axes_eq[0].plot(x_uirt_ose_1[mask1], y_uirt_ose_1[mask1], label="UIRT OSE", ls=':', color='black')
        axes_eq[0].axvline(x=sum_g_1, color='gray', linestyle=':', label=f"Sum(g)={sum_g_1:.1f}")
        axes_eq[0].set_title("Condition 2: Same RC")
        axes_eq[0].set_xlabel("Score (New Form)")
        axes_eq[0].set_ylabel("Equated (Base)")
        axes_eq[0].legend()
        axes_eq[0].grid(True)
        
        # Plot 2: Diff RC
        mask2 = x_mirt_tse_2 >= sum_g_2
        axes_eq[1].plot([0, n_items], [0, n_items], 'k--', alpha=0.3, label="Identity")
        axes_eq[1].plot(x_mirt_tse_2[mask2], y_mirt_tse_2[mask2], label=f"MIRT TSE (w={use_weights_mirt})", lw=2, color='red')
        axes_eq[1].plot(x_mirt_ose_2[mask2], y_mirt_ose_2[mask2], label="MIRT OSE", ls='--', color='orange')
        axes_eq[1].plot(x_uirt_tse_2[mask2], y_uirt_tse_2[mask2], label="UIRT TSE", ls='-.', color='blue')
        axes_eq[1].plot(x_uirt_ose_2[mask2], y_uirt_ose_2[mask2], label="UIRT OSE", ls=':', color='black')
        axes_eq[1].axvline(x=sum_g_2, color='gray', linestyle=':', label=f"Sum(g)={sum_g_2:.1f}")
        axes_eq[1].set_title("Condition 3: Diff RC")
        axes_eq[1].set_xlabel("Score (New Form)")
        axes_eq[1].set_ylabel("Equated (Base)")
        axes_eq[1].legend()
        axes_eq[1].grid(True)

        # Plot 3: Equiv Group removed
        
        st.pyplot(fig_eq)
        
        # Equating Table
        st.subheader("Equating Conversion Table")
        st.markdown("Scores below the sum of guessing parameters are excluded.")
        
        # Create DataFrame
        scores_idx = x_mirt_tse_1 
        
        eq_data = {
            "Raw Score": scores_idx,
            "Same RC (MIRT TSE)": y_mirt_tse_1,
            "Same RC (MIRT OSE)": y_mirt_ose_1,
            "Same RC (UIRT TSE)": y_uirt_tse_1,
            "Same RC (UIRT OSE)": y_uirt_ose_1,
            "Diff RC (MIRT TSE)": y_mirt_tse_2,
            "Diff RC (MIRT OSE)": y_mirt_ose_2,
            "Diff RC (UIRT TSE)": y_uirt_tse_2,
            "Diff RC (UIRT OSE)": y_uirt_ose_2,
        }

        df_eq_results = pd.DataFrame(eq_data)
        
        # Filter Table
        threshold_table = max(sum_g_1, sum_g_2)
            
        df_eq_results_filt = df_eq_results[df_eq_results["Raw Score"] >= threshold_table]
        
        st.dataframe(df_eq_results_filt.style.format("{:.4f}", subset=df_eq_results.columns[1:]))
        
        # Difference Plots
        st.subheader("Difference Plots")
        
        if use_target_group:
            st.markdown("For Same/Diff RC: Difference = Method - MIRT TSE")
            ref_label = "MIRT TSE"
        else:
            st.markdown("For Same/Diff RC: Difference = Method - Identity (Raw Score)")
            ref_label = "Identity"

        fig_diff, axes_diff = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        if n_plots == 1: axes_diff = [axes_diff]

        # Plot 1: Same RC Differences
        axes_diff[0].axhline(0, color='k', linestyle='-', alpha=0.3)
        
        if use_target_group:
            # Reference is MIRT TSE
            diff_mirt_ose_1 = y_mirt_ose_1 - y_mirt_tse_1
            diff_uirt_tse_1 = y_uirt_tse_1 - y_mirt_tse_1
            diff_uirt_ose_1 = y_uirt_ose_1 - y_mirt_tse_1
        else:
            # Reference is Identity (x_mirt_tse_1 is the raw score)
            diff_mirt_ose_1 = y_mirt_ose_1 - x_mirt_tse_1
            diff_uirt_tse_1 = y_uirt_tse_1 - x_mirt_tse_1
            diff_uirt_ose_1 = y_uirt_ose_1 - x_mirt_tse_1
            # Also plot MIRT TSE difference from Identity
            diff_mirt_tse_1 = y_mirt_tse_1 - x_mirt_tse_1
            axes_diff[0].plot(x_mirt_tse_1[mask1], diff_mirt_tse_1[mask1], label="MIRT TSE - Identity", lw=2, color='red')

        axes_diff[0].plot(x_mirt_tse_1[mask1], diff_mirt_ose_1[mask1], label=f"MIRT OSE - {ref_label}", ls='--', color='orange')
        axes_diff[0].plot(x_mirt_tse_1[mask1], diff_uirt_tse_1[mask1], label=f"UIRT TSE - {ref_label}", ls='-.', color='blue')
        axes_diff[0].plot(x_mirt_tse_1[mask1], diff_uirt_ose_1[mask1], label=f"UIRT OSE - {ref_label}", ls=':', color='cyan')
        axes_diff[0].set_title("Difference: Same RC")
        axes_diff[0].set_xlabel("Score (New Form)")
        axes_diff[0].set_ylabel("Difference")
        axes_diff[0].legend()
        axes_diff[0].grid(True)
        
        # Plot 2: Diff RC Differences
        axes_diff[1].axhline(0, color='k', linestyle='-', alpha=0.3)
        
        if use_target_group:
            diff_mirt_ose_2 = y_mirt_ose_2 - y_mirt_tse_2
            diff_uirt_tse_2 = y_uirt_tse_2 - y_mirt_tse_2
            diff_uirt_ose_2 = y_uirt_ose_2 - y_mirt_tse_2
        else:
            diff_mirt_ose_2 = y_mirt_ose_2 - x_mirt_tse_2
            diff_uirt_tse_2 = y_uirt_tse_2 - x_mirt_tse_2
            diff_uirt_ose_2 = y_uirt_ose_2 - x_mirt_tse_2
            diff_mirt_tse_2 = y_mirt_tse_2 - x_mirt_tse_2
            axes_diff[1].plot(x_mirt_tse_2[mask2], diff_mirt_tse_2[mask2], label="MIRT TSE - Identity", lw=2, color='red')

        axes_diff[1].plot(x_mirt_tse_2[mask2], diff_mirt_ose_2[mask2], label=f"MIRT OSE - {ref_label}", ls='--', color='orange')
        axes_diff[1].plot(x_mirt_tse_2[mask2], diff_uirt_tse_2[mask2], label=f"UIRT TSE - {ref_label}", ls='-.', color='blue')
        axes_diff[1].plot(x_mirt_tse_2[mask2], diff_uirt_ose_2[mask2], label=f"UIRT OSE - {ref_label}", ls=':', color='cyan')
        axes_diff[1].set_title("Difference: Diff RC")
        axes_diff[1].set_xlabel("Score (New Form)")
        axes_diff[1].set_ylabel("Difference")
        axes_diff[1].legend()
        axes_diff[1].grid(True)

        st.pyplot(fig_diff)
        
        # Sensitivity Analysis
        st.subheader("3. Sensitivity Analysis")
        st.markdown("""
        We quantify the **divergence** (RMSD).
        *   **Same/Diff RC**: Reference is **MIRT TSE**.
        *   *Note: Scores below the sum of guessing parameters are excluded from RMSD calculation.*
        """)
        
        def calc_rmsd(y1, y2, x_scores, threshold):
            # Ensure same length
            min_len = min(len(y1), len(y2))
            y1 = y1[:min_len]
            y2 = y2[:min_len]
            x = x_scores[:min_len]
            
            mask = x >= threshold
            if np.sum(mask) == 0:
                return 0.0
            return np.sqrt(np.mean((y1[mask] - y2[mask])**2))
            
        if use_target_group:
            # Reference: MIRT TSE
            data = {
                "Comparison": ["vs MIRT OSE", "vs UIRT TSE", "vs UIRT OSE"],
                "Same RC (Ref: MIRT TSE)": [
                    calc_rmsd(y_mirt_tse_1, y_mirt_ose_1, x_mirt_tse_1, sum_g_1),
                    calc_rmsd(y_mirt_tse_1, y_uirt_tse_1, x_mirt_tse_1, sum_g_1),
                    calc_rmsd(y_mirt_tse_1, y_uirt_ose_1, x_mirt_tse_1, sum_g_1)
                ],
                "Diff RC (Ref: MIRT TSE)": [
                    calc_rmsd(y_mirt_tse_2, y_mirt_ose_2, x_mirt_tse_2, sum_g_2),
                    calc_rmsd(y_mirt_tse_2, y_uirt_tse_2, x_mirt_tse_2, sum_g_2),
                    calc_rmsd(y_mirt_tse_2, y_uirt_ose_2, x_mirt_tse_2, sum_g_2)
                ]
            }
        else:
            # Reference: Identity (Raw Score)
            # x_mirt_tse_1 is the raw score array
            data = {
                "Method": ["MIRT TSE", "MIRT OSE", "UIRT TSE", "UIRT OSE"],
                "Same RC (Ref: Identity)": [
                    calc_rmsd(y_mirt_tse_1, x_mirt_tse_1, x_mirt_tse_1, sum_g_1),
                    calc_rmsd(y_mirt_ose_1, x_mirt_tse_1, x_mirt_tse_1, sum_g_1),
                    calc_rmsd(y_uirt_tse_1, x_mirt_tse_1, x_mirt_tse_1, sum_g_1),
                    calc_rmsd(y_uirt_ose_1, x_mirt_tse_1, x_mirt_tse_1, sum_g_1)
                ],
                "Diff RC (Ref: Identity)": [
                    calc_rmsd(y_mirt_tse_2, x_mirt_tse_2, x_mirt_tse_2, sum_g_2),
                    calc_rmsd(y_mirt_ose_2, x_mirt_tse_2, x_mirt_tse_2, sum_g_2),
                    calc_rmsd(y_uirt_tse_2, x_mirt_tse_2, x_mirt_tse_2, sum_g_2),
                    calc_rmsd(y_uirt_ose_2, x_mirt_tse_2, x_mirt_tse_2, sum_g_2)
                ]
            }
        
        df_rmsd = pd.DataFrame(data)
        st.dataframe(df_rmsd.style.format("{:.4f}", subset=df_rmsd.columns[1:]))
        
        st.markdown("""
        **Interpretation:**
        1.  **Structure Sensitivity**: If the RMSD for *MIRT TSE vs UIRT TSE* is significantly higher in the **Diff RC** condition than in **Same RC**, it confirms that MIRT TSE is sensitive to the shift in the Reference Composite.
        """)

        # ==========================================
        # 4. COMPREHENSIVE EVALUATION
        # ==========================================
        st.header("4. Comprehensive Evaluation (All Conditions)")
        st.markdown("This section evaluates the RMSD for all combinations of settings (Weights & Target Group).")
        
        conditions = [
            {"name": "No Weights, Equivalent Group", "weights": False, "target": False},
            {"name": "Weights, Equivalent Group", "weights": True, "target": False},
            {"name": "No Weights, Target Group", "weights": False, "target": True},
            {"name": "Weights, Target Group", "weights": True, "target": True},
        ]
        
        with st.expander("View Condition Configurations (Latent Distributions & UIRT Parameters)"):
            st.subheader("Latent Distributions per Condition")
            config_data = []
            for cond in conditions:
                if cond['target']:
                    mu_str = "[0.5, 0.5]"
                    cov_str = "[[1, 0.5], [0.5, 1.2]]"
                else:
                    mu_str = "[0, 0]"
                    cov_str = "[[1, 0.3], [0.3, 1]]"
                
                config_data.append({
                    "Condition": cond['name'],
                    "Weights": cond['weights'],
                    "Target Group": cond['target'],
                    "Latent Mean": mu_str,
                    "Latent Cov": cov_str
                })
            st.table(pd.DataFrame(config_data))
            
            st.subheader("Converted UIRT Item Parameters")
            st.markdown("Note: UIRT parameters are projected assuming a standard normal population and do not change across conditions.")
            
            tab_uirt1, tab_uirt2, tab_uirt3 = st.tabs(["Base Form", "Same RC Form", "Diff RC Form"])
            with tab_uirt1:
                st.dataframe(uirt_base.style.format("{:.3f}"))
            with tab_uirt2:
                st.dataframe(uirt_same_rc.style.format("{:.3f}"))
            with tab_uirt3:
                st.dataframe(uirt_diff_rc.style.format("{:.3f}"))
        
        results_all = []
        progress_eval = st.progress(0, text="Evaluating conditions...")
        
        for i, cond in enumerate(conditions):
            progress_eval.progress((i + 1) / 4, text=f"Evaluating: {cond['name']}")
            
            # Set params
            c_weights = cond['weights']
            if cond['target']:
                c_mu = np.array([0.5, 0.5])
                c_cov = np.array([[1, 0.5], [0.5, 1.2]])
            else:
                c_mu = np.array([0, 0])
                c_cov = np.array([[1, 0.3], [0.3, 1]])
                
            # Run MIRT TSE & OSE for Same RC
            _, y_mirt_tse_1_c = run_mirt_tse_approx(df_base, df_same_rc, use_weights=c_weights, mu=c_mu, cov=c_cov)
            _, y_mirt_ose_1_c = run_mirt_ose(df_base, df_same_rc, mu=c_mu, cov=c_cov)
            
            # Diff RC
            _, y_mirt_tse_2_c = run_mirt_tse_approx(df_base, df_diff_rc, use_weights=c_weights, mu=c_mu, cov=c_cov)
            _, y_mirt_ose_2_c = run_mirt_ose(df_base, df_diff_rc, mu=c_mu, cov=c_cov)
            
            # Calculate RMSD
            if cond['target']:
                # Reference: MIRT TSE
                # Same RC
                rmsd_same_mirt_tse = 0.0 # Reference
                rmsd_same_ose = calc_rmsd(y_mirt_tse_1_c, y_mirt_ose_1_c, x_mirt_tse_1, sum_g_1)
                rmsd_same_uirt_tse = calc_rmsd(y_mirt_tse_1_c, y_uirt_tse_1, x_mirt_tse_1, sum_g_1)
                
                # Diff RC
                rmsd_diff_mirt_tse = 0.0 # Reference
                rmsd_diff_ose = calc_rmsd(y_mirt_tse_2_c, y_mirt_ose_2_c, x_mirt_tse_2, sum_g_2)
                rmsd_diff_uirt_tse = calc_rmsd(y_mirt_tse_2_c, y_uirt_tse_2, x_mirt_tse_2, sum_g_2)
            else:
                # Reference: Identity
                # Same RC
                rmsd_same_mirt_tse = calc_rmsd(y_mirt_tse_1_c, x_mirt_tse_1, x_mirt_tse_1, sum_g_1)
                rmsd_same_ose = calc_rmsd(y_mirt_ose_1_c, x_mirt_tse_1, x_mirt_tse_1, sum_g_1)
                rmsd_same_uirt_tse = calc_rmsd(y_uirt_tse_1, x_mirt_tse_1, x_mirt_tse_1, sum_g_1)
                
                # Diff RC
                rmsd_diff_mirt_tse = calc_rmsd(y_mirt_tse_2_c, x_mirt_tse_2, x_mirt_tse_2, sum_g_2)
                rmsd_diff_ose = calc_rmsd(y_mirt_ose_2_c, x_mirt_tse_2, x_mirt_tse_2, sum_g_2)
                rmsd_diff_uirt_tse = calc_rmsd(y_uirt_tse_2, x_mirt_tse_2, x_mirt_tse_2, sum_g_2)
            
            res = {
                "Condition": cond['name'],
                "Same RC: MIRT TSE": rmsd_same_mirt_tse,
                "Same RC: MIRT OSE": rmsd_same_ose,
                "Same RC: UIRT TSE": rmsd_same_uirt_tse,
                "Diff RC: MIRT TSE": rmsd_diff_mirt_tse,
                "Diff RC: MIRT OSE": rmsd_diff_ose,
                "Diff RC: UIRT TSE": rmsd_diff_uirt_tse
            }
            
            results_all.append(res)
            
        df_all_results = pd.DataFrame(results_all)
        st.dataframe(df_all_results.style.format("{:.4f}", subset=df_all_results.columns[1:]))

else:
    st.info("Click 'Run Simulation' to start.")