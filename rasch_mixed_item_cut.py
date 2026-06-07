import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_items(items_list):
    """
    Parses a list of items and returns a list of step difficulties (lists of floats).
    
    Each item in items_list can be:
    - A float or int: represents a dichotomous item difficulty (e.g., 0.5).
    - A list or tuple of floats: represents step difficulties directly (PCM).
    - A dictionary with keys:
        - 'step_difficulties', 'steps', or 'thresholds': list of floats.
        - 'difficulty' or 'b': float (overall difficulty).
        If both are specified, the steps are treated as relative Andrich thresholds,
        and absolute step difficulties are computed as: step_difficulty = difficulty + step.
        If only difficulty is specified, the item is parsed as dichotomous.
        
    Raises:
      ValueError: If the items_list is empty, or if item dictionary keys are invalid.
      TypeError: If item format is unsupported.
    """
    if not items_list:
        raise ValueError("The items list cannot be empty.")
        
    parsed_items = []
    for item in items_list:
        if isinstance(item, (int, float)):
            parsed_items.append([float(item)])
        elif isinstance(item, (list, tuple)):
            if len(item) == 0:
                raise ValueError("An item cannot have empty step difficulties.")
            parsed_items.append([float(x) for x in item])
        elif isinstance(item, dict):
            # Safe check to support 0.0 or other falsy numeric values
            steps = item.get("step_difficulties")
            if steps is None:
                steps = item.get("steps")
            if steps is None:
                steps = item.get("thresholds")
                
            difficulty = item.get("difficulty")
            if difficulty is None:
                difficulty = item.get("b")
            
            if steps is not None:
                if len(steps) == 0:
                    raise ValueError(f"Steps list cannot be empty for item: {item}")
                if difficulty is not None:
                    # Treat steps as relative thresholds (Andrich thresholds)
                    parsed_items.append([float(difficulty) + float(s) for s in steps])
                else:
                    # Treat steps as absolute step difficulties (PCM)
                    parsed_items.append([float(s) for s in steps])
            elif difficulty is not None:
                parsed_items.append([float(difficulty)])
            else:
                raise ValueError(f"Invalid item dictionary structure. Provide 'difficulty' and/or 'steps': {item}")
        else:
            raise TypeError(f"Unsupported item format of type {type(item)}: {item}")
            
    return parsed_items

def compute_item_stats(theta, steps):
    """
    Computes the expected score and response variance for a single PCM item at ability level(s) theta.
    
    Parameters:
      theta: float or numpy array of shape (T,)
        The person ability level(s).
      steps: list or numpy array of shape (m,)
        The absolute step difficulties of the item (delta_i1, ..., delta_im).
        
    Returns:
      expected_score: float or numpy array of shape (T,)
        The expected score for this item.
      variance: float or numpy array of shape (T,)
        The variance of the item score (first derivative of expected score w.r.t theta).
    """
    is_scalar = np.isscalar(theta)
    theta_arr = np.atleast_1d(theta).astype(float)
    steps = np.array(steps, dtype=float)
    m = len(steps)
    T = len(theta_arr)
    
    # Compute exponent terms S_ik = sum_{j=1}^k (theta - step_j)
    # S_i0 = 0
    diffs = theta_arr[:, np.newaxis] - steps  # shape (T, m)
    cumsum_terms = np.cumsum(diffs, axis=1)   # shape (T, m)
    
    # Build complete S matrix of shape (T, m + 1)
    S = np.zeros((T, m + 1))
    S[:, 1:] = cumsum_terms
    
    # Numerically stable softmax using the max subtraction trick
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    prob = exp_S / np.sum(exp_S, axis=1, keepdims=True)  # shape (T, m + 1)
    
    # Expected score: Sum_{x=0}^m x * P_ix
    categories = np.arange(m + 1)
    expected_score = np.sum(categories * prob, axis=1)
    
    # Variance: Sum_{x=0}^m x^2 * P_ix - Expected^2
    variance = np.sum((categories ** 2) * prob, axis=1) - (expected_score ** 2)
    
    if is_scalar:
        return float(expected_score[0]), float(variance[0])
    return expected_score, variance

def compute_tcc_and_slope(theta, parsed_items):
    """
    Computes the Test Characteristic Curve (TCC) expected score and total information (slope) at theta.
    
    Parameters:
      theta: float or numpy array of shape (T,)
        Ability level(s).
      parsed_items: list of lists of step difficulties.
      
    Returns:
      tcc: float or numpy array of shape (T,)
        Expected raw test score.
      slope: float or numpy array of shape (T,)
        Slope of TCC (sum of item variances), which is also the Test Information.
    """
    if not parsed_items:
        raise ValueError("parsed_items list cannot be empty.")
        
    is_scalar = np.isscalar(theta)
    theta_arr = np.atleast_1d(theta)
    
    tcc = np.zeros_like(theta_arr, dtype=float)
    slope = np.zeros_like(theta_arr, dtype=float)
    
    for steps in parsed_items:
        exp_score, var = compute_item_stats(theta_arr, steps)
        tcc += exp_score
        slope += var
        
    if is_scalar:
        return float(tcc[0]), float(slope[0])
    return tcc, slope

def raw_to_logit(raw_score, parsed_items, extrscore=0.3, tol=1e-7, max_iter=100):
    """
    Estimates the logit measure and model standard error corresponding to a raw score
    under the Rasch model for mixed item types (PCM).
    
    Parameters:
      raw_score: float or int
        The raw cut score.
      parsed_items: list of lists of step difficulties (output of parse_items).
      extrscore: float
        Winsteps-style adjustment for extreme scores (default 0.3).
      tol: float
        Newton-Raphson convergence tolerance.
      max_iter: int
        Maximum Newton-Raphson iterations.
        
    Returns:
      dict containing:
        'raw_score': original raw score
        'adjusted_score': adjusted score for estimation
        'logit': estimated ability measure (logits)
        'se': model standard error (logits)
        'converged': boolean indicating convergence
        'iterations': number of iterations taken
        
    Raises:
      ValueError: If parsed_items is empty, or if raw_score is out of bounds.
    """
    if not parsed_items:
        raise ValueError("parsed_items list cannot be empty.")
        
    max_score = float(sum(len(steps) for steps in parsed_items))
    if max_score <= 0.0:
        raise ValueError("Maximum possible test score must be greater than zero.")
        
    raw_score = float(raw_score)
    if raw_score < 0.0 or raw_score > max_score:
        raise ValueError(f"Raw cut score {raw_score} is out of test boundaries [0, {max_score}].")
        
    # Clamp extreme score adjustment to prevent invalid boundaries on very short tests
    adj_val = min(extrscore, max_score * 0.5 - 1e-5)
    
    # 1. Apply Winsteps-style adjustment for extreme scores
    if raw_score <= 0.0:
        adjusted_score = adj_val
    elif raw_score >= max_score:
        adjusted_score = max_score - adj_val
    else:
        adjusted_score = raw_score
        
    # 2. Sensible starting value: logit(p) plus mean difficulty
    p = adjusted_score / max_score
    all_steps = [s for steps in parsed_items for s in steps]
    mean_difficulty = np.mean(all_steps) if all_steps else 0.0
    theta = np.log(p / (1.0 - p)) + mean_difficulty
    
    converged = False
    iterations = 0
    
    # 3. Newton-Raphson search
    for _ in range(max_iter):
        tcc, slope = compute_tcc_and_slope(theta, parsed_items)
        
        # Guard against zero slope/variance
        if slope < 1e-12:
            slope = 1e-12
            
        diff = tcc - adjusted_score
        step = diff / slope
        theta_new = theta - step
        
        iterations += 1
        if abs(step) < tol:
            theta = theta_new
            converged = True
            break
            
        theta = theta_new
        
    # 4. Standard Error at final theta: 1 / sqrt(Information)
    _, final_slope = compute_tcc_and_slope(theta, parsed_items)
    se = 1.0 / np.sqrt(final_slope) if final_slope > 0 else np.nan
    
    return {
        "raw_score": raw_score,
        "adjusted_score": adjusted_score,
        "logit": theta,
        "se": se,
        "converged": converged,
        "iterations": iterations
    }

def generate_conversion_table(items_list, extrscore=0.3, tol=1e-7, max_iter=100):
    """
    Generates a complete raw score-to-logit conversion table (similar to Winsteps SCOREFILE).
    
    Parameters:
      items_list: list of item definitions.
      extrscore: float (extreme score adjustment, default 0.3).
      
    Returns:
      pandas.DataFrame
        DataFrame with columns: Raw Score, Adjusted Score, Logit Measure, Model SE, Converged, Iterations
    """
    parsed_items = parse_items(items_list)
    max_score = sum(len(steps) for steps in parsed_items)
    
    rows = []
    for raw in range(max_score + 1):
        res = raw_to_logit(raw, parsed_items, extrscore=extrscore, tol=tol, max_iter=max_iter)
        rows.append({
            "Raw Score": raw,
            "Adjusted Score": res["adjusted_score"],
            "Logit Measure": res["logit"],
            "Model SE": res["se"],
            "Converged": res["converged"],
            "Iterations": res["iterations"]
        })
        
    return pd.DataFrame(rows)

def plot_tcc_and_tif(items_list, conversion_df=None, cut_score=None, save_path=None):
    """
    Plots the Test Characteristic Curve (TCC) and Test Information Function (TIF),
    with optional visualization of a specific cut score.
    
    Parameters:
      items_list: list of item definitions.
      conversion_df: optional DataFrame from generate_conversion_table to overlay raw points.
      cut_score: optional float or int, raw cut score to visualize.
      save_path: optional file path to save the generated figure.
      
    Returns:
      matplotlib.figure.Figure
    """
    parsed_items = parse_items(items_list)
    max_score = sum(len(steps) for steps in parsed_items)
    
    # Determine plot bounds based on step difficulties
    all_steps = [s for steps in parsed_items for s in steps]
    if all_steps:
        min_diff = min(all_steps)
        max_diff = max(all_steps)
        theta_min = min(-4.0, min_diff - 2.0)
        theta_max = max(4.0, max_diff + 2.0)
    else:
        theta_min, theta_max = -4.0, 4.0
        
    thetas = np.linspace(theta_min, theta_max, 300)
    tcc, slope = compute_tcc_and_slope(thetas, parsed_items)
    
    # Modern styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. TCC Plot
    ax1.plot(thetas, tcc, color='#1f77b4', linewidth=2.5, label='Expected Test Score')
    if conversion_df is not None:
        ax1.scatter(conversion_df['Logit Measure'], conversion_df['Raw Score'], 
                    color='#e377c2', s=30, zorder=5, label='Raw-to-Logit Points')
                    
    # Draw cut lines if cut_score is specified
    if cut_score is not None:
        res = raw_to_logit(cut_score, parsed_items)
        logit_cut = res['logit']
        adj_cut = res['adjusted_score']
        
        # Horizontal line from y-axis to the curve
        ax1.hlines(y=adj_cut, xmin=theta_min, xmax=logit_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        # Vertical line from the curve to x-axis
        ax1.vlines(x=logit_cut, ymin=-0.1, ymax=adj_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        # Intersection point marker
        ax1.scatter([logit_cut], [adj_cut], color='#d62728', s=120, marker='*', zorder=10, 
                    label=f'Cut Score: Raw {cut_score} -> {logit_cut:.3f} logits')
        # Annotation text
        ax1.annotate(f"Cut: {logit_cut:.3f} logits", 
                     xy=(logit_cut, adj_cut), 
                     xytext=(logit_cut - 1.5 if logit_cut > 0 else logit_cut + 0.5, adj_cut - 1.5),
                     arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                     fontsize=10, color='#d62728', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="orange"))
        
    ax1.set_title('Test Characteristic Curve (TCC)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Person Ability (θ in logits)', fontsize=12)
    ax1.set_ylabel('Expected Raw Score', fontsize=12)
    ax1.set_xlim(theta_min, theta_max)
    ax1.set_ylim(-0.1, max_score + 0.1)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. TIF Plot
    ax2.plot(thetas, slope, color='#2ca02c', linewidth=2.5, label='Test Information (I(θ))')
    if conversion_df is not None:
        ax2.scatter(conversion_df['Logit Measure'], 1.0 / (conversion_df['Model SE']**2), 
                    color='#e377c2', s=30, zorder=5, label='Raw-to-Info Points')
                    
    # Draw cut lines if cut_score is specified
    if cut_score is not None:
        res = raw_to_logit(cut_score, parsed_items)
        logit_cut = res['logit']
        _, final_slope = compute_tcc_and_slope(logit_cut, parsed_items)
        tif_cut = final_slope
        
        # Horizontal line from y-axis to the curve
        ax2.hlines(y=tif_cut, xmin=theta_min, xmax=logit_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        # Vertical line from the curve to x-axis
        ax2.vlines(x=logit_cut, ymin=0.0, ymax=tif_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        # Intersection point marker
        ax2.scatter([logit_cut], [tif_cut], color='#d62728', s=120, marker='*', zorder=10, 
                    label=f'Info at Cut: {tif_cut:.3f}')
        # Annotation text
        ax2.annotate(f"I(θ) = {tif_cut:.3f}", 
                     xy=(logit_cut, tif_cut), 
                     xytext=(logit_cut - 1.5 if logit_cut > 0 else logit_cut + 0.5, tif_cut - 0.2 * max(slope) if tif_cut > 0.3 * max(slope) else tif_cut + 0.1 * max(slope)),
                     arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                     fontsize=10, color='#d62728', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="orange"))
        
    ax2.set_title('Test Information Function (TIF)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Person Ability (θ in logits)', fontsize=12)
    ax2.set_ylabel('Test Information (I(θ))', fontsize=12)
    ax2.set_xlim(theta_min, theta_max)
    ax2.set_ylim(0.0, max(slope) * 1.1)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_tcc_and_se(items_list, conversion_df=None, cut_score=None, save_path=None):
    """
    Backward-compatible wrapper to plot TCC and Standard Error of Measurement (CSEM).
    """
    parsed_items = parse_items(items_list)
    max_score = sum(len(steps) for steps in parsed_items)
    
    all_steps = [s for steps in parsed_items for s in steps]
    if all_steps:
        min_diff = min(all_steps)
        max_diff = max(all_steps)
        theta_min = min(-4.0, min_diff - 2.0)
        theta_max = max(4.0, max_diff + 2.0)
    else:
        theta_min, theta_max = -4.0, 4.0
        
    thetas = np.linspace(theta_min, theta_max, 300)
    tcc, slope = compute_tcc_and_slope(thetas, parsed_items)
    se = 1.0 / np.sqrt(slope)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(thetas, tcc, color='#1f77b4', linewidth=2.5, label='Expected Test Score')
    if conversion_df is not None:
        ax1.scatter(conversion_df['Logit Measure'], conversion_df['Raw Score'], 
                    color='#e377c2', s=30, zorder=5, label='Raw-to-Logit Points')
    if cut_score is not None:
        res = raw_to_logit(cut_score, parsed_items)
        logit_cut = res['logit']
        adj_cut = res['adjusted_score']
        ax1.hlines(y=adj_cut, xmin=theta_min, xmax=logit_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        ax1.vlines(x=logit_cut, ymin=-0.1, ymax=adj_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        ax1.scatter([logit_cut], [adj_cut], color='#d62728', s=120, marker='*', zorder=10)
        ax1.annotate(f"Cut: {logit_cut:.3f} logits", xy=(logit_cut, adj_cut), 
                     xytext=(logit_cut - 1.5 if logit_cut > 0 else logit_cut + 0.5, adj_cut - 1.5),
                     arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                     fontsize=10, color='#d62728', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="orange"))
                     
    ax1.set_title('Test Characteristic Curve (TCC)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Person Ability (θ in logits)', fontsize=12)
    ax1.set_ylabel('Expected Raw Score', fontsize=12)
    ax1.set_xlim(theta_min, theta_max)
    ax1.set_ylim(-0.1, max_score + 0.1)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(thetas, se, color='#2ca02c', linewidth=2.5, label='Model Standard Error')
    if conversion_df is not None:
        ax2.scatter(conversion_df['Logit Measure'], conversion_df['Model SE'], 
                    color='#e377c2', s=30, zorder=5, label='Raw-to-SE Points')
    if cut_score is not None:
        res = raw_to_logit(cut_score, parsed_items)
        logit_cut = res['logit']
        se_cut = res['se']
        ax2.hlines(y=se_cut, xmin=theta_min, xmax=logit_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        ax2.vlines(x=logit_cut, ymin=0.0, ymax=se_cut, colors='#d62728', linestyles='dashed', linewidths=1.5)
        ax2.scatter([logit_cut], [se_cut], color='#d62728', s=120, marker='*', zorder=10)
        ax2.annotate(f"SE = {se_cut:.3f}", xy=(logit_cut, se_cut), 
                     xytext=(logit_cut - 1.5 if logit_cut > 0 else logit_cut + 0.5, se_cut + 0.15),
                     arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
                     fontsize=10, color='#d62728', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="orange"))
                     
    ax2.set_title('Standard Error of Measurement (CSEM)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Person Ability (θ in logits)', fontsize=12)
    ax2.set_ylabel('Standard Error (Logits)', fontsize=12)
    ax2.set_xlim(theta_min, theta_max)
    ax2.legend(loc='upper center', frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
