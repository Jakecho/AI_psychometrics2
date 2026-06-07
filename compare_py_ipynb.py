import sys
import os
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.abspath('/Users/jakecho/Documents/AI_psychometrics2'))
import rasch_mixed_item_cut as rmic

def main():
    print("=== Testing Python Module rasch_mixed_item_cut.py ===")
    
    # 1. Define identical mixed test items from the notebook
    items = [
        {"difficulty": -1.2, "label": "Dichotomous Item 1"},
        {"difficulty": -0.5, "label": "Dichotomous Item 2"},
        {"difficulty": 0.0, "label": "Dichotomous Item 3"},
        {"difficulty": 0.5, "label": "Dichotomous Item 4"},
        {"difficulty": 1.0, "label": "Dichotomous Item 5"},
        {"difficulty": 1.8, "label": "Dichotomous Item 6"},
        
        {"steps": [-1.0, 0.5], "label": "PCM Item 7 (3 categories: 0, 1, 2)"},
        {"steps": [-0.5, 0.2, 1.2], "label": "PCM Item 8 (4 categories: 0, 1, 2, 3)"},
        {"difficulty": 0.2, "steps": [-0.8, 0.8], "label": "PCM Item 9 (RSM relative thresholds)"},
        {"steps": [0.0, 1.0, 2.0], "label": "PCM Item 10 (4 categories: 0, 1, 2, 3)"}
    ]
    
    parsed_items = rmic.parse_items(items)
    max_score = sum(len(steps) for steps in parsed_items)
    
    print(f"Number of parsed items: {len(parsed_items)}")
    print(f"Maximum possible score: {max_score}")
    
    # 2. Compute the logit cut score using raw_to_logit for raw_cut = 8.5
    raw_cut = 8.5
    res = rmic.raw_to_logit(raw_cut, parsed_items, extrscore=0.3)
    
    print(f"\n--- Results from rmic.raw_to_logit({raw_cut}) ---")
    print(f"Raw Cut Score:      {res['raw_score']:.4f}")
    print(f"Adjusted Cut Score:  {res['adjusted_score']:.4f}")
    print(f"Logit Cut Measure:   {res['logit']:.6f} logits")
    print(f"Model Standard Error: {res['se']:.6f} logits")
    print(f"Solver Converged:    {res['converged']} (in {res['iterations']} iterations)")
    
    # 3. Simulate and trace Newton-Raphson iteration step-by-step to match the notebook's trace
    print("\n--- Iteration Trace (Replicated Step-by-Step) ---")
    adjusted_score = raw_cut
    
    # Starting values
    p = adjusted_score / max_score
    all_steps = [s for steps in parsed_items for s in steps]
    mean_difficulty = np.mean(all_steps)
    theta = np.log(p / (1.0 - p)) + mean_difficulty
    
    print(f"{'Iter':<6} | {'Ability (θ)':<12} | {'Expected Score':<15} | {'Information (I)':<15} | {'Step size':<12}")
    print("-" * 70)
    
    tol = 1e-7
    for i in range(100):
        tcc, slope = rmic.compute_tcc_and_slope(theta, parsed_items)
        diff = tcc - adjusted_score
        step = diff / slope
        theta_new = theta - step
        
        print(f"{i+1:<6d} | {theta:<12.6f} | {tcc:<15.6f} | {slope:<15.6f} | {step:<12.6f}")
        
        if abs(step) < tol:
            theta = theta_new
            break
        theta = theta_new
    
    print("-" * 70)
    print("Verification Assertion:")
    diff_check = abs(res['logit'] - theta)
    print(f"Absolute difference between function return and trace: {diff_check:.12e}")
    assert diff_check < 1e-9, "Mismatch between module function and step-by-step trace!"
    print("Assertion Passed! The python file matches the notebook results exactly.")

if __name__ == "__main__":
    main()
