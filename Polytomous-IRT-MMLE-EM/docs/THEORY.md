# Theory

## Models (canonical parameterisation, scaling constant D, default 1)

| Model | Category probability | Free item parameters |
|---|---|---|
| PCM | P_k ∝ exp[D Σ_{h≤k}(θ − δ_h)] | steps δ_h |
| RSM | δ_h = b + τ_h, Σ_h τ_h = 0 | location b per item; τ shared by items with the same categories |
| GPCM | P_k ∝ exp[D a Σ_{h≤k}(θ − δ_h)], a > 0 | a, δ_h |
| GRM | P(Y ≥ k) = logistic[D a(θ − b_k)], b_1 < … < b_K; P_k = P(Y ≥ k) − P(Y ≥ k+1) | a, ordered b_k |

Empty sums are zero. Adjacent-category log-odds are log(P_k/P_{k−1}) = D a(θ − δ_k), so adjacent curves cross at
θ = δ_k. A GRM threshold b_k is where P(Y ≥ k) = 0.5; it is a cumulative boundary, not an adjacent-category step.

Slope–intercept forms: adjacent models P_k ∝ exp(k·αθ + γ_k), α = D a, γ_k = −D a Σ_{h≤k} δ_h (mirt's d_k);
GRM P(Y ≥ k) = logistic(αθ + c_k), c_k = −D a b_k (mirt's d_k). TAM's PCM `xsi` are the δ_jh; TAM's RSM `xsi` are
b_j and τ_1..τ_{K−1} (τ_K = −Σ). irtQ's `GPCM` `d` are the δ_h and its `GRM` `d` are the b_k; irtQ fits PCM as
GPCM with `fix.a.gpcm = TRUE` and has no RSM. Round trips are tested by reconstructing probabilities.

Information: I(θ) = Σ_k (∂P_k/∂θ)²/P_k; for adjacent models this equals D²a²·Var(Y | θ).

Numerical stability: adjacent models use log-sum-exp; the GRM uses
log P_k = log σ(x_k) + log σ(−x_{k+1}) + log(1 − e^{x_{k+1} − x_k}) with x_k = D a(θ − b_k).

## Marginal likelihood

ℓ(ξ, π) = Σ_g Σ_{i∈g} log Σ_q π_gq Π_{j∈O_ig} P_{j, y_igj}(θ_gq; ξ)

O_ig are the items examinee i actually answered: responses missing by design never enter the product (they are not
zeros). The inner sum is computed with log-sum-exp.

## EM

E-step: w_igq = π_gq L_i(θ_gq) / Σ_s π_gs L_i(θ_gs), with L_i the product above. Σ_q w_igq = 1.

Expected counts: C_gjkq = Σ_{i∈g} I(Y_ij = k) w_igq. Summed over q they return the observed category counts.

Item M-step: maximise Q_item(ξ) = Σ_{g,j,k,q} C_gjkq log P_jk(θ_gq; ξ) over the free items, with fixed anchors
excluded. The objective separates by item except for RSM, where all items sharing τ form one problem.

| Model | Optimiser variables (L-BFGS-B, analytic gradient) |
|---|---|
| PCM | δ_h |
| RSM | free b_j and τ_1..τ_{K−1} jointly (τ_K = −Σ τ_h); τ fixed when anchors carry it |
| GPCM | log a, δ_h |
| GRM | log a, b_1, log(b_k − b_{k−1}), which keeps thresholds ordered |

A step is accepted only if it does not lower Q_item, so the item update is at least a generalised EM step. In
practice each item's Q is maximised to the optimiser tolerance. The bounds are |location| ≤ 10, 0.05 ≤ a ≤ 8 and
gaps between 1e-3 and 30. Estimates that end at a bound are flagged; this happens with empty categories or weak
identification.

Prior M-step: see [PRIOR_DISTRIBUTIONS.md](PRIOR_DISTRIBUTIONS.md).

Convergence: the largest item-parameter change is below `tol`. When a distribution is estimated, the largest
prior-mass change must also be below `prior_tol` and the largest change in group moments below `tol`. Every cycle
records ℓ, Δℓ, the three changes, each inner optimiser's status and a flag for any decrease in ℓ. For an anchored
MEM run, the first cycle's E-step uses the anchors only, so the drop from cycle 1 to cycle 2 is not flagged.

## Identification

For GPCM and GRM, θ* = (θ − m)/s with a* = a·s and δ* = (δ − m)/s leaves every probability unchanged. Without
anchors, the reference distribution must therefore fix both its mean and its SD. For PCM and RSM the slope is fixed
at 1, so only the location is indeterminate: the reference mean is fixed and its SD is estimated (the TAM
convention).

With anchors (FIPC, MG-FIPC), the fixed bank parameters set both location and unit, and every group's distribution
is free. The lab warns when there is only one anchor, or when a group sees no anchor and is linked only through
free items. It refuses to run when groups are not connected to an anchor, or, without anchors, not connected to each
other.

## Standard errors

Observed information comes from a numerical Jacobian of the Fisher-identity gradient of ℓ with respect to the free
canonical item parameters (central differences). The latent distribution is held at its estimate, so these SEs are
conditional on it. A test checks them against a finite-difference Hessian of ℓ itself.
