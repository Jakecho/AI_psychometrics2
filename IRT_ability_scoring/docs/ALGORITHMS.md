# Algorithms and implementation

```
irt_models/   dichotomous.py · partial_credit.py · grm.py   category log-probabilities and log-derivatives
              base.py        Item, pattern_terms (ℓ, ℓ′, ℓ″, I, J, I′, T, T′ per examinee), loglik_grid
scoring/      solvers.py     maximize (MLE, MAP) and find_root (WLE, TCC), vectorised over examinees, recorded
              mle.py · map_estimate.py · wle.py · tcc.py · eap.py · eapsum.py · extras.py
              settings.py    ScoringSettings (defaults = irtQ: N(0,1), 41 Gauss–Hermite nodes, range ±5)
simulation/   data.py        examples A–D, item and response generation, Dataset
              monte_carlo.py replications, bias / RMSE / MAD / coverage / MCSE, conditional statistics
r_bridge/     bridge.py · compare.py · irtq_scores.R · mirt_scores.R · tam_scores.R
components/   ui.py          layout, shared state, persistent sidebar widgets, step controls, tables
              explorer.py    Newton and root-finding step-through views
pages/        1–10 learning pages; IRT_ability_scoring.py is the home page
```

## Numerical safeguards

| Risk | Safeguard |
|---|---|
| Underflow of L(θ) on long tests | All likelihoods kept as log-likelihoods; posterior normalised with log-sum-exp |
| Cancellation in logistic tails | log Ψ via `scipy.special.log_expit`; 1 − Ψ computed as Ψ(−z) |
| GRM P*_k − P*_{k+1} when both ≈ 1 | difference taken in the lower tail Ψ(−z_{k+1}) − Ψ(−z_k), in logs with log1p |
| Divide-by-total overflow | log-softmax of the cumulative numerators |
| ℓ″ ≥ 0 (3PL/4PL) | Fisher-scoring step with the expected information |
| Overshooting | step capped at 1 logit; step halved while the objective decreases |
| Divergence (extreme patterns) | search range ±bound; the bounded value is never reported as an MLE |
| Multiple maxima (3PL/4PL) | comparison with an 801-point grid; Newton restarted from the grid maximum |
| WLE / TCC root | bracket with a sign change; Newton steps leaving the bracket replaced by bisection |
| Root outside the range / outside (T(−∞), T(+∞)) | reported as "no finite solution", not as a number |
| Examinees without responses | excluded from solving; reported as "no observed responses" |
| Quadrature accuracy | convergence table against `scipy.integrate.quad` on every EAP page |

## Step-through explorers

The solver runs once from the chosen starting value with `record=True`; each iteration stores θ⁽ᵗ⁾, the objective,
score, second derivative, curvature used, step type, raw and taken step and θ⁽ᵗ⁺¹⁾ (root finder: U, U′ and the
bracket). The explorer shows those records; item tables are recomputed at the recorded θ⁽ᵗ⁾ with the same
`pattern_terms`, so table sums equal the recorded values (tested). EAP stages (prior → likelihood → unnormalised
posterior → normalisation → mean → SD) reveal the columns of one real node table.

## State

`st.session_state`: `datasets` (name → Dataset), `active`, `person`, `cfg_*` scoring settings (persisted across pages
through hidden `w_*` widget keys), step positions `<lab>_pos`, cached whole-dataset results keyed by dataset identity +
settings, and reference-check results tagged with the settings and pattern they were computed for (shown only while
those still match).
