# Implementation

```text
Rasch_calibration/
├── Rasch_calibration.py   home page (entry point); app.py is an alias
├── pages/                 1_…9_ Streamlit pages
├── rasch/                 model, simulation, jmle, cmle, mml_em, scoring, metrics, validation, linking
├── r_bridge/              bridge.py + tam_jmle.R, erm_cmle.R, tam_mml.R, tam_wle.R, common.R, check_packages.R
├── components/ui.py       layout, shared state, charts, tables, teaching widgets
├── tests/                 pytest suite (engines, references, Streamlit AppTest)
└── docs/
```

## Numerical methods

| Component | Method and safeguards |
|---|---|
| Probabilities | `scipy.special.expit`; log-likelihoods via `logaddexp` (no `log(0)`) |
| Missing data | explicit observation mask everywhere; a missing response is never scored as 0 |
| JMLE | optional `fixed_b` (NaN = free): fixed items never updated, no centring, bias factor on free items only; alternating vectorised Newton blocks (≤ 10 steps each, step capped at 1 logit), centring, TAM-style `adj` and bias factor, stop when max \|Δθ\|, \|Δb\| < tol; every iteration stores its inputs and the abilities used by the item step |
| CMLE | optional `fixed_b` ($b=c+Az$ over the free items); persons grouped by (observed item set, raw score), which carry every ESF term, so a two-form design needs at most forms × scores rows; ESFs and leave-one-out ESFs by a log-space dynamic programme, vectorised over those rows (missing items get ε = 0); analytic gradient; L-BFGS-B on J − 1 free parameters with $b_J=-\sum$ others; SEs from a finite-difference Hessian of the exact gradient; convergence judged on max \|score\| < 1e-4 |
| MML-EM | log-sum-exp E-step; expected counts by matrix products; vectorised safeguarded Newton M-step (≤ 20 steps, capped); variance by the posterior second moment; stop when max \|Δb\|, \|Δσ\| < tol; each cycle stores b, σ and the M-step Newton trace |
| MML-EM, groups | `mml_em_groups(X, groups, fixed_b)`: one normal prior per group on the same grid; free items by Newton; group means (first fixed at 0 unless anchored) and SDs from posterior moments; complete-data SEs; anchor displacement from expected counts; identical to `mml_em` for one group |
| Linking | `rasch/linking.py`: two-form simulator (anchors, internal common items, group mean/SD, drift, bank error); four designs; anchors dropped from the anchor set are split by form in the stacked designs; median-centred anchor stability; recovery on the non-anchor items (same set whatever the anchor choice); Monte Carlo with items and bank fixed |
| Scoring | Brent root-finding on [−30, 30] for MLE and WLE; extreme MLEs returned as ±∞; Newton traces for the walkthrough |
| Walkthroughs | recomputed from the recorded inputs of the selected iteration (never from final estimates) |

No probabilities or parameters are clipped silently; step limits are documented controls.

## R bridge

`run_reference(method, X, settings)` writes `responses.csv` (blank = missing) and `settings.json` to a
temporary folder, runs `Rscript --vanilla <script> <folder>`, and reads `result.json`. Status is
**Executed**, **Unavailable** (no R / package; reason given) or **Failed** (stderr captured). Package
versions are recorded with each result. Conventions verified against the installed packages:

- `tam.jml`: `xsi` is the difficulty; defaults `adj = 0.3`, `bias = TRUE`, `constraint = "cases"`.
- `RM`: `betapar` is easiness, so difficulty = −`betapar`; `sum0 = TRUE`.
- `tam.mml`: Rasch, `constraint = "cases"`, nodes passed explicitly; `se.xsi` is complete-data; `ic$loglike`
  uses density × spacing weights.
- `tam.wle` with all `xsi` fixed via `tam.mml(xsi.fixed = …)`.
- Optional settings for linking runs: `fixed_index` / `fixed_value` → `xsi.fixed` in `tam.jml` and
  `tam.mml`; `groups` → `tam.mml(group = …)`, whose `beta` rows are the group means (first fixed at 0
  without anchors, all free with them) and whose `variance` gives each group's variance. eRm cannot fix
  item parameters, so fixed-anchor CMLE runs are reported as "No R reference".

## Application state

`st.session_state.dataset` holds the shared `Dataset` (responses, IDs, truth when simulated, settings,
identity hash). Results are stored per method with the dataset identity and a hash of the settings;
pages show a result only when both match, otherwise a "settings changed" notice. Reference results are
cached the same way. Downloads serialise the tables currently displayed.
