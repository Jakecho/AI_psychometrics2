# MMLE–EM Learning Lab

Run from this folder using the working system Python:

```sh
cd /Users/jakecho/Documents/AI_psychometrics2
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m streamlit run MMLE_EM.py --server.address 127.0.0.1 --server.port 8511
```

Use the framework Python by full path. If the project `.venv` is activated, a bare
`python3` resolves to it instead; that venv has Streamlit 1.55 (the app needs
≥ 1.58 for `st.iframe`) and lives in iCloud-synced storage, where imports can
stall while placeholder files download.

The app has six main tabs: Design, EM walkthrough, Results, Validation,
Compare, and Method notes. The bundled example contains 1,000 people and
20 binary 2PL items. It is data, not a claim of completed validation.
Generate/Regenerate explicitly replaces the dataset; fitting settings,
selection changes and animation replay reuse saved responses.

The current environment already supplies NumPy, SciPy, pandas, Plotly and
Streamlit. `requirements-mmle.txt` lists the app's dependencies separately
from unrelated workspace applications. No packages were installed or
unrelated requirements changed during implementation.

## Implemented computation

`mmle_engine.py` independently implements unidimensional 2PL and 3PL EM,
with numerically stable log likelihoods/posteriors and joint direct a/b/c
SciPy M-steps. R is not used to compute the Python estimate.

All four designs use the same engine: standard, concurrent, fixed bank
anchors, and concurrent plus bank anchors. One global item ID has one
coefficient vector, expected counts pool across participating groups, and
whole bank rows remain locked. Necessary observed-response connections to
scale sources are checked; connectivity is not a proof of identification.

Normal versus empirical histogram is a separate choice. Reference normal
weights remain fixed; free normal groups use posterior moment updates.
Reference histograms use the reviewed irtQ interpolation/exponential-tail
rescaling rule; free histograms retain posterior-average masses. Numerical
grid moments and density stability remain visible.

Anchored OEM fits free items once using fixed-item posteriors under the
initial population distribution, then returns updated population masses
without feedback. MEM uses anchors in the first E-step and all observed
items from cycle two onward. Default stopping checks free-item changes and
optimizer success; an optional stricter density rule is explicitly an app
choice. All-fixed fits use mean/variance changes.

Priors are independently enabled: lognormal on D×a, Normal on b, and Beta
on free 3PL c. Pure MMLE turns all item priors off. The irtQ reset enables
Beta(5,16) guessing priors in 3PL, with other item priors off. Raw marginal
likelihood and prior-penalized objectives are separate. Default a/b fitting
domains are unbounded; optional positive-a fitting is labeled an app
restriction. The engine optimizes the original direct-coefficient target,
without a transform Jacobian.

## Data, teaching and replay

Simulation supports normal, standardized skewed and bimodal populations,
literal custom ability nodes/masses, editable item truth, editable forms,
missingness, and separately supplied bank coefficients/drift. Generation
truth and fitted population assumptions are independent. Uploaded responses
accept a `group` column plus one global column per item. An administration
CSV distinguishes absent items from missing administered responses. Bank
metadata uses `id,fixed,a,b,c`; duplicate same-ID bank rows must agree.
Mixed binary item-model metadata is not supported: each run selects 2PL or
3PL for its free items.

`mmle_animation.py` implements E-step and convergence players using saved
input/output states. Selected-person posteriors are deterministically
reconstructed from the saved input and original responses. Every frame has
real expected counts, old/new curves, population updates and diagnostics.
Play/Pause, Restart, Previous/Next, cycle scrubbing, speed, range selection,
keyboard controls, reduced motion and numerical tables are implemented.
No interpolation or final-estimate-derived fake trajectory is used.

Simulation recovery aligns truth to the fitted reference scale using one
shared transformation for all groups/items. Bank runs retain supplied bank
coordinates and deliberate drift. Fixed anchors are excluded from free-item
recovery, and shared coefficients are counted once. Recovery is unavailable
when generating and fitted D differ. No standard-error/coverage claim is
made.

Compare stores copied run/data snapshots; incompatible datasets or scales
are not directly ranked. Simulation comparisons use the shared mapping from
known generating coordinates; bank drift is not transformed away.
ZIP export includes responses, administration, bank/truth, generation seed,
settings, versions/code hash, recorded cycles/counts and executed/pending
validation evidence. CSV import/export preserves the separate masks.

## Primary reference validation

`mmle_reference.py` and `mmle_irtq.R` call the real installed `irtQ::est_irt`
or `irtQ::est_mg`, matching data, starting coefficients, D, quadrature,
initial masses, priors, shared IDs, bank locks and OEM/MEM schedule. R uses
`nlminb`; Python uses L-BFGS-B, so iteration paths differ and only endpoints
are compared: coefficients, response curves, group moments, total-variation
mass distances and likelihoods recomputed on the common grid.

**Reference installation.** This machine has R 4.4.3. irtQ 1.2.0 and later
(current CRAN: 1.3.1) require R ≥ 4.5, so irtQ **1.1.0**, the last release for
R 4.4, is installed in the project's private R library
`~/.cache/ai_psychometrics2/R-library` (shared with the MHRM app's mirt; nothing
system-wide changed). The adapter puts that library first via `R_LIBS`; set
`MMLE_R_LIBRARY` to use another location. irtQ 1.1.0's EM code matches 1.3.0
except for the stopping test: before 1.3.0 it checks the signed maximum change,
so it can stop while a coefficient is still decreasing. Every irtQ endpoint is
therefore checked for **stationarity**: one Python EM cycle is run from it, and
a free coefficient moving more than 10× the item tolerance means irtQ stopped
early. That can block a Pass but never excuses a Fail (the check itself uses
Python's EM, so it must not hide a Python error).

**Executed baseline (2026-11-01).** Python vs irtQ 1.1.0 on all 16
design × model × density configurations (N = 500 per group, 12 items, default
settings) plus an anchored OEM fit; it now runs inside
`test_installed_irtq_endpoint_baselines`. Largest free-coefficient differences:
a 0.023, b 0.005, c 0.001; response curves 0.002. Every irtQ endpoint passed the
stationarity check. Results:

| Population | Configurations | Status |
|---|---|---|
| Normal | all 8 (every design, 2PL and 3PL) | **Pass** — all checks within tolerance |
| Histogram | all 8 | **Review** — density not converged in either program |

Histogram masses converge far more slowly than item coefficients. The usual
(irtQ) stopping rule checks items only, so both programs stop while node masses
are still moving (≈1e-4 per cycle), at different points; their densities can
then differ (total variation up to 0.04) even though the coefficients agree.
Tightening the item tolerance 100× shrank the gap (TV 0.044 → 0.007) without
fully settling it. When Python's own density is still moving, density-dependent
checks are marked Review with that explanation; coefficient checks stay strict.

**Acceptance policy.** Pass: every check within tolerance and the irtQ endpoint
stationary. Review: density not converged, irtQ stopped early, Python hit its
cycle limit, or the feasible domain differs (positive-a restriction). Fail: any
coefficient, curve or likelihood check outside tolerance. Pass is endpoint
agreement with one reference package, not proof of correctness, global
optimality or instrument validity.

**Bug found by the first real run.** The adapter wrote anchor flags as
`True`/`False`; R's `read.csv` keeps those as text, so `fix.id` became all `NA`
and every anchored design crashed inside irtQ. Flags are now written as 1/0 and
parsed explicitly in R.

A separate supplementary check compared Python histogram rescaling with the
reviewed R source function at commit
`5ad30cd2aa3a7e6a74964df3162e70b0352e7ebc`. This is not a package endpoint
comparison and is never promoted to the primary validation result.

The app has been inspected in a real browser (headless Chrome): all six tabs
render with no exceptions or JavaScript errors, the playback player runs, and
the Validation tab shows a Pass for the default example.

## Verification

```sh
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m unittest test_mmle_engine test_mmle_app -v
```

Tests cover analytic/prior gradients; all 16 design/model/density
combinations; missingness/posterior/count invariants; shared-item pooling and
bank locks; OEM/MEM timing and resume; fixed/free distributions; shared truth
alignment; reproducible generation; stored playback inputs/outputs;
CSV/ZIP round trips; actual playback JS controls; Streamlit run, generate,
priors, comparison and validation flows; the Pass/Review/Fail policy, the
stationarity guard and the anchor-flag file format. The installed irtQ
endpoint matrix (46 tests total, about 70 s) is skipped only when R/irtQ is unavailable.

Method references: [single-group irtQ](https://hwangq.github.io/irtQ/reference/est_irt.html),
[multiple-group irtQ](https://hwangq.github.io/irtQ/reference/est_mg.html), and
[reviewed source](https://github.com/hwangQ/irtQ/tree/5ad30cd2aa3a7e6a74964df3162e70b0352e7ebc).
