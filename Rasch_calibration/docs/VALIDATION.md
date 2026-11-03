# Validation report

Executed 2026-11-01 on macOS, Python 3.13, NumPy 2.5 / SciPy 1.18, R 4.4.3, **TAM 4.3-25**, **eRm 1.0-10**.
Only results that were actually run are reported.

## 1. Agreement with the reference packages

Maximum absolute differences between the Python educational algorithms and the R references on the same
data (default settings on both sides). Tolerance used by the app: 1e-3 for estimates and SEs, 1e-5 per
person for log-likelihoods.

| Dataset | Extreme persons | JMLE b | JMLE θ | JMLE SE | CMLE b | CMLE SE | CMLE log L_c | MML b | MML σ² | MML SE (complete-data) | MML log L (TAM convention) | WLE θ | WLE SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: N = 500, J = 20, complete | 2 | 1.6e-06 | 6.0e-07 | 1.3e-06 | 8.4e-06 | 8.5e-06 | 5.7e-09 | 5.8e-06 | 1.4e-07 | 1.2e-06 | 9.3e-06 | 1.9e-06 | 1.3e-05 |
| B: N = 500, J = 20, 10 % missing | 4 | 6.3e-06 | 2.0e-05 | 1.8e-06 | 1.1e-05 | 1.0e-05 | 1.3e-08 | 6.8e-06 | 3.3e-07 | 1.7e-06 | 1.2e-05 | 1.1e-05 | 4.2e-05 |
| C: N = 300, J = 8, ability SD 2 | 68 | 1.1e-05 | 2.1e-05 | 1.1e-06 | 7.3e-06 | 5.7e-06 | 1.2e-09 | 4.1e-07 | 6.0e-07 | 4.3e-07 | 2.5e-06 | 4.8e-06 | 1.6e-05 |

JMLE vs `tam.jml`, CMLE vs `eRm::RM`, MML-EM vs `tam.mml`, WLE (with TAM's MML difficulties) vs `tam.wle`.
Every comparison passes on all three datasets, including missing responses and many extreme scores.

### Documented differences (not errors)

1. **MML item SEs.** The app's main MML SE inverts the observed marginal information. TAM's `se.xsi`
   equals the complete-data value $1/\sqrt{\sum_qN_{jq}P(1-P)}$ (matched to 1e-6 above); it ignores the
   uncertainty in each person's θ and was 0.011 (A, B) to 0.044 (C) smaller. Both are displayed.
2. **MML log-likelihood constant.** TAM weights nodes by density × spacing; the app renormalises to sum to
   1. On dataset C (σ ≈ 2, grid ±6) TAM's weights sum to 0.9984 and the two log-likelihoods differ by
   N·log(0.9984) = 0.49; parameter estimates are identical. The app reports both and validates the TAM
   convention.
3. **Extreme-score MLE.** `tam.wle(WLE = FALSE)` moves extreme scores by 0.3; the app reports ±∞.
4. **Iteration paths** differ (TAM damping, L-BFGS-B vs eRm's optimiser); only endpoints are compared.

A machine-readable copy of these checks on seeded versions of datasets A–C (36 of 36 pass) is
`docs/validation_record.json`, written by `python -m r_bridge.record_validation` and shown on page 8 as the
recorded run. The same checks run as automated tests (`tests/test_estimators.py`) and live on the Theory page
("Run all reference comparisons": 12 of 12 checks passed on the default dataset in the browser run).

## 2. Monte Carlo (finite-sample behaviour)

30 replications, N = 500, θ ~ N(0, 1); item difficulties drawn once (b ~ N(0, 1)) and held fixed; estimates
aligned to the true mean in each replication.

| J | Method | RMSE | Empirical SD | Mean reported SE | Slope of estimate on truth |
|---|---|---|---|---|---|
| 20 | CMLE | 0.110 | 0.109 | 0.102 | 1.008 |
| 20 | JMLE (bias corrected) | 0.111 | 0.109 | 0.107 | 1.012 |
| 20 | JMLE (no correction) | 0.129 | 0.115 | 0.107 | 1.066 |
| 20 | MML-EM | 0.109 | 0.109 | 0.114 | 1.008 |
| 8 | CMLE | 0.097 | 0.096 | 0.099 | 1.019 |
| 8 | JMLE (bias corrected) | 0.096 | 0.095 | 0.112 | 1.013 |
| 8 | JMLE (no correction) | 0.167 | 0.109 | 0.112 | 1.158 |
| 8 | MML-EM | 0.097 | 0.096 | 0.115 | 1.017 |

Uncorrected JMLE stretches the scale by about J/(J − 1) (1.14 for J = 8), as theory predicts; TAM's
correction removes most of it. JMLE and MML-EM anchor the scale on persons, so their reported SEs include
origin uncertainty that the per-replication alignment removes from the empirical SD.

A first version of the Monte Carlo redrew item difficulties in every replication, which made the
"empirical SD" measure the spread of the true difficulties (≈ 0.94). It was corrected to hold items fixed;
`tests/test_app.py::test_monte_carlo_holds_items_fixed_and_reports_empirical_sd` guards this.

## 3. Linking designs (page 9)

Default two-form design: 500 + 500 persons, 6 bank anchors, 4 internal common items, 10 unique items per
form, group B ability N(0.5, 1), seed 2026.

**Reference agreement per calibration run** (native scale, before any linking shift; max absolute difference):

| Design | JMLE vs `tam.jml(xsi.fixed=)` (b / θ) | MML-EM vs `tam.mml(group=, xsi.fixed=)` (b / group means / variances) | CMLE vs `eRm::RM` (b) |
|---|---|---|---|
| Separate + link (2 runs) | 3.5e-06 / 9.7e-06 | 3.9e-06 / 0 / 2.5e-07 | 3.4e-05 |
| Concurrent (1 run) | 2.6e-05 / 2.7e-05 | 1.2e-06 / 1.2e-06 / 4.3e-08 | 3.8e-05 |
| Fixed anchor (2 runs) | 2.0e-04 / 1.9e-04 | 3.9e-05 / 3.9e-05 / 2.1e-06 | no R reference |
| Fixed + concurrent (1 run) | 3.9e-04 / 3.9e-04 | 1.7e-05 / 1.7e-05 / 2.2e-06 | no R reference |

All 15 JMLE, 21 MML and 3 CMLE executed checks pass (tolerance 1e-3); anchors are held at the bank values
to 4e-15 in both programs. The larger fixed-anchor JMLE differences come from the two programs' stopping
rules (anchored JMLE has no centring step and converges more slowly). eRm cannot fix item parameters;
fixed-anchor CMLE is instead checked exactly: fixing the anchors at the free solution + 0.7 moves every free
item by 0.7 (to 2e-9). The same shift check holds for JMLE (data without extreme scores) and MML-EM.

**Repeated samples** (items and bank fixed, persons redrawn; MML-EM, 100 replications; error in the
estimated B − A mean difference):

| Scenario | Design | Item RMSE | Difference bias | Difference SD |
|---|---|---|---|---|
| No drift | Separate + link | 0.108 | 0.008 | 0.071 |
| | Concurrent | 0.105 | 0.007 | 0.047 |
| | Fixed anchor | 0.108 | 0.008 | 0.069 |
| | Fixed + concurrent | 0.105 | 0.008 | 0.047 |
| 2 anchors drift +1.0, all used | Separate + link | 0.244 | −0.325 | 0.070 |
| | Concurrent | 0.214 | −0.206 | 0.045 |
| | Fixed anchor | 0.265 | −0.359 | 0.068 |
| | Fixed + concurrent | 0.226 | −0.203 | 0.046 |
| 2 anchors drift +1.0, drifted anchors dropped | Separate + link | 0.113 | 0.006 | 0.089 |
| | Concurrent | 0.108 | 0.006 | 0.054 |
| | Fixed anchor | 0.112 | 0.006 | 0.086 |
| | Fixed + concurrent | 0.107 | 0.008 | 0.054 |

- The stacked designs estimate the group difference about a third more precisely, because the internal
  common items link the groups too. Item RMSE differs little between designs.
- Drift biases every design. The concurrent designs dilute it but do not remove it; dropping the drifted
  anchors removes it everywhere, at a small cost in precision.
- Without drift, CMLE (30 replications) gives the same pattern (difference SD 0.079 / 0.047 / 0.075 / 0.047,
  bias ≤ 0.004). JMLE (50 replications) estimates the group difference from person estimates, which are
  stretched, so it is biased upward by 0.016–0.047.
- Concurrent MML that assumes one population when the groups differ (group B ~ N(1, 1.5²), 50
  replications): item RMSE 0.150 vs 0.109 with group-specific distributions, and group difference
  biased by −0.282 (vs +0.007). The single-population group means are EAP averages, shrunk toward the
  common mean.

Three things were corrected during testing:
1. Fixed and fixed + concurrent first gave identical results because the forms shared only the fixed
   anchors (the likelihood factorises). Internal common items were added to the design; the identity is
   kept as a test.
2. An anchor dropped from the anchor set first stayed one shared item in the stacked designs, so its drift
   still biased them (difference bias −0.21). Dropped anchors are now split into a Form A and a Form B item.
3. The stability check first centred anchor differences on their mean. With two of six anchors drifting,
   that pushed a stable anchor past the flag threshold. It is now centred on the median; a test covers this.

## 4. Automated tests

`python -m pytest tests -q` → **62 passed** (Streamlit 1.66), including 19 linking tests. See README for
the coverage list.

## 5. Browser check

All pages were opened in headless Chrome against a running server; every estimator and reference button
was pressed. On page 9 all four design tabs, the reference checks (21 of 21 MML checks passed), repeated
samples, and a drift scenario (two drifted anchors flagged, dropped, rerun) were exercised. No Streamlit
exceptions or JavaScript errors occurred.

## Limitations

- Dichotomous Rasch only (no 2PL/3PL, polytomous or multidimensional models), by design.
- MML-EM assumes a normal ability distribution; no empirical-histogram option.
- CMLE standard errors use a finite-difference Hessian of the analytic gradient.
- The Monte Carlo runs in the Streamlit process; it has a time limit but no mid-run cancel button other than
  Streamlit's Stop.
- Reference validation needs R with TAM and eRm; without them it reports Unavailable.
- Linking: two forms, simulated data only (no upload of multi-form data); mean/mean linking only (no
  Stocking–Lord or Haebara, which matter for 2PL/3PL rather than Rasch); fixed-anchor CMLE has no R
  reference; group means for JMLE/CMLE are averages of person estimates, not population estimates.
