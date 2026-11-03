# Validation

Executed 2026-11-02 on R 4.4.3 with **irtQ 1.1.0** (primary reference), **mirt 1.46.1** and **TAM 4.3-25**; Python
3.13, NumPy 2, Streamlit 1.66.1 and 1.55.0. A method is not considered validated because its optimiser converges: each
method is checked against independent calculations (section 1) and against the reference packages (section 2).

## 1. Independent mathematical checks (`tests/test_models.py`, `tests/test_scoring.py`)

| Check | How | Tolerance / result |
|---|---|---|
| Category probabilities sum to 1, all positive | 8 models incl. D = 1.702, θ ∈ [−12, 12] | 10⁻¹² |
| P′, P″ correct | central finite differences | 10⁻⁷ / 10⁻⁴ |
| Σ_k P′ = Σ_k P″ = 0 | every model | 10⁻¹² |
| I(θ) = E[−ℓ″] | every model | 10⁻¹⁰ |
| J = I′ for Rasch, 2PL, PCM, GPCM, RSM **and GRM**; J ≠ I′ for 3PL, 4PL | finite-difference I′ | 10⁻⁷ (difference > 10⁻³ for 3PL/4PL) |
| ℓ′, ℓ″, T′, I′ of mixed-format patterns | finite differences | 10⁻⁶ |
| Special cases (GPCM/GRM with 2 categories = 2PL; RSM = PCM with b + τ; 4PL with c = 0, d = 1 = 2PL) | identity | 10⁻¹² |
| Example A MLE | `scipy.optimize.minimize_scalar` on a separately coded log-likelihood | 10⁻⁶ |
| Example B MLE ±0.4196, TCC 0 | specification | 5·10⁻⁵ / 10⁻¹⁰ |
| Example D GRM probabilities | specification (0.26894, 0.46212, 0.26894) | 5·10⁻⁶ |
| Recorded Newton history | θ⁽ᵗ⁺¹⁾ of row t = θ⁽ᵗ⁾ of row t+1; row values = `pattern_terms` at θ⁽ᵗ⁾ | exact |
| EAP = Σθ·w·L / Σw·L; Σπ = 1 | node table | 10⁻¹⁴ |
| EAP converges to adaptive quadrature (`scipy.integrate.quad`, rel. tol 10⁻¹²) | 81 GH nodes, PCM example; 101 nodes, 3PL+GRM | 10⁻⁹ / 10⁻⁷ |
| Log-domain stability | 1,500-item test (likelihood ≈ 10⁻⁴⁰⁰) | finite EAP and PSD |
| MAP → MLE as σ → ∞; MAP → μ as σ → 0; ℓ′(θ̂) = (θ̂ − μ)/σ² | Example D | 10⁻⁴ / 10⁻² / 10⁻⁷ |
| WLE solves ℓ′ + J/(2I) = 0 | Examples A, C, 3PL+GRM | 10⁻⁸ |
| WLE between 0 and MLE (Rasch); finite for extreme patterns | Example A | exact |
| TCC: T(θ̂) = r; Rasch-family MLE = TCC | Rasch + PCM simulation | 10⁻⁹ / 10⁻⁷ |
| Lord–Wingersky recursion | brute-force enumeration of all 3·3·4 PCM patterns | 10⁻¹⁴ |
| Missing responses | scoring with NaN = scoring the reduced item set, all 6 methods | 10⁻¹⁰ |
| 3PL multiple maxima | a pattern whose run from θ⁽⁰⁾ = −2 reaches the bound; the restart finds the grid maximum | 10⁻³ of a 20,001-point grid argmax |
| Reproducibility | same seed ⇒ identical data, identical Monte Carlo results | exact |

## 2. Reference packages (`tests/test_reference.py`, page 8 "R validation", page 10 "Validation")

Matching rules: same item parameters (converted to each package's parameterisation), same D, prior, quadrature
nodes and search range. irtQ tolerance set to 10⁻¹⁰ (its default is 10⁻⁴). mirt has no Gauss–Hermite option in
`fscores`, so it is compared on a rectangular grid of the same size on ±6. Patterns whose MLE / TCC is not finite are
counted, not compared numerically (irtQ returns the range bound with SE 99.9999; mirt returns ±∞; TAM adjusts by
0.3; irtQ's INV.TCC interpolates toward ±7).

Tolerances: irtQ 10⁻⁵ (θ and SE); mirt 2·10⁻³ (θ), 2·10⁻² (SE) because it uses `nlm` and a numerical Hessian; TAM
10⁻³; EAP and EAPsum 10⁻⁶ for every package. Summary of 96 comparisons over 8 datasets:

- **85 Pass.** Largest |Δθ| among passes: irtQ 1.2·10⁻⁹, mirt 4.1·10⁻⁶, TAM 2.4·10⁻⁵ (TAM's own WLE convergence).
  EAP and EAPsum agree to ≤ 10⁻¹⁴ with irtQ and mirt.
- **1 Explained difference.** irtQ INV.TCC, Example B, |Δθ| = 0.01: the true root is exactly θ = 0, which lies on
  irtQ's 0.01 search grid; irtQ's bracketing then returns the neighbouring grid point (−0.01). Moving b by 0.003
  removes the difference. This is an irtQ edge case, not a Python discrepancy.
- **10 Information.** (i) Warm's WLE versus mirt's "WLE": mirt maximises ℓ + ½ log I, i.e. solves ℓ′ + I′/(2I) = 0.
  The difference is ≤ 2.5·10⁻⁶ whenever J = I′ (Rasch, 2PL, PCM, GPCM, RSM, GRM) and 0.13–0.15 logits on tests with
  3PL / 4PL items. The app's L·√I mode reproduces mirt (Pass rows). (ii) irtQ EAP.SUM and INV.TCC are not run on data
  with missing responses because irtQ replaces missing responses with 0; the app scores them on the observed items.
- irtQ has no 4PL; TAM is limited here to Rasch-family items: these combinations report *Unavailable*.

SE definitions compared: MLE 1/√I (irtQ) and 1/√(−ℓ″) (mirt); MAP [I + 1/σ²]^−½ (irtQ) and [−ℓ″ + 1/σ²]^−½ (mirt); WLE
1/√I (irtQ, TAM); EAP / EAPsum posterior SD. Not compared: mirt's WLE SE (numerical Hessian of ℓ + ½ log I) and irtQ's
INV.TCC SE (a summed-score spread, not the delta-method SE the app reports).

### Executed comparison table

A machine-readable copy of this comparison (same counts: 85 Pass, 10 Information, 1 Explained difference, plus
16 Unavailable combinations) is `docs/validation_record.json`, written by `python -m r_bridge.record_validation`
and shown on page 10 as the recorded run.

| Dataset | Package | Method | Python counterpart | n | max |Δθ| | max |ΔSE| | Result |
|---|---|---|---|---|---|---|---|
| A · Rasch (8 patterns) | irtQ | MLE | MLE | 6 | 2.2e-16 | 2.2e-14 | Pass |
| A · Rasch (8 patterns) | irtQ | WLE | WLE | 8 | 1.9e-11 | 3.0e-11 | Pass |
| A · Rasch (8 patterns) | irtQ | MAP | MAP | 8 | 1.1e-16 | 3.5e-12 | Pass |
| A · Rasch (8 patterns) | irtQ | EAP | EAP | 8 | 8.9e-16 | 6.7e-16 | Pass |
| A · Rasch (8 patterns) | irtQ | EAPsum | EAPsum | 8 | 1.0e-15 | 7.8e-16 | Pass |
| A · Rasch (8 patterns) | irtQ | TCC | TCC | 6 | 1.1e-11 | — | Pass |
| A · Rasch (8 patterns) | mirt | MLE | MLE (observed-information SE) | 6 | 6.7e-07 | 1.6e-05 | Pass |
| A · Rasch (8 patterns) | mirt | WLE | L·√I mode (mirt's definition) | 8 | 1.1e-06 | — | Pass |
| A · Rasch (8 patterns) | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 8 | 1.1e-06 | — | Information |
| A · Rasch (8 patterns) | mirt | MAP | MAP (observed-information SE) | 8 | 1.8e-06 | 4.1e-06 | Pass |
| A · Rasch (8 patterns) | mirt | EAP | EAP | 8 | 5.6e-16 | 2.2e-16 | Pass |
| A · Rasch (8 patterns) | mirt | EAPsum | EAPsum | 8 | 5.6e-16 | 3.3e-16 | Pass |
| A · Rasch (8 patterns) | TAM | MLE | MLE | 6 | 2.2e-16 | 2.2e-15 | Pass |
| A · Rasch (8 patterns) | TAM | WLE | WLE | 8 | 2.4e-05 | 3.7e-05 | Pass |
| B · 2PL (4) | irtQ | MLE | MLE | 2 | 5.6e-17 | 5.6e-16 | Pass |
| B · 2PL (4) | irtQ | WLE | WLE | 4 | 3.9e-11 | 3.5e-11 | Pass |
| B · 2PL (4) | irtQ | MAP | MAP | 4 | 9.3e-15 | 1.1e-15 | Pass |
| B · 2PL (4) | irtQ | EAP | EAP | 4 | 5.6e-16 | 1.1e-16 | Pass |
| B · 2PL (4) | irtQ | EAPsum | EAPsum | 4 | 4.4e-16 | 2.2e-16 | Pass |
| B · 2PL (4) | irtQ | TCC | TCC | 2 | 1.0e-02 | — | Explained difference |
| B · 2PL (4) | mirt | MLE | MLE (observed-information SE) | 2 | 5.0e-07 | 3.2e-05 | Pass |
| B · 2PL (4) | mirt | WLE | L·√I mode (mirt's definition) | 4 | 7.6e-07 | — | Pass |
| B · 2PL (4) | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 4 | 7.6e-07 | — | Information |
| B · 2PL (4) | mirt | MAP | MAP (observed-information SE) | 4 | 7.3e-07 | 1.7e-05 | Pass |
| B · 2PL (4) | mirt | EAP | EAP | 4 | 5.0e-16 | 2.2e-16 | Pass |
| B · 2PL (4) | mirt | EAPsum | EAPsum | 4 | 3.3e-16 | 4.4e-16 | Pass |
| C · PCM (8) | irtQ | MLE | MLE | 6 | 1.6e-13 | 3.6e-14 | Pass |
| C · PCM (8) | irtQ | WLE | WLE | 8 | 1.4e-11 | 3.8e-11 | Pass |
| C · PCM (8) | irtQ | MAP | MAP | 8 | 4.4e-15 | 3.5e-13 | Pass |
| C · PCM (8) | irtQ | EAP | EAP | 8 | 3.6e-15 | 7.8e-16 | Pass |
| C · PCM (8) | irtQ | EAPsum | EAPsum | 8 | 2.9e-15 | 8.9e-16 | Pass |
| C · PCM (8) | irtQ | TCC | TCC | 6 | 6.5e-10 | — | Pass |
| C · PCM (8) | mirt | MLE | MLE (observed-information SE) | 6 | 7.8e-07 | 4.2e-05 | Pass |
| C · PCM (8) | mirt | WLE | L·√I mode (mirt's definition) | 8 | 1.4e-06 | — | Pass |
| C · PCM (8) | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 8 | 1.4e-06 | — | Information |
| C · PCM (8) | mirt | MAP | MAP (observed-information SE) | 8 | 1.1e-06 | 1.6e-05 | Pass |
| C · PCM (8) | mirt | EAP | EAP | 8 | 2.4e-15 | 6.7e-16 | Pass |
| C · PCM (8) | mirt | EAPsum | EAPsum | 8 | 2.0e-15 | 1.4e-15 | Pass |
| C · PCM (8) | TAM | MLE | MLE | 6 | 1.6e-13 | 3.5e-14 | Pass |
| C · PCM (8) | TAM | WLE | WLE | 8 | 1.6e-05 | 2.4e-05 | Pass |
| D · GRM (6) | irtQ | MLE | MLE | 4 | 1.3e-11 | 1.7e-11 | Pass |
| D · GRM (6) | irtQ | WLE | WLE | 6 | 2.3e-11 | 3.2e-11 | Pass |
| D · GRM (6) | irtQ | MAP | MAP | 6 | 5.0e-12 | 1.4e-12 | Pass |
| D · GRM (6) | irtQ | EAP | EAP | 6 | 6.0e-15 | 1.3e-15 | Pass |
| D · GRM (6) | irtQ | EAPsum | EAPsum | 6 | 5.3e-15 | 3.3e-16 | Pass |
| D · GRM (6) | irtQ | TCC | TCC | 4 | 1.2e-09 | — | Pass |
| D · GRM (6) | mirt | MLE | MLE (observed-information SE) | 4 | 1.1e-06 | 4.5e-05 | Pass |
| D · GRM (6) | mirt | WLE | L·√I mode (mirt's definition) | 6 | 1.6e-06 | — | Pass |
| D · GRM (6) | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 6 | 1.6e-06 | — | Information |
| D · GRM (6) | mirt | MAP | MAP (observed-information SE) | 6 | 1.3e-06 | 1.2e-05 | Pass |
| D · GRM (6) | mirt | EAP | EAP | 6 | 3.1e-15 | 5.6e-16 | Pass |
| D · GRM (6) | mirt | EAPsum | EAPsum | 6 | 1.3e-15 | 4.4e-16 | Pass |
| 3PL+GRM, N=200 | irtQ | MLE | MLE | 198 | 1.3e-10 | 4.8e-11 | Pass |
| 3PL+GRM, N=200 | irtQ | WLE | WLE | 200 | 8.7e-11 | 3.7e-11 | Pass |
| 3PL+GRM, N=200 | irtQ | MAP | MAP | 200 | 2.6e-11 | 1.2e-11 | Pass |
| 3PL+GRM, N=200 | irtQ | EAP | EAP | 200 | 7.5e-15 | 3.6e-15 | Pass |
| 3PL+GRM, N=200 | irtQ | EAPsum | EAPsum | 200 | 1.4e-11 | 4.6e-12 | Pass |
| 3PL+GRM, N=200 | irtQ | TCC | TCC | 198 | 3.5e-11 | — | Pass |
| 3PL+GRM, N=200 | mirt | MLE | MLE (observed-information SE) | 198 | 4.1e-06 | 1.4e-04 | Pass |
| 3PL+GRM, N=200 | mirt | WLE | L·√I mode (mirt's definition) | 200 | 3.6e-06 | — | Pass |
| 3PL+GRM, N=200 | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 200 | 1.3e-01 | — | Information |
| 3PL+GRM, N=200 | mirt | MAP | MAP (observed-information SE) | 200 | 3.0e-06 | 2.8e-05 | Pass |
| 3PL+GRM, N=200 | mirt | EAP | EAP | 200 | 5.8e-15 | 2.6e-15 | Pass |
| 3PL+GRM, N=200 | mirt | EAPsum | EAPsum | 200 | 4.4e-15 | 7.8e-16 | Pass |
| 2PL+GPCM(5 cat), N=200 | irtQ | MLE | MLE | 199 | 3.4e-13 | 4.3e-10 | Pass |
| 2PL+GPCM(5 cat), N=200 | irtQ | WLE | WLE | 200 | 1.7e-10 | 1.2e-10 | Pass |
| 2PL+GPCM(5 cat), N=200 | irtQ | MAP | MAP | 200 | 1.6e-13 | 7.0e-12 | Pass |
| 2PL+GPCM(5 cat), N=200 | irtQ | EAP | EAP | 200 | 6.2e-15 | 5.4e-15 | Pass |
| 2PL+GPCM(5 cat), N=200 | irtQ | EAPsum | EAPsum | 200 | 1.2e-13 | 3.0e-13 | Pass |
| 2PL+GPCM(5 cat), N=200 | irtQ | TCC | TCC | 199 | 4.2e-10 | — | Pass |
| 2PL+GPCM(5 cat), N=200 | mirt | MLE | MLE (observed-information SE) | 199 | 2.3e-06 | 9.4e-05 | Pass |
| 2PL+GPCM(5 cat), N=200 | mirt | WLE | L·√I mode (mirt's definition) | 200 | 2.5e-06 | — | Pass |
| 2PL+GPCM(5 cat), N=200 | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 200 | 2.5e-06 | — | Information |
| 2PL+GPCM(5 cat), N=200 | mirt | MAP | MAP (observed-information SE) | 200 | 1.8e-06 | 2.7e-05 | Pass |
| 2PL+GPCM(5 cat), N=200 | mirt | EAP | EAP | 200 | 5.8e-15 | 4.4e-15 | Pass |
| 2PL+GPCM(5 cat), N=200 | mirt | EAPsum | EAPsum | 200 | 3.3e-15 | 1.9e-15 | Pass |
| 4PL+RSM, N=200 | mirt | MLE | MLE (observed-information SE) | 194 | 3.5e-06 | 2.8e-04 | Pass |
| 4PL+RSM, N=200 | mirt | WLE | L·√I mode (mirt's definition) | 200 | 3.3e-06 | — | Pass |
| 4PL+RSM, N=200 | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 200 | 1.5e-01 | — | Information |
| 4PL+RSM, N=200 | mirt | MAP | MAP (observed-information SE) | 200 | 2.7e-06 | 2.3e-05 | Pass |
| 4PL+RSM, N=200 | mirt | EAP | EAP | 200 | 5.8e-15 | 3.1e-15 | Pass |
| 4PL+RSM, N=200 | mirt | EAPsum | EAPsum | 200 | 3.8e-15 | 3.3e-15 | Pass |
| Rasch+PCM, 10% missing, N=200 | irtQ | MLE | MLE | 199 | 2.5e-13 | 4.0e-11 | Pass |
| Rasch+PCM, 10% missing, N=200 | irtQ | WLE | WLE | 200 | 5.7e-12 | 2.3e-11 | Pass |
| Rasch+PCM, 10% missing, N=200 | irtQ | MAP | MAP | 200 | 1.2e-13 | 6.3e-12 | Pass |
| Rasch+PCM, 10% missing, N=200 | irtQ | EAP | EAP | 200 | 5.6e-15 | 3.3e-15 | Pass |
| Rasch+PCM, 10% missing, N=200 | irtQ | EAPsum | — | 0 | — | — | Information |
| Rasch+PCM, 10% missing, N=200 | irtQ | TCC | — | 0 | — | — | Information |
| Rasch+PCM, 10% missing, N=200 | mirt | MLE | MLE (observed-information SE) | 199 | 2.3e-06 | 1.4e-04 | Pass |
| Rasch+PCM, 10% missing, N=200 | mirt | WLE | L·√I mode (mirt's definition) | 200 | 2.3e-06 | — | Pass |
| Rasch+PCM, 10% missing, N=200 | mirt | WLE | Warm WLE (ℓ′ + J/2I) — information only | 200 | 2.3e-06 | — | Information |
| Rasch+PCM, 10% missing, N=200 | mirt | MAP | MAP (observed-information SE) | 200 | 3.0e-06 | 3.1e-05 | Pass |
| Rasch+PCM, 10% missing, N=200 | mirt | EAP | EAP | 200 | 5.1e-15 | 2.4e-15 | Pass |
| Rasch+PCM, 10% missing, N=200 | mirt | EAPsum | EAPsum | 42 | 4.4e-15 | 2.3e-15 | Pass |
| Rasch+PCM, 10% missing, N=200 | TAM | MLE | MLE | 199 | 2.5e-13 | 6.4e-14 | Pass |
| Rasch+PCM, 10% missing, N=200 | TAM | WLE | WLE | 200 | 5.2e-06 | 4.6e-05 | Pass |

## 3. Application tests (`tests/test_app.py`)

Every page and both entry points load without exceptions on four datasets (default 2PL + GPCM, Example B, 3PL + GRM
with 15% missing, 4PL + RSM). Step controls (Initialize, Previous, Next, Run to convergence, Reset) on all five
explorers; displayed iteration values equal the recorded solver values; extreme-pattern warning; the selected
examinee and sidebar settings carry across pages; simulation and pattern editing; Monte Carlo run, downloads and the
stale-results notice; reference buttons on page 6 and page 8.

Full suite: **125 passed** on Streamlit 1.66.1 and on 1.55.0 (R tests included). Browser check (Chrome, all 11
pages, explorers stepped, Monte Carlo run): 0 exceptions, 0 application console errors. Opening a page by its
direct URL produces two 404s for Streamlit's own `_stcore/host-config` and `_stcore/health` probes; these come from
Streamlit, not the app.

## 4. Known limitations

- The Python 4PL results have no irtQ reference (irtQ has no 4PL); they are checked against mirt and the independent
  tests only.
- The TCC SE is a delta-method approximation; WLE and MLE SEs are asymptotic.
- 3PL/4PL global-maximum checks use an 801-point grid on the search range; maxima narrower than the grid spacing
  (0.0125 logits at ±5) could be missed.
- Plausible values are drawn from a fine-grid posterior (801 points, uniform within cells): an approximation adequate
  for teaching, not a replacement for a full population model with background variables.
