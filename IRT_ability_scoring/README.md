# IRT Ability Scoring Learning Lab

An interactive Streamlit application that teaches **how** an examinee's ability is estimated from item responses
when the item parameters are known: maximum likelihood (MLE), expected a posteriori (EAP), maximum a posteriori
(MAP), Warm's weighted likelihood (WLE), test-characteristic-curve inversion (TCC) and summed-score EAP (EAPsum).
It covers Rasch/1PL, 2PL, 3PL and 4PL items, the partial credit, generalized partial credit, rating scale and graded
response models, and mixed-format tests. Every number shown is computed on the current data; iterative methods
record every iteration and can be stepped through; quadrature is shown node by node; results are checked against
the R packages **irtQ** (primary), **mirt** and **TAM**.

It is the person-scoring companion of the MMLE-EM Learning Lab and the Rasch Calibration Learning Lab and shares their
layout.

## Run it

```bash
cd IRT_ability_scoring
pip install -r requirements.txt
streamlit run IRT_ability_scoring.py        # or: streamlit run app.py
```

In this workspace the project `.venv` sits in iCloud-synced storage, where imports can stall while files download;
any local Python with the packages above works (the suite was run with Streamlit 1.55 and 1.66).

### R reference packages (optional but recommended)

```bash
Rscript install_packages.R                                   # default R library
SCORING_R_LIBRARY=~/my-r-lib Rscript install_packages.R      # or a project library
```

The app looks for packages in `SCORING_R_LIBRARY` first, then `~/.cache/ai_psychometrics2/R-library` (the shared
project library on this machine), then R's defaults. Without R the Python methods still run and every reference check
reads **Unavailable**; nothing is substituted for a missing reference result. Page 10 (Validation tab) also shows the
recorded local run, `docs/validation_record.json`, labelled "Recorded, not live" with its date and package versions, so
the evidence is visible on Streamlit Community Cloud. Regenerate it with `python -m r_bridge.record_validation`.
Verified with R 4.4.3, irtQ 1.1.0, mirt 1.46.1 and TAM 4.3-25.

## Pages

| # | Page | What you can do |
|---|---|---|
| — | Home | Route, result-type labels, R status |
| 1 | IRT Model Explorer | Category curves, boundary curves (GRM), expected score, information, derivatives at any θ for 8 models; Example D check; TCC and information of the current test |
| 2 | Data & Simulation | Load examples A–D; simulate dichotomous / polytomous / mixed tests (N, θ distribution, item distributions, missingness, D, seed); edit item parameters and the selected examinee's responses |
| 3 | MLE Lab | Newton–Raphson step by step: quadratic approximation, tangent of the score function, item-by-item terms, safeguards, extreme patterns, 3PL multiple maxima, Example B |
| 4 | EAP Lab | Quadrature explorer: prior × likelihood → posterior per node, stage by stage; node inspector; Gauss–Hermite vs rectangular convergence against adaptive integration; prior sensitivity |
| 5 | MAP Lab | Likelihood score + prior score; log-likelihood + log-prior = log-posterior; prior sensitivity; MAP vs MLE |
| 6 | WLE Lab | Warm's adjusted score ℓ′ + J/(2I) step by step; J vs I′; bias term J/(2I²); extreme patterns; Warm (irtQ, TAM) vs mirt's L·√I definition |
| 7 | TCC & EAPsum Lab | TCC root finding with monotonicity check; item expected scores; Lord–Wingersky recursion at any node; summed-score posterior; Rasch MLE = TCC and its 2PL counterexample |
| 8 | Estimator Comparison | All six methods for the selected examinee and the whole dataset; recovery; prior sensitivity; extreme scores; test length; SE vs posterior SD; posterior median, plausible values, PROX, adjusted MLE; R validation of every examinee |
| 9 | Monte Carlo Simulation | Bias, RMSE, MAD with Monte Carlo SEs; conditional bias/RMSE; coverage; reported vs empirical uncertainty; failure and non-finite rates; agreement with a reference estimator; downloads |
| 10 | Theory & Algorithms | Estimator summary, derivations, the executed source code, live numerical examples A–D, the validation suite, references |

The sidebar holds the shared dataset, the selected examinee and the scoring settings (prior, quadrature rule and
nodes, search range, starting value, tolerance). Every page uses them, so methods can be switched without losing the
current responses. Defaults follow irtQ: N(0, 1) prior, 41 Gauss–Hermite nodes, range ±5.

## Tests

```bash
cd IRT_ability_scoring
python -m pytest tests -q          # 125 tests; R-dependent tests are skipped when R / irtQ / mirt / TAM are missing
```

`test_models.py`: probabilities, derivatives, information, J vs I′ (including the GRM identity), special cases.
`test_scoring.py`: examples A–D, independent optimisation / integration / enumeration checks, safeguards, missing
data, 3PL restarts, extras, reproducibility, Monte Carlo. `test_reference.py`: irtQ, mirt and TAM agreement.
`test_app.py`: every page on four datasets, step controls, shared state, simulation, editing, Monte Carlo, reference
buttons.

## Documentation

- `docs/REFERENCE_AUDIT.md` — what was taken from the MMLE-EM app and the reference packages, and how
- `docs/THEORY.md` — models, estimators, derivations, SE vs posterior SD
- `docs/ALGORITHMS.md` — code layout, numerical safeguards, step-through design, state
- `docs/NUMERICAL_EXAMPLES.md` — examples A–D with every intermediate number
- `docs/VALIDATION.md` — executed checks, tolerances, actual differences and explained discrepancies
