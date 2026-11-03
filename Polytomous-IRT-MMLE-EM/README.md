# Polytomous IRT MMLE–EM Learning Lab

A working educational Streamlit app that shows how polytomous item parameters are estimated by marginal maximum
likelihood with the EM algorithm. Every number shown comes from the Python engine in `core/`.

- **Models:** PCM, RSM (shared steps estimated jointly), GPCM, GRM
- **Designs:** single-group, concurrent, fixed-item (FIPC: NWU–OEM, MWU–MEM), concurrent fixed-item (MG-FIPC)
- **Ability priors:** fixed normal, estimated normal, fixed empirical, estimated empirical (exact constrained
  M-step; Woods rescaling as an irtQ-compatible option)
- **References:** TAM (Rasch), irtQ (general polytomous), mirt (supplementary), run through Rscript

## Run

```bash
cd Polytomous-IRT-MMLE-EM
python -m pip install -r requirements.txt
python -m streamlit run Polytomous_MMLE_EM.py      # or: streamlit run app.py
```

Reference validation is optional. Install R and run `Rscript install_packages.R`. The bridge looks for packages in
the default R library and in `$POLY_R_LIBRARY` (falling back to `~/.cache/ai_psychometrics2/R-library`, the
library shared with the other labs). Without R, the app works and reference results show as *Unavailable*.

## Streamlit Community Cloud

Repository `Jakecho/AI_psychometrics2`, branch `main`, main file path
`Polytomous-IRT-MMLE-EM/Polytomous_MMLE_EM.py`. Cloud installs this folder's `requirements.txt`. R is not
available there, so live TAM/irtQ/mirt runs on the Reference Validation page report *Unavailable*. The page still
shows the recorded local run (`docs/validation_record.json`, written with `docs/VALIDATION_RESULTS.md`), labelled
with its date and package versions.

## Pages

1 Overview · 2 Model Explorer · 3 Prior Distribution Explorer · 4 Data Simulation · 5 Quadrature Laboratory ·
6 EM Walkthrough (N = 5, J = 3, Q = 3; 13 steps) · 7 Calibration Designs (identification, run, comparison
laboratory) · 8 Results (inspect any iteration) · 9 Parameter Recovery · 10 Model Fit · 11 Drift Analysis ·
12 Reference Validation. A sidebar switch selects **Guided** mode (steps and knowledge checks) or **Exploration**
mode (all controls open).

## Layout

```
core/         models · quadrature · priors · em (E-step, counts, M-steps, designs, identification, SEs)
              simulation · diagnostics (fit, recovery, Monte Carlo, drift) · walkthrough · data
components/   ui.py (shared state, sidebar, four-view tabs, guided boxes, tables, plots)
pages/        2–12 (page 1 is Polytomous_MMLE_EM.py)
r_reference/  bridge.py · tam_validation.R · irtq_validation.R · mirt_validation.R · cases.py · run_validation.py
docs/         THEORY · CALIBRATION_DESIGNS · PRIOR_DISTRIBUTIONS · NUMERICAL_EXAMPLES (generated) · VALIDATION ·
              VALIDATION_RESULTS (generated) · LIMITATIONS
tests/        models · priors/quadrature · EM/designs/edge cases · reference (R) · app (AppTest) · docs
```

## Tests

```bash
python -m pytest tests -q
```

The reference tests are skipped when R or a package is missing. `docs/NUMERICAL_EXAMPLES.md` and
`docs/VALIDATION_RESULTS.md` are regenerated with `python docs/make_numerical_examples.py` and
`python -m r_reference.run_validation`.

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what is not covered.
