# Rasch Calibration Learning Lab

An interactive Streamlit application that teaches **how** dichotomous Rasch item difficulties are
calibrated — by joint (JMLE), conditional (CMLE) and marginal maximum likelihood with EM (MML-EM) — and
how persons are then scored by MLE and Warm's weighted likelihood (WLE). It also shows how two test forms
are put on one item-bank scale: separate calibration + linking, concurrent, fixed-anchor, and fixed anchor
+ concurrent calibration. Every estimate is computed on the current data; iterations are recorded and can
be replayed step by step; results are checked against the R packages TAM and eRm.

## Run it

```bash
cd Rasch_calibration
pip install -r requirements.txt
streamlit run Rasch_calibration.py        # or: streamlit run app.py
```

In this workspace the project `.venv` sits in iCloud-synced storage, where imports can stall while files
download; any local Python with the packages above works (the suite was run with Streamlit 1.55 and 1.66).

### R reference packages (optional but recommended)

```bash
Rscript install_packages.R                                 # default R library
RASCH_R_LIBRARY=~/my-r-lib Rscript install_packages.R      # or a project library
```

The app looks for packages in `RASCH_R_LIBRARY` first, then
`~/.cache/ai_psychometrics2/R-library` (the shared project library on this machine), then R's defaults.
Without R the Python algorithms still run and every reference check reads **Unavailable**; nothing is
substituted for a missing reference result. Page 8 also shows the recorded local run,
`docs/validation_record.json`, labelled "Recorded, not live" with its date and package versions, so the evidence is
visible on Streamlit Community Cloud. Regenerate it with `python -m r_bridge.record_validation`.

Verified here with R 4.4.3, TAM 4.3-25 and eRm 1.0-10.

## Pages

| # | Page | What you can do |
|---|---|---|
| — | Home | Recommended route, result-type labels, R status |
| 1 | Rasch Model Explorer | ICCs, θ − b, information, shift invariance, numerical example |
| 2 | Data & Simulation | Simulate (N, J, means, SDs, missingness, seed) or upload a CSV; matrix, scores, p-values, extremes |
| 3 | JMLE Lab | Alternating person/item Newton updates; trajectories; per-iteration walkthrough; TAM `tam.jml` |
| 4 | CMLE Lab | Enumeration demo with the ESF denominator; conditional likelihood; L-BFGS-B iterates; eRm `RM` |
| 5 | MML-EM Lab | Likelihood × prior → posterior for any person/cycle; expected counts; M-step Newton trace; TAM `tam.mml` |
| 6 | Person Scoring | MLE vs WLE by raw score and per person (Newton traces), SEs, recovery; TAM `tam.wle` |
| 7 | Estimator Comparison | Same data, all methods: recovery, SEs, runtime, convergence; Monte Carlo with fixed items |
| 8 | Theory & Algorithms | Estimator summary, derivations, notation, references; run every reference check on the current data |
| 9 | Linking & Anchoring | Two-form design with bank anchors, internal common items, group differences and anchor drift; all four linking designs with JMLE, CMLE or MML-EM; anchor stability and displacement; repeated samples; TAM `tam.jml(xsi.fixed=)` / `tam.mml(group=, xsi.fixed=)` and eRm checks per calibration run |

Pages 1–8 share one dataset (`st.session_state`); page 9 simulates its own two-form data. Calibrations are
tied to the dataset identity and the settings that produced them; changing either shows a "settings
changed" notice instead of stale results.

## Tests

```bash
cd Rasch_calibration
python -m pytest tests -q          # 62 tests; R-dependent tests are skipped when R/TAM/eRm are missing
```

Coverage: model probabilities and shift invariance; JMLE score equations, identification, bias factor,
extreme scores; ESFs vs brute force and the generating polynomial, leave-one-out ESFs, conditional
gradient vs finite differences, missing-data conditional likelihood vs enumeration; E-step posteriors vs
direct computation, exposure conservation, M-step root, EM monotonicity; WLE/MLE roots, extreme scores,
sign of Warm's adjustment; agreement with TAM and eRm; every page loads; shared state, stale-settings
notices, uploads, simulation reproducibility and Monte Carlo design. Linking (`tests/test_linking.py`):
fixed anchors shift every estimator's solution exactly; multigroup MML reduces to single-group MML; every
design lands on the bank scale; fixed and fixed + concurrent coincide when the forms share only anchors;
drifted anchors are flagged (median-centred, not fooled by several drifting anchors), displacement points
to them and dropping them removes the bias; TAM/eRm agreement for every calibration run; the page runs all
designs, repeated samples and the drop-anchor action.

## Documentation

- `docs/REFERENCE_AUDIT.md` — what was reused from the MMLE-EM and MHRM apps and how
- `docs/THEORY.md` — equations, identification, standard errors, extreme scores, linking designs
- `docs/IMPLEMENTATION.md` — algorithms, numerical safeguards, R bridge, state handling
- `docs/VALIDATION.md` — executed reference comparisons and Monte Carlo results
