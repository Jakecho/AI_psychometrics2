# Reference audit

Inspected before implementation (2026-11-02). No reference file was modified.

| Reference | What it contains | What was reused | What was adapted |
|---|---|---|---|
| `MMLE_EM.py`, `mmle_engine.py`, `mmle_animation.py`, `mmle_lessons.py` | MMLE-EM Learning Lab (2PL/3PL item calibration): "Experiment settings" sidebar, numbered sections, recorded E/M states replayed step by step, Equation / Explanation / Notation lessons | Layout and interaction conventions: sidebar header, numbered phase cards, Previous / Next / Run controls over *recorded* states, three-tab lessons | Content replaced by person scoring with fixed items; the iteration explorer steps through recorded Newton / root-finding iterations (MLE, MAP, WLE, TCC); quadrature is explored node by node, not as fake iterations |
| `MMLE EM educational app design.md` | Design principles: likelihood × prior → posterior panels, never invent intermediate values, reference agreement as primary validation, three result labels | Reused directly | EAP page uses the same "likelihood × prior → posterior" three-panel view on quadrature nodes; Reference / Educational / Demonstration badges on every page |
| `mmle-em-layout.png`, `.svg` | Visual hierarchy: blue-grey sidebar, navy titles, phase row, three chart cards with × and →, validation cards | Colours, CSS, phase cards, × / → separators | `components/ui.py` (CSS shared with the Rasch Calibration Lab) |
| `mmle_irtq.R`, `mmle_reference.py` | Rscript bridge to irtQ (installed package only, temporary folders, explicit status) | irtQ as the **primary** reference, temporary-folder JSON/CSV interchange, "never substitute emulated code" | `r_bridge/irtq_scores.R` builds the irtQ item data frame directly (1PLM/2PLM/3PLM/GRM/GPCM) and calls `est_score()` for ML, WL, MAP, EAP, EAP.SUM, INV.TCC |
| `mirt_validation.py`, `mirt_validation.R` | mirt bridge for MIRT models | Secondary reference pattern | `r_bridge/mirt_scores.R`: fixed-parameter model via `pars = "values"`, `est = FALSE`, `TOL = NaN`; `fscores()` |
| `Rasch_calibration/` (sister app built from the same reference) | `components/ui.py`, `r_bridge/bridge.py`, AppTest suite, docs layout | Shared UI helpers, bridge structure, test style, documentation set | Ability-scoring state: shared dataset + **selected examinee** + scoring settings in the sidebar |
| `MHRM_2D.py` | Data-driven "📘 How to read this" boxes | Reused | Every page ends with an interpretation that quotes the current numbers |
| `MMLE_EM_README.md` | Lessons: run with a local Python (iCloud `.venv` stalls), record executed baselines | Reused | README run notes; `docs/VALIDATION.md` lists executed comparisons only |

No `.Rmd` files exist in the repository; the Markdown design document above served as the mathematical-style
reference.

Deviations from the suggested layout in the specification: PCM, GPCM and RSM share one kernel
(`irt_models/partial_credit.py`) because they are the same divide-by-total formula with different constraints;
MAP lives in `scoring/map_estimate.py` (a module called `map` would shadow Python's built-in). The extra
`scoring/solvers.py` holds the two safeguarded solvers shared by MLE/MAP and WLE/TCC.

Inspection findings that shaped the implementation:

- irtQ `est_score()` (v1.1.0): ML/WL/MAP use Fisher scoring (step = score / information, capped at 1) from a grid
  start, tolerance 10⁻⁴, θ clipped to `range = c(−5, 5)` with SE 99.9999 at the bound; EAP and EAP.SUM use
  `statmod::gauss.quad.prob(41, "norm")` (Gauss–Hermite); WL solves ℓ′ + J/(2I) = 0 with J = Σ P′P″/P (Warm's
  general form); INV.TCC brackets on a 0.01 grid and bisects; EAP.SUM / INV.TCC replace missing responses with 0.
  The app's defaults follow irtQ (41 Gauss–Hermite nodes, range ±5).
- mirt `fscores()` (v1.46.1): "WLE" maximises ℓ + ½ log I (equal to Warm only when J = I′); ML / MAP / WLE SEs come
  from `nlm`'s numerical Hessian (observed information); EAP uses an equally spaced grid on `theta_lim`.
- TAM `tam.wle()`: Warm's WLE with SE 1/√I; its PCM `xsi` are the step difficulties δ_v.
