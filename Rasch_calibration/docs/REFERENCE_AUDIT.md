# Reference audit

Inspected before implementation (2026-11-01). No reference file was modified.

| Reference | Content / function | Reuse decision | Adaptation in this app |
|---|---|---|---|
| `MMLE_EM.py`, `mmle_engine.py`, `mmle_animation.py`, `mmle_lessons.py` | MMLE-EM Learning Lab for 2PL/3PL: sidebar "Experiment settings", numbered sections, recorded E/M states, Equation / Explanation / Notation lessons, validation status rows | Reuse patterns, not code | Rasch-specific engines (`rasch/`); every iteration records its input state so walkthroughs recompute the real step; equations rewritten for a = 1 |
| `MMLE EM educational app design.md` | Design: likelihood × prior → posterior panels, phase cards, recorded-state replay, "never invent intermediate values", reference agreement as primary validation, three result labels | Reuse | Same three-panel E-step view on the MML-EM page; phase cards on every lab; Reference / Educational / Demonstration badges; reference checks report Unavailable rather than substitute values |
| `mmle-em-layout.png` / `.svg` | Visual hierarchy: light blue-grey sidebar, navy titles, numbered phase row, three chart cards with × and →, Equation/Explanation/Notation switch, validation cards | Reuse | `components/ui.py` CSS (sidebar `#edf1f8`, titles `#17273f`, bordered metrics, `.phase` cards); MML-EM E-step panels with × / → separators; teaching tabs on every page |
| `MHRM_2D.py` | Educational step layout with data-driven "📘 How to read this" boxes, column-matched figure sizing, consistent number formatting | Reuse | `interpret()` boxes with run-specific numbers; uniform table formatting with scientific notation for tiny differences |
| `mirt_validation.py`, `mmle_reference.py` | Rscript bridges with temporary folders, explicit status, project R library | Reuse pattern | `r_bridge/bridge.py` (+ one R script per reference); `RASCH_R_LIBRARY` / shared project library |
| `MMLE_EM_README.md` | Lessons from that project: run with a local Python (iCloud `.venv` stalls), document package versions and executed baselines | Reuse | README run notes; `docs/VALIDATION.md` lists executed comparisons only |
| `.streamlit/config.toml` (repo root) | Theme | Adapted | Own theme in `Rasch_calibration/.streamlit/` matching the layout reference |
| `requirements.txt` (repo root) | Heavy shared dependencies (PyTorch etc.) | Not reused | Separate minimal `requirements.txt` |

No `.Rmd` files exist in the repository. Statistical content was written for the Rasch model rather than
copied: the 2PL/3PL formulas, item priors and multigroup designs of the MMLE app are out of scope here.
