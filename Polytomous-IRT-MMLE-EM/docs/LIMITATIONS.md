# Limitations

- **Platform.** The master prompt targets GPT Sites. This lab is a Streamlit application, which is the framework
  used by the existing MMLE-EM, Rasch and Ability-Scoring labs in this repository. It can be deployed to
  Streamlit Community Cloud (see README). R is not installed there, so live reference runs show as *Unavailable*.
- **Reference material inspected.** The repository's MMLE_EM.py, mmle_engine.py, the Rasch_calibration and
  IRT_ability_scoring labs, their UI conventions, R bridges and tests. The MMLE-EM layout screenshot
  (`mmle-em-layout.png`) was not re-examined pixel by pixel; the visual system is taken from those labs' CSS.
- **One model per calibration.** All items in a run use the selected model. Mixed-format tests are not calibrated
  here (the ability-scoring lab scores them).
- **RSM scale groups** are formed from items with the same number of categories, labelled by `ItemSpec.scale`.
  There is no UI for custom scale groups.
- **Item-parameter priors / penalties** are not implemented. Every estimate is pure MMLE, and the latent ability
  distribution is never mixed up with item-parameter priors.
- **Standard errors** are conditional on the estimated latent distribution (they do not propagate its
  uncertainty). They are computed by numerical differentiation and are skipped above 400 free parameters.
- **Unidimensional only**: no multidimensional models and no latent regression.
- **Estimated priors need a fixed grid.** Gauss–Hermite is available for fixed normal priors only.
- **Free histograms converge slowly** and can become spiky in small samples. This is shown and flagged, not
  smoothed away.
- **Fit statistics** are descriptive: category proportions, standardised residuals, a summed-score X² and
  θ-conditional pseudo-counts. Formal item-fit tests such as S-X² with simulated p-values, and M2, are not
  implemented.
- **Monte Carlo runtime** grows with N × J × cycles. Empirical priors and SEs are the slowest; the page reports
  runtime per replication.
- **Reference coverage** is listed in VALIDATION.md. Combinations outside a package's scope are reported as not
  verified.
