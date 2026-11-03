# Reference validation

**Primary references:** TAM for the Rasch family (PCM, RSM) and irtQ for general polytomous calibration (PCM as
GPCM with fixed slope, GPCM, GRM). **Supplementary:** mirt. Each Rscript receives the same responses (blank =
missing), the same equally spaced grid (`Quadrature = c(Q, bound)` in irtQ, `control$nodes` in TAM,
`quadpts` + `theta_lim` in mirt) and the same anchors. Its estimates are converted back to the lab's canonical
parameters.

| Package | Calls used |
|---|---|
| TAM | `tam.mml(irtmodel = "PCM" / "RSM")`; concurrent: `group` plus focal dummies in `Y`; FIPC/MG-FIPC: `xsi.fixed` with `beta.fixed = FALSE`; supplementary GPCM: `tam.mml.2pl(irtmodel = "GPCM")` (a = alpha, δ_k = beta + tau_k) |
| irtQ | `est_irt()` (single; `fipc = TRUE`, `fipc.method`, `fix.id`; `EmpHist`; `weights`); `est_mg()` (concurrent with `free.group = 2:G`; MG-FIPC with `fipc = TRUE`, `free.group = 1:G`); results via `getirt(..., what = "par.est" / "group.par" / "weights")`, with item metadata built by `shape_df()` |
| mirt | `mirt(itemtype = "Rasch" / "gpcm" / "graded")`, `multipleGroup(invariance = c(items, "free_means", "free_var"))`, `dentype = "empiricalhist_Woods"` |

The bridge (`r_reference/bridge.py`) reports **Not applicable** with a reason for combinations a package cannot
reproduce. Examples: irtQ has no RSM; irtQ fixes the reference variance where Rasch identification estimates it;
TAM's empirical-prior interface differs. When R or a package is missing, it reports **Unavailable**. Results are
never filled in when a package was not run.

## Executed results

See [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md). Regenerate it with `python -m r_reference.run_validation`; it
records the software versions, settings, grid and tolerances. `tests/test_reference.py` asserts agreement for the
same cases:

- **Agreement to about 1e-5** in item parameters, group moments and ℓ: TAM for PCM, RSM, RSM concurrent, PCM FIPC,
  RSM MG-FIPC and GPCM (tam.mml.2pl); irtQ for GPCM with a fixed normal, GPCM with a fixed empirical prior (`weights`), GRM concurrent, GPCM FIPC (MEM and OEM) and GRM MG-FIPC;
  mirt for GPCM, PCM and GRM concurrent.
- **irtQ EmpHist** (Woods) vs the lab's Woods option: items within 7e-4 and ℓ within 0.02. Free histograms keep
  creeping; irtQ stops on item changes only.

## Documented differences (explained, not bugs)

1. **Exact constrained histogram vs Woods**: these are different estimators. The lab's ℓ is higher and items differ
   by up to 4e-3.
2. **Free histogram in FIPC**: irtQ stopped after 169 cycles, based on item changes alone. Run longer, the free
   histogram grows spikier and ℓ keeps rising, so the histograms differ (TV distance 0.28) even though the items
   agree to 2e-3.
3. **TAM on a very coarse grid**: TAM's quadrature weights are φ(θ_q)·Δθ without renormalisation (they sum to 0.50
   with 7 nodes on ±6). With those weights the lab reproduces TAM's ℓ exactly. From about 15 nodes the two
   conventions coincide.
4. **NWU–OEM reporting**: irtQ reports the distribution implied by the single E-step. The lab reports the same
   distribution (as `implied_weights`) but keeps the prior used for estimation unchanged; the comparison uses the
   implied distribution.

## Not verified here

TAM fixed-normal multigroup runs, TAM empirical priors and mirt fixed-item designs are not compared, for the reasons
given in the scope table on page 12. irtQ FIPC with a fixed empirical prior is out of scope because irtQ estimates
the new group's distribution in FIPC.
