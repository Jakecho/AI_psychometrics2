# Ability prior distributions

These are latent ability distributions g(θ) inside the marginal likelihood. They are **not** Bayesian priors on item
parameters. The lab estimates items by pure MMLE, with no item-parameter penalties.

| Setting | Representation | Updated? | M-step |
|---|---|---|---|
| A · Fixed normal | Gauss–Hermite nodes μ + σx_q, or normalised φ on a fixed grid | no | — |
| B · Estimated normal | π_q(μ, σ) ∝ φ((θ_q − μ)/σ) on a fixed grid | free μ, σ | exact: maximise Σ n_q log π_q(μ, σ) (L-BFGS-B); optional "posterior moments" |
| C · Fixed empirical | supplied masses on the grid | no | — |
| D · Estimated empirical | free masses on a fixed grid | yes | exact (see below); optional Woods rescaling |

Gauss–Hermite nodes move with μ and σ. With them, an estimated prior would need the item likelihood at new nodes
inside the prior M-step, so the lab allows Gauss–Hermite only with a fixed normal and rejects the other
combinations, with an explanation.

## Estimated empirical prior: exact constrained M-step

The prior part of the expected complete-data objective is Q_prior(π) = Σ_q n_q log π_q, with n_q = Σ_i w_iq.

- **Unconstrained** (anchored designs, focal groups): π_q = n_q/N, the fractional-count average of the posteriors.
- **Constrained** (a reference distribution without anchors): maximise Q_prior subject to Σπ = 1, Σπθ = μ0, and,
  for GPCM/GRM, Σπ(θ − μ0)² = σ0². The Lagrange conditions give

  π_q = (n_q/N) / (1 + λ1(θ_q − μ0) + λ2[(θ_q − μ0)² − σ0²]),

  where λ minimises the convex dual −Σ_q (n_q/N) log(1 + λᵀg_q). The lab solves this with Newton steps and
  backtracking that keeps every denominator positive. This is the exact maximiser (tested against SLSQP), so EM
  stays monotone and the constraints hold to about 1e-12 in every cycle. The masses are **not** standardised
  afterwards while item parameters stay fixed. If the requested moments are impossible on the grid, the run stops
  with an error.

**Comparison with irtQ's Woods (2007) EmpHist.** irtQ takes n/N and re-standardises the histogram to the target
moments by interpolating onto a shifted and stretched grid, with exponential tails. That step only approximately
satisfies the moments and is not guaranteed to increase ℓ. The lab includes a port of it ("Woods (irtQ)" in the
sidebar) for like-for-like comparisons. Executed difference on the same data: item parameters within 4e-3, and
the lab's ℓ higher by 0.045 (see VALIDATION_RESULTS.md).

**Instability.** The unpenalised histogram MLE tends towards a few support points. As EM keeps going, the
histogram becomes spiky, with nearly empty bins, while ℓ creeps up slowly. The lab therefore judges convergence
on the masses as well as the items (default prior-mass tolerance 1e-4) and shows the trajectories. Empirical priors
help when the true distribution is skewed or multimodal; with a normal truth they mostly add noise (and BIC
penalises their Q − 1 − constraints parameters).

## Estimated normal prior

The exact M-step maximises the discretised-normal objective. "Posterior moments" (TAM, irtQ and mirt) sets μ and σ
to the posterior moments. That is exact for a continuous normal and a generalised step on a grid. The two agree to
about 1e-7 from roughly 15 nodes; on a 7-node grid they differ by about 0.02 in σ.

## Fixed empirical prior

Masses can be typed in, built from an illustrative shape (normal, skewed, bimodal or uniform), or imported from a
CSV with columns theta and weight. They are normalised to sum to 1, and their moments are shown. An imported grid
that does not match the calibration grid is rejected, not silently re-gridded, because the historical distribution
must already be on the intended metric.
