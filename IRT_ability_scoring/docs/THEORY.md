# Theory

## Setting

Item parameters are known. Examinee i answered items O_i (missing responses are skipped); y_ij ∈ {0, …, m_j}
scores y_ij points. Every model gives category probabilities P_jk(θ) with θ-derivatives P′_jk, P″_jk, and

    ℓ_i(θ) = Σ_{j∈O_i} log P_{j,y_ij}(θ)                 log-likelihood
    I(θ)   = Σ_j Σ_k P′_jk² / P_jk = E[−ℓ″(θ)]            test information
    J(θ)   = Σ_j Σ_k P′_jk P″_jk / P_jk                    Warm's J
    T(θ)   = Σ_j Σ_k k P_jk(θ)                            test characteristic curve

## Models (D = 1 by default; 1.702 approximates the normal ogive)

| Model | Category probabilities | Constraints |
|---|---|---|
| Rasch / 1PL | P = Ψ(D(θ − b)) | a = 1 |
| 2PL | P = Ψ(Da(θ − b)) | a > 0 |
| 3PL | P = c + (1 − c)Ψ(Da(θ − b)) | 0 ≤ c < 1 |
| 4PL | P = c + (d − c)Ψ(Da(θ − b)) | 0 ≤ c < d ≤ 1 |
| PCM | P_k ∝ exp Σ_{v≤k} D(θ − δ_v) | a = 1; δ unordered allowed |
| GPCM | P_k ∝ exp Σ_{v≤k} Da(θ − δ_v) | a > 0 |
| RSM | PCM with δ_jv = b_j + τ_v, τ shared | a = 1 |
| GRM | P*_k = Ψ(Da(θ − b_k)), P_k = P*_k − P*_{k+1} | b_1 < … < b_m |

Ψ(z) = 1/(1 + e^−z). Log-derivatives used by the code:

- 4PL: P′ = (d − c)Da Ψ(1 − Ψ), P″ = (d − c)(Da)² Ψ(1 − Ψ)(1 − 2Ψ).
- Divide-by-total: d log P_k/dθ = Da(k − E), d² log P_k/dθ² = −(Da)² Var (same for every category).
- GRM: d log P_k/dθ = Da(1 − P*_k − P*_{k+1}), d² log P_k/dθ² = −(Da)²[P*_kQ*_k + P*_{k+1}Q*_{k+1}].

Exponential-family items (Rasch, 2PL, PCM, GPCM, RSM) have −ℓ″ = I exactly; for 3PL, 4PL and GRM the observed and
expected information differ.

## Estimators

**MLE.** θ̂ = argmax ℓ(θ). Newton–Raphson θ ← θ − ℓ′/ℓ″. SE = 1/√I(θ̂). No finite MLE exists when every observed
response is in the lowest (highest) category: every log P_{j,y} is then monotone in θ, so ℓ′ never vanishes.
3PL/4PL likelihoods can have several maxima or a supremum at ±∞ even for mixed patterns.

**EAP.** θ̂ = ∫θ L g / ∫L g ≈ Σ θ_q L(θ_q) w_q / Σ L(θ_q) w_q, posterior SD as uncertainty. Gauss–Hermite:
θ_q = μ + √2σx_q, w_q = h_q/√π. Rectangular: equally spaced θ_q, w_q ∝ φ. Log-sum-exp normalisation. Always finite;
shrunk toward μ.

**MAP.** θ̂ = argmax ℓ + log g; with g = N(μ, σ²): U_MAP = ℓ′ − (θ − μ)/σ², curvature ℓ″ − 1/σ².
SE ≈ [I + 1/σ²]^−½ (expected) or [−ℓ″ + 1/σ²]^−½ (observed).

**WLE (Warm, 1989).** The MLE's first-order bias is B(θ) = J/(2I²); solving U_W = ℓ′ + J/(2I) = 0 removes it.
I′ − J = Σ_j Σ_k P_jk (d log P_jk/dθ)(d² log P_jk/dθ²), which is zero for divide-by-total items (the second factor is
constant across categories and Σ P dlogP = 0) and, by telescoping, for the GRM: with w_k = P*_k(1 − P*_k),
P_k dlogP_k = Da(w_k − w_{k+1}) and d²logP_k = −(Da)²(w_k + w_{k+1}), so the sum is −(Da)³Σ(w_k² − w_{k+1}²) = 0. For
these models the WLE maximises L·√I (mirt's definition); with 3PL/4PL items it does not. SE = 1/√I(θ̂_W).

**TCC inversion.** Solve T(θ) = r on the observed items. T′ = Σ_k k P′_jk > 0 when every a > 0, so the root is unique
when T(−∞) < r < T(+∞), with T(−∞) = Σc_j (3PL/4PL) and T(+∞) = Σ d_j or Σ m_j. Delta-method SE √ΣVar_j / T′. For
Rasch-family items ℓ′ = D(r − T), so MLE = TCC; for 2PL ℓ′ = DΣa_j(y_j − P_j), which depends on Σa_j y_j.

**EAPsum.** P(R = r | θ) by the Lord–Wingersky recursion f_j(s) = Σ_k f_{j−1}(s − k) P_jk; θ̂(r) = E(θ | R = r) over
the same quadrature as EAP. One value per summed score (per observed-item set).

**Advanced.** Posterior median (fine-grid CDF); plausible values (posterior draws, for population statistics only);
Rasch PROX θ ≈ b̄ + √(1 + s_b²/2.89)·ln(r/(n − r)); adjusted MLE (extreme scores solved at r ± 0.3, a convention).

## SE versus posterior SD

A frequentist SE describes sampling variation of θ̂ for a fixed θ; a posterior SD describes uncertainty about θ given
the responses and the prior. PSDs are smaller (the prior adds information) and do not include the shrinkage bias,
so their intervals have nominal coverage on average over the population, not for every θ.
