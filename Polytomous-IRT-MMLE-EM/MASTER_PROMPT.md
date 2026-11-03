# MASTER PROMPT v3.0
## Interactive Polytomous IRT Calibration Learning Laboratory

**Target:** GPT Sites / ChatGPT Work  
**Reference:** Existing MMMLEM educational application  
**Estimation framework:** Marginal Maximum Likelihood Estimation with Expectation–Maximization (MMLE–EM)  
**Model families:** Rasch-based polytomous models and general polytomous IRT models  
**Calibration designs:** Single-group, concurrent, fixed-item, concurrent fixed-item  
**Ability prior distributions:** Fixed normal, estimated normal, fixed empirical, estimated empirical  
**Primary reference packages:** TAM and irtQ  
**Supplementary reference:** mirt  
**Version:** 3.0 — November 2026

> Implementation note: this repository implements the prompt as a Streamlit application (the framework of the
> existing MMLE-EM labs here). See README.md and docs/LIMITATIONS.md.

---

# 1. Mission

Create a fully functional, privately previewable educational web application that teaches how polytomous item parameters are estimated using MMLE–EM.

The application must enable learners to understand the entire estimation process, from response probabilities and latent ability distributions through posterior calculations, expected counts, constrained parameter optimization, and convergence.

The learner must be able to change model assumptions and immediately examine their consequences.

**Primary educational objective:**

A learner should be able to inspect, reproduce, and explain every numerical calculation used in polytomous MMLE–EM calibration.

Build actual working functionality, not a prototype, static demonstration, or superficial mockup.

Do not overwrite the existing MMMLEM application. Do not publicly deploy the new application without approval.

# 2. Existing Project Inspection

Before implementation, inspect the available MMMLEM project materials:

- Existing application source code
- Master prompt and Markdown/R Markdown documentation
- Layout screenshots
- Simulation components
- EM walkthroughs
- Equations and numerical examples
- Parameter estimation functions
- Existing test cases

Preserve the useful instructional progression:

**Explore model → Simulate data → Calibrate items → Inspect EM iterations → Evaluate results → Compare methods.**

Reuse existing visual patterns when available.

Maintain separate **Concept**, **Mathematics**, **Numerical Example**, and **Implementation** views for each major concept.

If source material is unavailable, document the limitation rather than claiming it was inspected.

# 3. Required IRT Models

Implement four unidimensional polytomous IRT models.

| Family | Model | Abbreviation | Discrimination |
|---|---|---|---|
| Rasch | Partial Credit Model | PCM | Fixed at 1 |
| Rasch | Rating Scale Model | RSM | Fixed at 1 |
| General IRT | Generalized Partial Credit Model | GPCM | Estimated |
| General IRT | Graded Response Model | GRM | Estimated |

Use a common internal logistic scaling convention, initially \(D=1\).

## 3.1 Partial Credit Model

For item \(j\), ordered category \(k=0,\ldots,K_j\), and step difficulties \(\delta_{jh}\):

\[
P_{jk}(\theta)=
\frac{
\exp\left[\sum_{h=1}^{k}(\theta-\delta_{jh})\right]
}{
\sum_{c=0}^{K_j}
\exp\left[\sum_{h=1}^{c}(\theta-\delta_{jh})\right]
}.
\]

Empty sums equal zero.

Teach step parameters, adjacent-category odds, category curves, expected score, item information, and identification.

## 3.2 Rating Scale Model

Use the shared-step structure:

\[
\delta_{jh}=b_j+\tau_h,
\qquad \sum_h\tau_h=0.
\]

Then:

\[
P_{jk}(\theta)=
\frac{
\exp\left[\sum_{h=1}^{k}(\theta-b_j-\tau_h)\right]
}{
\sum_{c=0}^{K}
\exp\left[\sum_{h=1}^{c}(\theta-b_j-\tau_h)\right]
}.
\]

Estimate item locations \(b_j\) and common steps \(\tau_h\).

RSM items sharing the same rating-scale structure must have compatible category definitions.

**Mandatory:** Estimate shared steps through a joint constrained M-step, not independent PCM estimation followed by averaging.

## 3.3 Generalized Partial Credit Model

\[
P_{jk}(\theta)=
\frac{
\exp\left[a_j\sum_{h=1}^{k}(\theta-\delta_{jh})\right]
}{
\sum_{c=0}^{K_j}
\exp\left[a_j\sum_{h=1}^{c}(\theta-\delta_{jh})\right]
},
\qquad a_j>0.
\]

Provide interactive slope controls and explain their effect on category transition probabilities and item information.

Demonstrate the PCM special case \(a_j=1\).

## 3.4 Graded Response Model

For \(k=1,\ldots,K_j\):

\[
P(Y_j\ge k\mid\theta)
=
\operatorname{logit}^{-1}
[a_j(\theta-b_{jk})].
\]

Individual category probabilities are obtained by subtracting consecutive cumulative probabilities, with boundary probabilities equal to 1 and 0.

Require ordered thresholds:

\[
b_{j1}<b_{j2}<\cdots<b_{jK_j}.
\]

Demonstrate the difference between GRM cumulative boundaries and GPCM adjacent-category parameters.

All model calculations must use numerically stable probability functions.

# 4. Four Calibration Designs

Implement four calibration workflows for each model, subject to identification and design requirements.

| Design | Data structure | Fixed item parameters | Parameter estimation |
|---|---|---|---|
| Single-group | One population/form | None | Freely estimate all eligible items |
| Concurrent | Multiple groups/forms | None | Jointly estimate common and unique items |
| Fixed-item (FIPC) | New responses plus bank anchors | Existing anchor parameters | Estimate new items and eligible ability-distribution parameters |
| Concurrent fixed-item (MG-FIPC) | Multiple groups/forms plus bank anchors | Existing anchor parameters | Jointly estimate eligible free items and group distributions |

## 4.1 Single-Group Calibration

Calibrate a response matrix using one latent ability distribution.

Allow normal, fixed empirical, and estimated empirical distributions.

When no externally fixed anchors establish the metric, enforce the appropriate latent location and scale identification constraints.

## 4.2 Concurrent Calibration

Calibrate two or more groups simultaneously.

Common items must share the same parameter vector wherever administered.

Support:

- A pooled common-population assumption
- Distinct group ability distributions
- A reference-group identification design
- Normal or empirical ability distributions

A typical reference group uses:

\[
\mu_1=0,\qquad \sigma_1^2=1.
\]

Estimate eligible focal-group means, variances, and distribution shapes on that reference metric.

## 4.3 Fixed-Item Calibration

Allow previously calibrated items to be fixed while new items are estimated.

The fixed item parameters must remain unchanged.

Their response probabilities still contribute to posterior ability estimation.

Support two explicitly distinguished methods:

- **NWU–OEM:** No prior-weights updating and one EM cycle.
- **MWU–MEM:** Multiple prior-weights updates and multiple EM cycles.

Do not describe an OEM one-cycle estimate as a converged MEM solution.

Allow estimation of the new examinee population distribution on the established anchor-item scale.

## 4.4 Concurrent Fixed-Item Calibration

Combine multiple administrations with previously calibrated fixed bank items.

Support:

- Multiple years
- Multiple test forms
- Shared fixed anchors
- Shared freely estimated items
- Unique free items
- Group-specific ability distributions

Fixed anchors define the established metric when sufficiently informative.

Every shared free item's parameters must remain equal across the groups in which it appears.

Include simulations showing how anchor drift affects concurrent fixed-item calibration.

# 5. Ability Prior Distribution Module

**This is a mandatory central learning module.**

Provide four distinct prior settings, available across calibration designs when identifiable.

| Prior setting | Distribution | Parameters updated during EM? |
|---|---|---|
| A. Fixed normal | \(N(\mu,\sigma^2)\) | No |
| B. Estimated normal | \(N(\mu,\sigma^2)\) | Eligible mean and variance |
| C. Fixed empirical | Discrete weights supplied from historical data | No |
| D. Estimated empirical | Discrete masses on an ability grid | Yes, subject to identification constraints |

Use fixed standard normal as the default introductory setting.

Explain that these are **latent ability distributions**, not Bayesian priors on item parameters.

If optional Bayesian or penalized item parameter estimation is introduced, clearly distinguish its penalty or item-parameter prior from the ability distribution used in MMLE.

## 5.1 Fixed Normal Prior

\[
\theta\sim N(\mu,\sigma^2).
\]

The learner specifies mean and variance.

The quadrature probabilities or integration weights remain fixed during estimation.

For the introductory case:

\[
\mu=0,\quad \sigma^2=1.
\]

Provide controls to vary both values while showing the effect on marginal probabilities and posterior distributions.

## 5.2 Estimated Normal Prior

For groups whose ability distribution parameters are identified and freely estimable, estimate:

\[
\mu_g,\qquad \sigma_g^2.
\]

Show how the group distribution changes over EM iterations.

Use a mathematically consistent estimation method with stable integration.

The learner must distinguish:

- Fixed item parameters versus free item parameters
- Fixed group distribution parameters versus free group distribution parameters
- Reference-group constraints versus group-specific estimates

Never freely estimate an unidentified common location and scale simultaneously with an otherwise fully free item parameterization.

## 5.3 Fixed Empirical Prior

Allow importing or constructing a discrete ability distribution.

For quadrature points:

\[
\theta_1,\theta_2,\ldots,\theta_Q,
\]

define weights:

\[
\pi_1,\pi_2,\ldots,\pi_Q
\]

with:

\[
\pi_q\ge0,\qquad \sum_q\pi_q=1.
\]

These weights remain unchanged during calibration.

Allow users to:

- Enter weights manually
- Import historical quadrature weights
- Generate illustrative empirical distributions
- Normalize weights
- Inspect moments
- Plot the discrete distribution
- Compare it against a normal distribution

Require the historical ability distribution to be on the intended item calibration metric.

Do not silently standardize or rescale historical priors without showing the transformation and its implications.

## 5.4 Estimated Empirical Prior

Implement nonparametric discrete ability distribution estimation on a fixed quadrature grid.

For examinee \(i\), at point \(q\):

\[
w_{iq}^{(t)}
=
\frac{
\pi_q^{(t)}L_i(\theta_q;\xi^{(t)})
}{
\sum_s\pi_s^{(t)}L_i(\theta_s;\xi^{(t)})
}.
\]

Where:

- \(w_{iq}\): posterior probability
- \(\pi_q\): prior mass at quadrature point \(q\)
- \(L_i(\theta_q)\): conditional response likelihood
- \(\xi\): item parameter vector

For an unconstrained discrete distribution:

\[
\boxed{
\pi_q^{(t+1)}
=
\frac{1}{N}
\sum_{i=1}^{N}w_{iq}^{(t)}
}
\]

This update maximizes the expected complete-data prior-mass objective on the fixed grid when no additional moment constraints or penalties are imposed.

**Important identification requirement:**

When the latent mean and variance are fixed to identify freely calibrated item parameters, the empirical masses must satisfy the corresponding moment constraints, such as:

\[
\sum_q\pi_q\theta_q=0
\]

and

\[
\sum_q\pi_q\theta_q^2=1.
\]

The unconstrained update generally does not satisfy these conditions.

Implement an actual constrained optimization method for the empirical weights when these restrictions apply.

Do not simply calculate unconstrained weights and standardize or recenter them afterward while leaving item parameters unchanged.

For a fixed-anchor calibration, allow the population mean and variance to vary if the anchor configuration adequately identifies the scale.

Document differences between the implemented empirical histogram estimator and irtQ's Woods (2007) empirical histogram approach.

## 5.5 Group-Specific Empirical Priors

For concurrent and MG-FIPC calibration, support:

\[
\boldsymbol\pi_g=
(\pi_{g1},\ldots,\pi_{gQ}).
\]

The E-step must use the relevant group's prior masses.

For a group of size \(N_g\), the unconstrained update is:

\[
\pi_{gq}^{(t+1)}
=
\frac{1}{N_g}
\sum_{i\in g}w_{igq}^{(t)}.
\]

Apply appropriate reference-group or anchor constraints.

Provide options for:

- One common prior across groups
- Group-specific normal priors
- Group-specific fixed empirical priors
- Group-specific estimated empirical priors

In a pooled common-prior design, pool posterior expected masses across groups using sample-size weighting.

In a group-specific design, do not inadvertently pool distributions.

## 5.6 Prior Distribution Explorer

Build an interactive teaching panel containing:

1. Prior distribution plot
2. Quadrature grid
3. Current prior weights
4. Selected examinee's conditional likelihood
5. Selected examinee's posterior distribution
6. Expected category counts
7. Updated prior weights
8. Prior moments
9. Updated item parameters
10. Marginal likelihood change

Allow direct comparison of normal and empirical priors.

Show how a skewed or multimodal ability distribution can change calibration results.

Do not imply that empirical priors always improve estimation; provide simulations showing when they help, when differences are negligible, and when they increase instability.

# 6. Shared MMLE–EM Engine

Use one reusable estimation engine for all models, prior options, and calibration designs.

Let:

- \(g\): group
- \(i\): examinee
- \(j\): item
- \(k\): response category
- \(q\): quadrature point
- \(\xi\): item parameters
- \(\pi_{gq}\): group-specific prior masses or quadrature weights

## 6.1 Observed-data Marginal Likelihood

\[
\ell(\xi,\pi)=
\sum_g\sum_{i\in g}
\log\left[
\sum_q\pi_{gq}
\prod_{j\in O_{ig}}
P_{j,y_{igj}}(\theta_q;\xi)
\right],
\]

where \(O_{ig}\) denotes the items observed for examinee \(i\).

Use a numerically stable log-sum-exp implementation.

Correctly handle missing-by-design responses without treating them as incorrect or zero-category responses.

## 6.2 E-step

Calculate:

\[
w_{igq}
=
\frac{
\pi_{gq}
\prod_{j\in O_{ig}}P_{j,y_{igj}}(\theta_q;\xi)
}{
\sum_s\pi_{gs}
\prod_{j\in O_{ig}}P_{j,y_{igj}}(\theta_s;\xi)
}.
\]

Display the actual numerical contribution of each term.

Verify:

\[
\sum_qw_{igq}=1.
\]

## 6.3 Category Expected Counts

For each observed item:

\[
C_{gjkq}=
\sum_{i\in g}
I(Y_{ij}=k)w_{igq}.
\]

Display contributions by examinee, category, quadrature point, item, and group.

For shared items, aggregate category counts appropriately while preserving the common-parameter constraints.

## 6.4 M-step for Item Parameters

The item portion of the expected complete-data objective is:

\[
Q_{\text{item}}(\xi)=
\sum_{g,j,k,q}
C_{gjkq}\log P_{jk}(\theta_q;\xi).
\]

Implement model-specific optimizers:

- PCM: item-specific step parameters
- RSM: jointly constrained item locations and shared steps
- GPCM: positive slopes and step parameters
- GRM: positive slopes and ordered cumulative thresholds

Exclude fixed item parameters from optimization.

## 6.5 M-step for Prior Distribution

For a discrete empirical prior:

\[
Q_{\text{prior}}(\pi)
=
\sum_{g,i,q}
w_{igq}\log\pi_{gq}.
\]

Update free masses subject to their required constraints.

For normal priors, estimate free group-distribution parameters by maximizing a correctly defined expected complete-data or observed-data objective under the selected numerical integration representation.

Clearly distinguish true M-step maximization from generalized EM updates that merely improve the objective.

If a parameter-dependent quadrature grid is used, implement the necessary changes consistently rather than assuming fixed-grid formulas automatically apply.

## 6.6 Convergence

Report at each iteration:

- Marginal log likelihood
- Log-likelihood difference
- Largest item parameter change
- Largest prior-weight change
- Largest group-parameter change
- Inner optimizer status
- Convergence flag

Show likelihood and parameter trajectories.

Detect and flag unexpected likelihood decreases.

For empirical priors, use convergence diagnostics that include the latent distribution and not only item parameters.

# 7. Numerical Walkthrough Laboratory

Use a small deterministic dataset for all initial examples.

Suggested teaching example:

- \(N=5\) examinees
- \(J=3\) items
- Categories 0, 1, 2
- Three quadrature points

For standard-normal three-point Gaussian–Hermite integration:

\[
\theta=(-\sqrt3,0,\sqrt3)
\]

and

\[
\pi=(1/6,2/3,1/6).
\]

Explain that this is a low-order teaching approximation.

Enable clicking through:

1. View response data
2. Choose initial item parameters
3. Choose prior distribution
4. Calculate category probabilities
5. Calculate response-pattern likelihoods
6. Calculate unnormalized posterior probabilities
7. Normalize posterior probabilities
8. Compute expected category counts
9. Optimize item parameters
10. Update eligible prior parameters
11. Recalculate marginal likelihood
12. Evaluate convergence
13. Continue to the next iteration

Allow a learner to inspect every intermediate matrix.

Include copyable calculation tables and full-precision values.

All displayed numbers must come from the actual computational engine.

# 8. Gauss–Hermite and Numerical Integration Module

Teach the distinction between:

- Gaussian–Hermite integration
- Equally spaced quadrature
- Fixed discrete empirical integration
- Estimated discrete empirical distributions

Implement interactive controls for:

- Number of nodes
- Grid boundaries
- Ability distribution
- Mean and variance
- Grid resolution
- Numerical tolerance

Compare low-order and high-order integration.

**Reference-package distinction:** irtQ's documented `Quadrature = c(Q, bound)` defines evenly spaced points on a symmetric interval. Do not describe that setting as Gaussian–Hermite quadrature.

Allow category probabilities and marginal likelihoods to be compared across integration methods.

# 9. Calibration and Prior Comparison Laboratory

Create a dedicated interactive comparison environment.

The learner selects:

**Model:** PCM / RSM / GPCM / GRM

**Calibration:** Single / Concurrent / FIPC / MG-FIPC

**Prior:** Fixed normal / Estimated normal / Fixed empirical / Estimated empirical

The system must enforce identification requirements and explain why certain combinations require constraints or sufficiently informative anchors.

Provide comparisons of:

- Estimated item thresholds
- Estimated discriminations
- Estimated group ability moments
- Empirical distribution shapes
- Marginal log likelihoods
- Parameter recovery
- Convergence behavior
- Anchor sensitivity

Every comparison must use a consistent parameterization and valid latent metric.

# 10. Simulation and Parameter Recovery

Generate actual response data under the selected model and latent distribution.

Default conditions:

- \(N=500,2000,5000\)
- \(J=10,20,40\)
- Categories: 3, 4, or 5
- Replications: 30
- Reproducible random seeds

Support generating distributions that are:

- Normal
- Skewed
- Bimodal
- Discrete empirical
- Group-specific mixtures

Compare model calibration under correctly and incorrectly specified ability distributions.

Report bias, RMSE, MAD, empirical standard deviation, convergence rate, and runtime.

Calculate standard errors and confidence interval coverage when the estimator supports a defensible uncertainty calculation.

Distinguish empirical recovery variability from reported model-based standard errors.

For concurrent designs, report group-parameter recovery.

For FIPC and MG-FIPC, report recovery of free item parameters and sensitivity to anchor error.

# 11. Model Fit and Category Probability Comparison

Calculate expected marginal category probabilities:

\[
\widehat p_{gjk}
=
\sum_q\widehat\pi_{gq}
P_{jk}(\theta_q;\widehat\xi).
\]

Compare with observed category proportions for each item and group.

Provide:

- Observed vs predicted scatterplots
- Absolute differences
- RMSE and MAD of category-proportion discrepancies
- Category residuals
- Probability curves
- Item-level summaries
- Appropriate additional IRT model-fit checks

Make clear that close agreement in category proportions alone is not sufficient to establish good model fit.

Allow fitting and comparison of normal versus empirical priors to reveal potential latent-distribution misspecification.

# 12. Scale Transformation and Identification

Use a consistent canonical parameterization.

Document conversions to TAM, irtQ, and mirt.

For adjacent-category models:

\[
P_{jk}(\theta)
\propto
\exp(k a_j\theta+\gamma_{jk}),
\]

where:

\[
\gamma_{jk}
=
-a_j\sum_{h=1}^{k}\delta_{jh}.
\]

For GRM, distinguish threshold and intercept conventions.

Test parameter transformations by reconstructing category response probabilities under both representations.

For freely estimated item parameters, explain location-scale indeterminacy and the identifying role of the latent distribution.

For fixed-anchor calibration, explain how an established item metric can identify the latent population distribution.

Never treat the mere use of a historical passing score or raw-score standard as sufficient evidence of scale invariance.

# 13. Drift and Anchor Sensitivity

Include a simulation module for item parameter drift.

Scenarios:

- Stable items across administrations
- Items becoming easier
- Items becoming harder
- Group ability shift
- Changed latent distribution shape
- Drifting anchors
- Stable anchors with changing examinee populations

Compare:

- Free concurrent calibration
- Fixed-item calibration
- Concurrent fixed-item calibration
- Different ability prior specifications

Display estimated drift against generating drift.

Teach learners how changes in item parameters may be confounded with changes in latent population distributions when identification is inadequate.

# 14. User Interface

Provide the following navigation structure:

| Page | Function |
|---|---|
| 1. Overview | Orientation, learning objectives, model families |
| 2. Model Explorer | Equations, category probability curves, information |
| 3. Prior Distribution Explorer | Normal and empirical prior comparisons |
| 4. Data Simulation | Generate/load response data |
| 5. Quadrature Laboratory | Numerical integration |
| 6. EM Walkthrough | E-step, M-step, every intermediate calculation |
| 7. Calibration Designs | Single, concurrent, FIPC, MG-FIPC |
| 8. Results | Item, group, prior estimates and convergence |
| 9. Parameter Recovery | Monte Carlo experiments |
| 10. Model Fit | Observed vs predicted probabilities |
| 11. Drift Analysis | Anchor sensitivity and scale maintenance |
| 12. Reference Validation | TAM, irtQ, and supported alternatives |

Provide two modes:

### Guided mode

Structured instruction, step navigation, equations, numerical explanations, and knowledge-check questions.

### Exploration mode

Full parameter controls, simulation settings, model selection, calibration design, prior options, and diagnostic exploration.

All controls must work and affect the reported results.

# 15. R Package Validation

## 15.1 TAM

Use TAM as the primary Rasch reference.

Verify:

- `TAM::tam.mml()` with `irtmodel = "PCM"`
- `TAM::tam.mml()` with `irtmodel = "RSM"`
- Relevant fixed-item parameter mechanisms where the full parameterization supports them
- Group-distribution calibration mechanisms

Use `tam.mml.2pl()` for supplementary GPCM validation where appropriate.

Do not assume that TAM supports every empirical-prior design through the same interface as irtQ.

## 15.2 irtQ

Use irtQ as the primary reference for general polytomous MMLE–EM calibration.

Verify:

- `irtQ::est_irt()` for single-group calibration
- `irtQ::est_mg()` for multiple-group calibration
- `fipc = TRUE` for fixed-item calibration
- `fipc.method = "OEM"` and `"MEM"`
- `EmpHist = TRUE` for empirical histogram estimation
- `weights` for supplied fixed quadrature distributions
- `free.group` for free group-distribution parameters
- `fix.id` and `fix.loc` for anchor identification

Retrieve empirical distribution results using the documented `getirt(..., what = "weights")` interface.

Use accurate metadata structures for each item model. Do not assume that simply passing a model label and response matrix is sufficient in every reference-function call.

## 15.3 Independent Validation

Compare:

- Category probabilities
- Item parameter estimates after transformation
- Marginal likelihoods where definitions agree
- Prior distribution estimates
- Group mean and variance estimates
- Fixed-anchor invariance
- Numerical convergence

Record the software version, estimation settings, quadrature representation, identification constraints, and numerical tolerances.

If an R validation run cannot be executed, report it as unverified rather than claiming agreement.

# 16. Software Architecture

Separate UI from computation.

Suggested modules:

```text
polytomous-mmle-em-lab/
  README.md
  MASTER_PROMPT.md
  docs/
    THEORY.md
    CALIBRATION_DESIGNS.md
    PRIOR_DISTRIBUTIONS.md
    NUMERICAL_EXAMPLES.md
    VALIDATION.md
  core/
    models/
      PCM
      RSM
      GPCM
      GRM
    quadrature/
    posterior/
    expected_counts/
    em_engine/
    prior_estimation/
      fixed_normal
      estimated_normal
      fixed_empirical
      estimated_empirical
      constrained_weights
    calibration/
      single
      concurrent
      fipc
      mg_fipc
    transformations/
    simulation/
    diagnostics/
  ui/
    guided_mode/
    exploration_mode/
    prior_explorer/
    em_walkthrough/
    calibration_comparison/
  r_reference/
    tam_validation.R
    irtq_validation.R
    mirt_validation.R
  tests/
    model_tests
    quadrature_tests
    posterior_tests
    prior_tests
    constraint_tests
    em_tests
    calibration_tests
    simulation_tests
    validation_tests
```

Use a supported framework for GPT Sites.

Prefer a portable TypeScript computational core when supported, or a tested Python-based implementation when necessary.

Never substitute static demonstration results for working calculations.

# 17. Testing and Acceptance Requirements

Implement automated tests verifying:

1. Category probabilities sum to one.
2. PCM and fixed-unit-slope GPCM are equivalent.
3. RSM common-step constraints hold.
4. GRM thresholds remain ordered.
5. Quadrature and empirical prior weights are valid.
6. Posterior probabilities normalize correctly.
7. Expected category counts reconcile with observed responses.
8. Unconstrained empirical weight updates match the fractional-count formula.
9. Constrained empirical weight estimates satisfy identification conditions.
10. Fixed anchors remain exactly unchanged.
11. Shared free items have identical parameters across groups.
12. Reference-group and anchor-based identification is correct.
13. Estimated normal distributions have valid positive variances.
14. Marginal likelihood behavior is numerically consistent with the optimizer.
15. Parameter transformations preserve model probabilities.
16. Numerical examples reproduce engine-generated values.
17. Parameter recovery simulations are reproducible.
18. Prior misspecification can be detected in suitable simulation scenarios.
19. OEM and MEM produce results consistent with their different update schedules.
20. All UI controls and calculation navigation work.

Explicitly handle:

- Nonconvergence
- Weak identification
- Sparse response categories
- Extreme thresholds
- Nearly empty empirical prior bins
- Insufficient anchors
- Incompatible rating scale categories
- Invalid or duplicated item IDs
- Missing-by-design observations
- Numerical instability

# 18. Deliverables

Produce:

1. Functional private GPT Sites preview, if supported
2. Complete runnable source code
3. Mathematical theory documentation
4. Calibration design explanations
5. Prior distribution estimation explanations
6. Detailed numerical EM walkthrough
7. Simulation and parameter recovery functionality
8. Model-fit diagnostics
9. R reference-validation scripts and results
10. Automated test results and a limitations report

Do not publicly deploy or overwrite the original application before approval.

## Final Acceptance Criterion

A learner must be able to select:

- **Model:** PCM, RSM, GPCM, or GRM
- **Calibration:** Single, concurrent, fixed-item, or concurrent fixed-item
- **Ability prior:** Fixed normal, estimated normal, fixed empirical, or estimated empirical

Then generate or load data, run actual MMLE–EM estimation, and inspect every stage of the calculation.

For any iteration, the learner must be able to see:

- Category probabilities
- Quadrature nodes and weights
- Examinee posterior probabilities
- Expected category counts
- Item parameter updates
- Prior distribution updates
- Group parameter updates
- Anchor constraints
- Marginal log likelihood
- Convergence diagnostics

**The application is complete only when the numerical engine, interactive educational interface, calibration workflows, prior estimation methods, and automated tests all genuinely function.**
