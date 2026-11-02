# MASTER PROMPT — Rasch Calibration Learning Lab

**Project:** Interactive Educational Rasch Calibration App  
**Technology:** Python, Streamlit, R, TAM, eRm  
**Purpose:** Teach, demonstrate, simulate, and compare Rasch model calibration methods  
**Scope:** Dichotomous Rasch (1PL) model  
**Implementation requirement:** Build and test a working application, not just a prototype, plan, or UI mockup.

> Saved verbatim as the specification for this project (supplied 2026-11-01). The implementation lives in
> this folder; see README.md, docs/REFERENCE_AUDIT.md and docs/VALIDATION.md for how each section was met.

---

## 1. Role and Mission

You are an expert team consisting of:

- Psychometrician specializing in Rasch and IRT estimation
- Statistical computing researcher specializing in maximum likelihood and EM
- Python and R integration developer
- Streamlit application developer
- Educational interaction and visualization designer
- Software quality-assurance engineer

Your task is to build a fully functional, interactive **Rasch Calibration Learning Lab**.

The application must support:

1. Joint Maximum Likelihood Estimation (JMLE)
2. Conditional Maximum Likelihood Estimation (CMLE)
3. Marginal Maximum Likelihood Estimation with Expectation-Maximization (MML-EM)
4. Warm's Weighted Likelihood Estimation (WLE) for person ability
5. Comparisons among estimation methods
6. Interactive explanations of mathematical estimation procedures

The educational goal is to help users understand **not only what an estimator produces, but how its estimates are obtained**.

Users should be able to manipulate data, inspect equations, examine intermediate numerical results, observe convergence, and compare estimation accuracy.

## 2. Mandatory Reference-Folder Inspection

Before designing or implementing anything, inspect the current project folder recursively.

Identify all potentially relevant resources, including:

- Existing MML-EM / MMLEEM Python applications
- `.py` files
- `.Rmd` and `.md` files containing prompts, layouts, or technical explanations
- Layout screenshots, diagrams, and image files
- Existing Streamlit configurations
- Dependency files, existing tests, and reusable modules

Do not assume a specific filename. Discover reference files by filename, content, and project structure.

### 2.1 Review the existing application

Analyze the existing MML-EM application for:

- Page structure and navigation
- Sidebar and control-panel configuration
- Simulation controls
- Mathematical explanation style
- Equation presentation
- Interactive diagrams
- Estimation progress presentation
- Numerical examples
- Iteration tables and diagnostics
- Chart and visualization styles
- Results presentation
- Educational guidance and tooltips

Identify components that should be reused or adapted.

### 2.2 Review Markdown/R Markdown resources

Extract relevant requirements about:

- User interface
- Educational sequencing
- Statistical explanations
- Required visualizations
- Layout consistency
- Data workflow
- Implementation constraints

Use these references to inform the new Rasch-specific design.

### 2.3 Review the layout image

Inspect the actual image visually.

Preserve its useful visual hierarchy, spacing, navigation patterns, typography, panel arrangements, and interactions as closely as practical using Streamlit.

Do not assume that all content or mathematical expressions shown in the old layout are appropriate for Rasch calibration.

The **visual design should be reference-driven**, while the **statistical content should be Rasch-specific**.

### 2.4 Reference audit

Before coding, produce a concise internal reference audit documenting:

| Reference | Content/function | Reuse decision | Proposed adaptation |
|---|---|---|---|
| Python application | Existing estimation workflow | Reuse/adapt | Rasch estimation |
| Markdown/Rmd | Teaching sequence | Reuse/adapt | Rasch theory |
| Layout image | UI reference | Reuse | Rasch learning lab |
| Other files | Dependencies/utilities | Case-by-case | Modular implementation |

Do not modify or delete the original reference materials.

Proceed from the audit directly into implementation without stopping for approval.

---

## 3. Core Statistical Model

All calibration methods must be based on the dichotomous Rasch model:

\[
P_{ij}=P(X_{ij}=1\mid\theta_i,b_j)
=\frac{\exp(\theta_i-b_j)}{1+\exp(\theta_i-b_j)}.
\]

Definitions:

- \(X_{ij}\): binary response for person \(i\), item \(j\)
- \(\theta_i\): person ability
- \(b_j\): item difficulty
- \(P_{ij}\): probability of a correct response
- \(N\): number of persons
- \(J\): number of items

All items must have discrimination fixed at 1.

The application must explain parameter identification because:

\[
P(\theta_i,b_j)
=
P(\theta_i+c,b_j+c).
\]

Use and document appropriate identification constraints for each estimation method.

Never introduce 2PL, 3PL, multidimensional, or polytomous estimation in the initial version.

---

## 4. Statistical Engine and Software Architecture

### 4.1 Reference estimation packages

Use the following validated R package functions.

| Estimation | Package/function | Role |
|---|---|---|
| JMLE | `TAM::tam.jml()` | Reference item/person calibration |
| CMLE | `eRm::RM()` | Conditional item calibration |
| MML-EM | `TAM::tam.mml()` | Reference marginal calibration |
| WLE | `TAM::tam.wle()` | Reference weighted person scoring |

Use TAM as the primary reference estimation engine for supported methods.

Use eRm for CMLE because it provides conditional Rasch estimation.

Verify function signatures and output structures against the installed package versions rather than assuming a particular version.

Official documentation:

- https://cran.r-project.org/package=TAM
- https://cran.r-project.org/package=eRm
- https://alexanderrobitzsch.r-universe.dev/TAM
- https://search.r-project.org/

### 4.2 Python educational algorithms

Implement transparent Python teaching versions of:

- JMLE iterative calibration
- CMLE conditional-likelihood optimization
- MML-EM with numerical quadrature
- Person MLE and Warm WLE

Use NumPy, SciPy, pandas, and other appropriate numerical libraries.

These implementations are for educational transparency and verification, not automatically assumed equivalent to TAM under all settings.

Clearly distinguish:

**Reference Result:** Output from the relevant R package.

**Educational Algorithm:** Results from the transparent Python implementation.

**Intermediate Demonstration:** Calculations illustrating a selected iteration or step.

Never present a reconstructed educational iteration as an actual internal iteration from TAM unless the values were genuinely extracted from TAM.

### 4.3 R–Python integration

Prefer a robust local bridge using:

- Python `subprocess`
- `Rscript`
- Validated CSV or JSON interchange
- Temporary directories
- Explicit data and parameter schemas

A direct `rpy2` integration can be supported if environment compatibility is confirmed, but must not be required.

Provide explicit checks for:

- R availability
- Installed TAM/eRm versions
- Package loading
- Data integrity
- Estimation convergence
- Execution failures

When R is unavailable, clearly indicate that reference validation cannot be performed. The Python educational implementation may continue independently if it is supported.

Never fabricate TAM results.

---

## 5. Application Layout and Navigation

Use the existing project's layout image and Markdown resources as primary design references.

Build a responsive, professional Streamlit interface with a persistent sidebar and a main educational workspace.

Recommended navigation:

| Page | Title | Main functionality |
|---|---|---|
| 1 | Rasch Model Explorer | Model introduction and ICC |
| 2 | Data & Simulation | Generate or upload responses |
| 3 | JMLE Lab | Joint estimation, step-by-step |
| 4 | CMLE Lab | Conditional estimation |
| 5 | MML-EM Lab | Marginal likelihood and EM |
| 6 | Person Scoring | MLE and Warm WLE |
| 7 | Estimator Comparison | Recovery and performance |
| 8 | Theory & Algorithms | Equations, derivations, references |

Each estimation page should use a common organization:

**A. Overview** — definition, rationale, what is estimated.

**B. Estimation controls** — settings relevant to the selected estimator.

**C. Algorithm visualization** — steps and iterative updates.

**D. Numerical walkthrough** — intermediate computations for a selected person, item, or iteration.

**E. Results** — calibrated parameters, uncertainties, likelihoods, and status.

**F. Diagnostics and interpretation** — convergence, limitations, and comparisons.

Use Streamlit and Plotly for interactive visuals where appropriate.

All displays must be driven by actual application state.

Avoid decorative controls, nonfunctional buttons, and misleading progress animations.

---

## 6. Rasch Model Explorer

Provide an interactive mathematical introduction.

Controls:

- Person ability \(\theta\)
- Item difficulty \(b\)
- Ability range
- Number of example items
- Item difficulty spread

Show:

1. Rasch equation
2. Probability of a correct response
3. Item characteristic curve
4. Relationship between \(\theta-b\) and probability
5. Fisher information

\[
I_j(\theta)=P_j(\theta)[1-P_j(\theta)].
\]

Display the effect of changing ability and difficulty in real time.

Explain how shifting \(\theta\) and \(b\) by the same constant leaves probabilities unchanged.

Add a simple interactive numerical example.

---

## 7. Data Generation and Upload

### 7.1 Simulation settings

Provide controls for:

| Parameter | Default | Suggested range |
|---|---:|---|
| Number of persons \(N\) | 500 | 50–5,000 |
| Number of items \(J\) | 20 | 5–100 |
| Ability mean | 0 | -2 to 2 |
| Ability SD | 1 | 0.3–2 |
| Difficulty mean | 0 | -2 to 2 |
| Difficulty SD | 1 | 0.3–2 |
| Missing response proportion | 0% | 0–20% |
| Random seed | 2026 | User-defined |

Use:

\[
\theta_i\sim N(\mu_\theta,\sigma_\theta^2),
\]

\[
b_j\sim N(\mu_b,\sigma_b^2),
\]

\[
X_{ij}\sim\operatorname{Bernoulli}(P_{ij}).
\]

Ensure consistent seed-based reproducibility.

Allow the learner to generate datasets with extreme response scores and challenging item/person targeting.

### 7.2 Data display

Show:

- Person-by-item response matrix
- Raw-score distribution
- Item p-values
- True ability distribution
- True difficulty distribution
- Missingness summary
- Extreme-score counts

### 7.3 Upload

Support uploaded response CSV files.

Validate:

- Binary 0/1 responses
- Missing values
- Unique item identifiers
- Correct person/item orientation
- At least the minimum data requirements
- Missingness and disconnected designs

Do not silently treat identifier columns as items.

Do not invent true parameters for uploaded operational data.

Only display recovery-to-truth statistics when known generating parameters are available.

---

## 8. JMLE Laboratory

### 8.1 Concept

Explain that JMLE estimates person abilities and item difficulties jointly:

\[
L(\boldsymbol\theta,\mathbf b)
=
\prod_{i=1}^N\prod_{j=1}^J
P_{ij}^{X_{ij}}(1-P_{ij})^{1-X_{ij}},
\]

with the product restricted to observed responses.

### 8.2 Estimation equations

Person score equation:

\[
U_{\theta_i}
=
\sum_{j\in O_i}(X_{ij}-P_{ij})=0.
\]

Item score equation:

\[
U_{b_j}
=
\sum_{i:j\in O_i}(P_{ij}-X_{ij})=0.
\]

Show iterative Newton/Fisher-scoring updates with observed-item information:

\[
I_{\theta_i}
=
\sum_{j\in O_i}P_{ij}(1-P_{ij}),
\]

\[
I_{b_j}
=
\sum_{i:j\in O_i}P_{ij}(1-P_{ij}).
\]

Explain and implement:

- Starting values
- Alternating person/item updates
- Identification adjustment
- Damping or step control
- Convergence thresholds
- Extreme-score handling
- Bias adjustment where selected

### 8.3 Interactive estimation

Show an iteration table with:

- Iteration number
- Selected \(\hat\theta_i\)
- Selected \(\hat b_j\)
- Score and information
- Maximum parameter change
- Joint log-likelihood
- Convergence status

Provide an iteration slider.

Visualize:

- Ability trajectories
- Item-difficulty trajectories
- Log-likelihood convergence
- Maximum change by iteration

Display the TAM output alongside the educational Python result.

Clearly label TAM-specific adjustment defaults and any differences from the Python implementation.

### 8.4 Statistical limitations

Explain:

- Joint estimation of many incidental person parameters
- Potential finite-test-length bias
- Nonfinite unadjusted MLEs for all-correct/all-incorrect patterns
- Why convergence alone does not prove unbiasedness
- Differences between adjusted and unadjusted JMLE

---

## 9. CMLE Laboratory

### 9.1 Concept

Explain why a person's raw score is sufficient for ability under the Rasch model.

Let

\[
R_i=\sum_{j\in O_i}X_{ij}.
\]

Then:

\[
P(\mathbf X_i=\mathbf x_i\mid R_i=r_i,\mathbf b)
=
\frac{
\exp\left(-\sum_{j\in O_i}x_{ij}b_j\right)
}{
\gamma_{r_i}(\mathbf b;O_i)
},
\]

where \(\gamma_{r_i}\) is the corresponding elementary symmetric function over the observed item set.

### 9.2 Calculation

For complete data, demonstrate the coefficients of:

\[
\prod_{j=1}^J(1+te^{-b_j}).
\]

Use a numerically stable dynamic-programming algorithm for elementary symmetric functions.

Implement:

- Conditional log-likelihood
- Conditional score or numerical gradient
- Constrained item-difficulty optimization
- Proper handling of extreme raw-score groups
- Standard errors when defensibly available

Do not use brute-force enumeration of all \(2^J\) response patterns.

For missing data, condition on each person's actual observed item subset and handle patterns correctly. If the educational solver does not support a missingness design, display an explicit limitation and block that unsupported run.

### 9.3 Educational demonstration

Use a small example with approximately 4–6 items.

Show:

- Raw score
- Possible response patterns
- Numerator
- Conditional denominator
- Conditional probabilities
- Conditional log-likelihood
- Iterative item updates

Explain intuitively why ability disappears from the conditional likelihood.

Compare the Python implementation against `eRm::RM()`.

### 9.4 Important distinction

CMLE estimates item parameters conditionally.

Person MLE/WLE may then be computed separately using calibrated item parameters.

Do not incorrectly describe CMLE as estimating abilities and difficulties jointly.

---

## 10. MML-EM Laboratory

### 10.1 Concept

Explain integrating person ability out of the likelihood.

Assume:

\[
\theta\sim g(\theta;\boldsymbol\psi),
\]

with a normal distribution as the default.

\[
L(\mathbf b,\boldsymbol\psi)
=
\prod_i
\int
P(\mathbf x_i\mid\theta,\mathbf b)
g(\theta;\boldsymbol\psi)\,d\theta.
\]

### 10.2 Numerical quadrature

Approximate the marginal integral using quadrature points:

\[
L_i\approx
\sum_{q=1}^Q
w_q L_i(\theta_q\mid\mathbf b).
\]

Provide a control for quadrature resolution.

Clearly explain quadrature nodes and prior weights.

### 10.3 E-step

Compute posterior class/node weights:

\[
\tau_{iq}
=
\frac{
w_qL_i(\theta_q\mid\mathbf b)
}{
\sum_{h=1}^Q
w_hL_i(\theta_h\mid\mathbf b)
}.
\]

Provide an interactive examination of one person's posterior weight distribution.

Show intermediate values for:

- Quadrature node
- Prior weight
- Conditional likelihood
- Unnormalized weight
- Posterior weight

### 10.4 M-step

Maximize the expected complete-data log-likelihood.

For the item difficulty update, demonstrate solving:

\[
\sum_{i,j\in O_i}
\sum_q
\tau_{iq}
\left[
X_{ij}-P_j(\theta_q;b_j)
\right]=0
\]

separately for each item \(j\), where the outer sum is understood to include only the selected item.

Implement a stable one-dimensional optimizer or a safeguarded Newton procedure.

Compute the observed marginal log-likelihood to monitor EM convergence.

### 10.5 Identification and distribution settings

Default to a fixed ability distribution mean of zero with clearly specified variance estimation settings.

Make any estimated latent variance, mean constraints, or item-location constraints explicit.

Do not impose incompatible or redundant location constraints.

### 10.6 Educational EM visualization

Create an interactive loop:

**Initialize → E-step → M-step → Calculate likelihood → Check convergence**

Users must be able to inspect selected iterations rather than only final results.

Include:

- EM log-likelihood convergence
- Item-difficulty trajectories
- Posterior quadrature weights
- Before/after E-step and M-step results
- Maximum parameter change
- Termination criteria

Validate against `TAM::tam.mml()` under equivalent model and integration settings.

---

## 11. Person Scoring and WLE Laboratory

This page must clearly separate **item calibration** from **person scoring**.

After selecting calibrated item parameters from JMLE, CMLE, or MML-EM, estimate ability using:

1. Person maximum likelihood estimation (MLE)
2. Warm's weighted likelihood estimation (WLE)

For MML calibration, optional EAP scoring can be included as a clearly separate Bayesian/posterior approach, but it is not required.

### 11.1 Person MLE

Explain the person score equation:

\[
\sum_j(X_{ij}-P_{ij})=0.
\]

Show:

- Observed raw score
- Expected score
- Score residual
- Person information
- Estimated ability
- Standard error

### 11.2 Warm WLE

Explain Warm's bias-reduction principle.

For dichotomous Rasch scoring, illustrate the adjusted score:

\[
U_W(\theta)
=
U(\theta)
+
\frac{I'(\theta)}{2I(\theta)}
=0,
\]

where

\[
I(\theta)=\sum_j P_j(\theta)[1-P_j(\theta)].
\]

Verify the implementation and its sign convention against the reference formulation and TAM.

### 11.3 Person-level comparison

Show ability estimates by raw score or selected individual.

Visualize:

- MLE versus WLE
- Ability recovery for simulated data
- Standard errors
- Extreme score behavior
- Ability estimation bias

Compare to `TAM::tam.wle()`.

Do not describe WLE as a fourth item-calibration estimator.

---

## 12. Method Comparison Laboratory

Build a comparison page that uses the same response data across all estimators.

### 12.1 Item estimates

Compare JMLE, CMLE, and MML-EM for:

- Estimated item difficulties
- Item standard errors
- Parameter recovery
- Runtime
- Convergence
- Estimation failures
- Method-specific assumptions

### 12.2 Evaluation statistics

For simulated data with known true item parameters:

\[
Bias=\frac1J\sum_j(\hat b_j-b_j),
\]

\[
RMSE=\sqrt{\frac1J\sum_j(\hat b_j-b_j)^2},
\]

\[
MAD=\frac1J\sum_j|\hat b_j-b_j|.
\]

Also report:

- Correlation between true and estimated difficulty
- Mean signed error
- Mean absolute error
- Maximum absolute error
- Appropriate uncertainty summaries

Use the same identification convention or an explicitly documented scale alignment when comparing estimates.

Keep any post-hoc location alignment separate from the native fitted parameterization.

Do not silently change parameters or the latent distribution in ways that alter fitted probabilities.

### 12.3 Interactive comparisons

Required visualizations:

- True versus estimated difficulty scatterplots
- Difficulty-estimate difference plots
- Estimation error distributions
- Method comparison table
- Person-ability recovery comparisons
- Runtime comparisons
- Convergence trajectories

Include a clear explanation of why estimates may differ.

### 12.4 Optional repeated simulation

Add a controlled Monte Carlo mode.

Default:

- Replications: 30
- \(N=500\)
- \(J=20\)
- \(\theta\sim N(0,1)\)
- \(b\sim N(0,1)\), with documented location alignment

Report across replications:

- Average bias
- RMSE
- MAD
- Empirical SD
- Mean reported SE where available
- Convergence rate
- Runtime

Provide simulation progress and allow cancellation or reasonable computational limits.

Never substitute a single-run standard error for an empirical across-replication SD.

---

## 13. Numerical Walkthrough Requirements

This is a central educational feature.

Each estimation method must provide at least one complete numerical walkthrough with:

1. Small sample data
2. Starting parameter values
3. Equation applied
4. Substituted numerical values
5. Intermediate result
6. Updated parameter
7. Convergence or stopping check

The calculations must be generated from actual code, not hard-coded explanations disconnected from the displayed dataset.

For example, the MML-EM walkthrough must show actual quadrature posterior weights that sum to one, followed by an M-step update.

The JMLE walkthrough must show real score, information, and update calculations.

The CMLE walkthrough must show the conditional denominator computed from the current item difficulties.

Include previous/next-step navigation or an iteration selector.

---

## 14. Mathematical and Educational Standards

All instructional explanations must be psychometrically defensible.

Use:

- Rendered LaTeX equations
- Consistent notation
- Plain-language interpretations
- Small numerical examples
- Intermediate computation tables
- Relevant academic references

For every estimator, explain:

- What parameters are estimated?
- What likelihood is maximized?
- Which parameters are conditioned out or integrated out?
- What assumptions are required?
- What identification constraints apply?
- How are standard errors computed?
- What happens with extreme scores?
- What determines convergence?
- What are the statistical limitations?

Use collapsible panels so advanced technical content does not overwhelm introductory users.

Maintain separate **Concept**, **Mathematics**, **Numerical Example**, and **Implementation** teaching views wherever the existing layout supports this organization.

---

## 15. User Experience Requirements

The interface must be suitable for graduate students, psychometricians, and statistical analysts.

Design principles:

- Professional, uncluttered appearance
- Consistent navigation
- Responsive page layout
- Clear terminology
- Readable mathematical notation
- Interactive but not overly animated
- Proper loading, error, and completion states
- Tooltips for advanced settings
- Reasonable defaults
- Persistent shared dataset across pages

Add an onboarding walkthrough explaining the recommended sequence:

**Explore model → Simulate data → Calibrate items → Inspect estimation → Score persons → Compare methods**

No control may be decorative or inert.

All numerical results must correspond to the selected dataset, method, and estimation settings.

---

## 16. Software Architecture

Use a modular project structure resembling:

```text
rasch-calibration-learning-lab/
├── app.py
├── pages/
│   ├── 01_rasch_model.py
│   ├── 02_data_simulation.py
│   ├── 03_jmle.py
│   ├── 04_cmle.py
│   ├── 05_mml_em.py
│   ├── 06_person_scoring.py
│   ├── 07_comparison.py
│   └── 08_theory.py
├── rasch/
│   ├── model.py
│   ├── simulation.py
│   ├── jmle.py
│   ├── cmle.py
│   ├── mml_em.py
│   ├── scoring.py
│   ├── metrics.py
│   └── validation.py
├── r_bridge/
│   ├── bridge.py
│   ├── tam_jmle.R
│   ├── tam_mml.R
│   ├── tam_wle.R
│   └── erm_cmle.R
├── components/
│   ├── equations.py
│   ├── charts.py
│   ├── tables.py
│   └── walkthroughs.py
├── assets/
├── tests/
├── docs/
│   ├── THEORY.md
│   ├── IMPLEMENTATION.md
│   ├── REFERENCE_AUDIT.md
│   └── VALIDATION.md
├── requirements.txt
├── install_packages.R
├── README.md
└── RASCH_CALIBRATION_MASTER_PROMPT.md
```

Adapt the folder structure to the existing repository if needed.

Do not overwrite reference files.

### 16.1 Implementation standards

Use:

- NumPy and SciPy for numerical work
- pandas for data tables
- Plotly for interactive plotting
- Streamlit for the application
- R/TAM/eRm for reference estimation
- pytest for automated tests

Use stable numerical methods:

- Log-sum-exp for mixture/posterior calculations
- Stable logistic functions
- Appropriate optimizer safeguards
- Finite-value checks
- Documented tolerances
- Explicit convergence states

Avoid silently clipping parameters or probabilities in ways that change the estimator without disclosure.

---

## 17. Validation and Quality Assurance

Implement automatic tests for:

### Rasch model

- Correct logistic probabilities
- Parameter-shift invariance
- Expected score and information calculations

### JMLE

- Correct score equations
- Correct update directions
- Identification invariance
- Convergence behavior
- Extreme response behavior
- Reference comparison with TAM

### CMLE

- Correct elementary symmetric functions
- Conditional probability normalization
- Correct constrained optimization
- Correct treatment of extreme scores
- Agreement with eRm under equivalent settings

### MML-EM

- Posterior weights sum to one
- Quadrature weights normalized appropriately
- Correct E-step calculations
- M-step improvements
- Observed marginal likelihood convergence
- Reference comparison with TAM

### WLE

- Correct score adjustment
- Numerical root convergence
- Reasonable extreme-score behavior
- Agreement with TAM under equivalent definitions

### Application

- All pages load
- Shared dataset state persists
- Parameter controls modify actual results
- No stale-cache mixing of datasets or configurations
- CSV upload validation works
- Simulation results are reproducible
- Invalid inputs generate helpful errors
- Downloads contain actual current results

Include small deterministic test fixtures and test datasets with known or independently computed expected results.

Do not claim reference equivalence without actual numerical tests.

Record differences and explain their likely causes, including identification, quadrature, adjustment settings, tolerances, or package conventions.

---

## 18. Deliverables

Create a complete working local project.

Required outputs:

1. Functional Streamlit application
2. Python estimation implementations
3. R integration scripts
4. Reference-driven UI/layout implementation
5. Interactive statistical visualizations
6. Estimation walkthroughs
7. Method comparison dashboard
8. Automated tests
9. Installation and execution instructions
10. Mathematical documentation
11. Validation report
12. Final implementation summary

The Markdown documentation should contain equations, notation, algorithm steps, and implementation details.

Include commands for:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Include R setup instructions and a script for checking/installing the necessary R packages.

The application must be functional locally without requiring cloud deployment, authentication, or external API keys.

---

## 19. Implementation Order

Execute the project in this sequence.

**Phase 1 — Reference inspection**

Inspect all relevant files, identify the visual and educational design, and create `docs/REFERENCE_AUDIT.md`.

**Phase 2 — Statistical implementation**

Implement and test the Rasch model, data generation, JMLE, CMLE, MML-EM, and WLE algorithms.

**Phase 3 — R integration**

Implement TAM/eRm bridge functions and verify output conventions.

**Phase 4 — Application layout**

Create the Streamlit navigation, shared controls, visual design, and data workflow using the reference layout.

**Phase 5 — Educational interaction**

Implement equations, iteration explorers, numerical walkthroughs, and convergence visualization.

**Phase 6 — Comparison and validation**

Add method comparisons, simulation recovery, automated tests, and reference validation.

**Phase 7 — Final review**

Run the application, check the pages and workflows, fix errors, and document the final results.

Do not stop at planning or scaffolding.

---

## 20. Definition of Done

The project is complete only when:

- The Streamlit application launches successfully.
- Users can simulate Rasch response data.
- JMLE, CMLE, and MML-EM produce real item estimates.
- Person MLE and WLE produce real ability estimates.
- Estimation results are computed rather than hard-coded.
- Intermediate equations and numerical results are visible.
- Iterative convergence can be inspected.
- The statistical methods are correctly distinguished.
- R reference routines operate when the dependencies are installed.
- Reference comparisons and validation outcomes are documented.
- The interface meaningfully reflects the provided layout references.
- Core automated tests pass.
- The README explains installation, execution, and use.
- No button, tab, chart, or export is merely a placeholder.

## Final Instruction

**Build the actual Rasch Calibration Learning Lab now.**

Start by inspecting the current directory and its existing MML-EM Python files, R Markdown/Markdown documents, and layout images.

Use them as design references, but implement a statistically accurate Rasch calibration curriculum covering JMLE, CMLE, MML-EM, and WLE.

Prioritize mathematical accuracy, reproducibility, instructional transparency, UI consistency, and executable software.

Finish by reporting the implemented files, tests performed, any unresolved limitations, and the exact command to run the app.
