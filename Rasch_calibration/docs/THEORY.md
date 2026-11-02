# Theory

## Model and notation

$$P_{ij}=P(X_{ij}=1\mid\theta_i,b_j)=\frac{\exp(\theta_i-b_j)}{1+\exp(\theta_i-b_j)}$$

$X_{ij}\in\{0,1\}$ (missing responses excluded through the observed set $O_i$), $\theta_i$ ability,
$b_j$ difficulty, $r_i=\sum_{j\in O_i}X_{ij}$ raw score, $s_j$ item score, $I(\theta)=\sum_jP_j(1-P_j)$.
Discrimination is fixed at 1.

**Identification.** $P(\theta_i,b_j)=P(\theta_i+c,b_j+c)$ for any $c$, so one location constraint is needed:

| Method | Constraint used |
|---|---|
| JMLE | mean of person estimates = 0 (TAM `constraint = "cases"`), or mean difficulty = 0 |
| CMLE | $\sum_j b_j=0$ (eRm `sum0 = TRUE`) |
| MML-EM | ability distribution mean = 0 (no item constraint) |
| MLE / WLE | inherit the scale of the supplied difficulties |

Recovery statistics shift each method's estimates so their mean equals the mean true difficulty; that
post-hoc alignment is reported separately and never changes the fitted parameters.

## JMLE

Joint likelihood $L(\boldsymbol\theta,\mathbf b)=\prod_i\prod_{j\in O_i}P_{ij}^{X_{ij}}(1-P_{ij})^{1-X_{ij}}$.
Score equations: $\sum_{j\in O_i}(X_{ij}-P_{ij})=0$ (persons) and $\sum_i(P_{ij}-X_{ij})=0$ (items).
Newton updates divide the residual by the information $\sum P(1-P)$. Each outer iteration updates all
persons, re-anchors, then updates all items.

- **Extreme scores:** $r_i=0$ or $r_i=|O_i|$ has no finite MLE. Following TAM, the score is moved inward by
  `adj = 0.3`. With adjusted extremes the adjusted person and item totals differ, so the person equations
  cannot all hold exactly at the solution (the item equations do); TAM behaves identically.
- **Bias:** one parameter per person makes the item estimates inconsistent for fixed $J$ (Neyman & Scott,
  1948; Andersen, 1973). The difficulty scale is stretched by roughly $J/(J-1)$; TAM's default multiplies
  the final difficulties by $(J-1)/J$. Monte Carlo (VALIDATION.md): slope of estimates on truth 1.16 (J = 8)
  and 1.07 (J = 20) without the correction, 1.01 with it.
- **SE:** $1/\sqrt{\sum_iP_{ij}(1-P_{ij})}$ at the final estimates, conditional on $\hat{\boldsymbol\theta}$.

## CMLE

$$P(\mathbf x_i\mid r_i,\mathbf b)=\frac{\exp(-\sum_{j\in O_i}x_{ij}b_j)}{\gamma_{r_i}(\mathbf b;O_i)},\qquad
\gamma_r=\sum_{\mathbf x:\sum x=r}\prod_j\varepsilon_j^{x_j},\ \varepsilon_j=e^{-b_j}.$$

$\theta$ cancels because every pattern with score $r$ carries the same factor $e^{r\theta}$. ESFs are the
coefficients of $\prod_j(1+t\varepsilon_j)$ and follow $\gamma_r^{(k)}=\gamma_r^{(k-1)}+\varepsilon_k\gamma_{r-1}^{(k-1)}$.
Conditional score: $\partial\ell_c/\partial b_j=\sum_i[\varepsilon_j\gamma^{(-j)}_{r_i-1}/\gamma_{r_i}-x_{ij}]$.
Extreme scores have conditional probability 1 and are dropped. SEs: inverse observed conditional
information under $\sum b=0$. CMLE estimates items only; abilities are scored afterwards.

## MML-EM

$\theta\sim N(0,\sigma^2)$ on nodes $\theta_1..\theta_Q$ (default 21 on $[-6,6]$) with weights $w_q$.

- E-step: $\tau_{iq}=w_qL_i(\theta_q)/\sum_hw_hL_i(\theta_h)$; expected counts $N_{jq}=\sum_im_{ij}\tau_{iq}$,
  $R_{jq}=\sum_im_{ij}x_{ij}\tau_{iq}$.
- M-step: per item, root of $\sum_q[R_{jq}-N_{jq}P_j(\theta_q)]=0$ (safeguarded Newton); if estimated,
  $\sigma^2=\sum_i\sum_q\tau_{iq}\theta_q^2/N$.
- The observed marginal log-likelihood is non-decreasing across EM cycles (Bock & Aitkin, 1981).
- **SEs:** the app's main SE inverts the observed marginal information (finite differences of the exact
  Fisher-identity score). TAM's `se.xsi` is the complete-data value $1/\sqrt{\sum_qN_{jq}P(1-P)}$, which
  ignores uncertainty about each $\theta$ and is smaller; both are shown.
- **Log-likelihood convention:** the app renormalises node weights to sum to 1. TAM uses normal density ×
  node spacing, which sums to less than 1 when the grid truncates the distribution; the two log-likelihoods
  then differ by $N\log\sum_q w_q^{TAM}$. Estimates are identical (the constant cancels in $\tau$).

## Person scoring

MLE: $\sum_j(X_{ij}-P_j(\theta))=0$; infinite for extreme scores. WLE (Warm, 1989):
$\sum_j(X_{ij}-P_j)+I'(\theta)/(2I(\theta))=0$ with $I'=\sum_jP_j(1-P_j)(1-2P_j)$; finite for every score and
pulled toward the items' centre. SE $=1/\sqrt{I(\hat\theta)}$. TAM's `tam.wle` uses no extreme-score
adjustment for WLE; its MLE option (`WLE = FALSE`) moves extreme scores by 0.3, whereas this app reports
±∞ for extreme MLEs.

## Linking and anchoring (page 9)

Each calibration fixes its own origin, so two forms calibrated separately are on different scales: if group
B is abler, a separate Form B calibration absorbs the ability difference into easier-looking difficulties.
Linking puts every form on one scale, here an item bank, through **anchors** $K$ with bank values
$b_k^{bank}$. *Internal common items* are on both forms but have no bank value.

| Design | Runs | Puts the forms on the bank scale by |
|---|---|---|
| Separate + mean/mean | one per form | $c_F=\overline{b}^{\,bank}_K-\overline{\hat b}^{(F)}_K$, then $b^*=\hat b+c_F$, $\theta^*=\hat\theta+c_F$ (Rasch: slope 1, shift only) |
| Concurrent | one, stacked | one likelihood over $\mathbf X_A\cup\mathbf X_B$ (unadministered = missing); one shift $c$ |
| Fixed anchor | one per form | $b_k=b_k^{bank}$ for $k\in K$; other items free |
| Fixed anchor + concurrent | one, stacked | anchors fixed; common items estimated once from both groups |

- **Fixed items in each estimator.** JMLE: fixed items are never updated and no centring is applied; TAM's
  bias factor $(J-1)/J$ multiplies the free items only (`tam.jml(xsi.fixed=)` does the same). CMLE: the
  conditional likelihood is maximised over the free items, $b=c+Az$. MML-EM: the M-step updates only free
  items, and every group mean becomes free because the anchors carry the scale (`tam.mml(xsi.fixed=)`).
- **MML with groups.** $\theta_i\mid g\sim N(\mu_g,\sigma_g^2)$; E-step uses each person's own group prior;
  M-step $\mu_g$ = mean posterior mean, $\sigma_g^2$ = mean posterior $E[(\theta-\mu_g)^2]$; $\mu_A=0$ unless
  anchors are fixed (`tam.mml(group=)`). Concurrent MML with one population for two different groups is
  biased (VALIDATION.md).
- **Equivalence.** If the forms share only fixed anchors, the stacked likelihood factorises by form, so
  fixed + concurrent equals fixed anchor (exactly for JMLE and CMLE; for MML with free group means too).
  Internal common items break the factorisation and are why the stacked designs are more precise.
- **Anchor drift.** An anchor whose difficulty changed between the bank and a new form biases that form's
  items and group mean. Stability check: $d_k=\hat b_k^{(B)}-\hat b_k^{(A)}$ from the separate calibrations,
  centred on the **median** of $d$ (mean centring lets a few drifted anchors push the stable ones the other
  way), $z_k=d_k/\sqrt{SE_A^2+SE_B^2}$; flag when $|d_k|$ exceeds a threshold (default 0.5) and $|z_k|>1.96$.
  **Displacement** of a fixed anchor = one Newton step it would take if freed: (expected − observed score)
  ÷ information at the fixed value. A dropped anchor no longer links the forms: separate/fixed designs
  estimate it on each form; concurrent designs split it into a Form A item and a Form B item.

## References

Andersen (1970, 1973); Bock & Aitkin (1981); Fischer & Molenaar (1995); Kolen & Brennan (2014, *Test
Equating, Scaling, and Linking*); Neyman & Scott (1948); Warm (1989); Wright & Masters (1982); TAM and eRm
documentation on CRAN.
