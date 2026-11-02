"""Equation / explanation / notation lessons aligned with mmle_engine.py."""
LESSONS = [
("1 · Current response model",r"s_{jq}=\operatorname{logistic}[Da_j(x_q-b_j)],\quad P_j(x_q)=c_j+(1-c_j)s_{jq}",
 "Ability determines a probability; a Bernoulli draw determines a realized answer. D is a positive logistic scale constant. In 2PL, c=0. At θ=b, 3PL probability is (1+c)/2. Starting and generating slopes are positive; default fitting a/b remain unbounded.",
 r"a_j: \text{discrimination},\quad b_j: \text{difficulty},\quad c_j: \text{lower asymptote},\quad x_q: \text{ability node}",
 r"\alpha_j=Da_j,\quad\delta_j=-Da_jb_j,\quad\alpha_j\theta+\delta_j=Da_j(\theta-b_j)"),
("2 · Masked likelihood and posterior",r"\tau_{igq}^{(t)}=\frac{w_{gq}^{(t)}L_{igq}^{(t)}}{\sum_r w_{gr}^{(t)}L_{igr}^{(t)}},\qquad\sum_q\tau_{igq}^{(t)}=1",
 "Multiply likelihood by the current population masses, then normalize over nodes for this person. A posterior is a person-specific distribution; it is distinct from the group population update. Compute the products and normalization in log space. The first anchored E-step uses fixed-item responses; MEM later uses all observed items.",
 r"m_{igj}=A_{igj}O_{igj},\quad A:\text{administered},\quad O:\text{observed},\quad w:\text{population mass}",
 r"L_{igq}^{(t)}=\prod_{j:m_{igj}=1}P_j(x_q)^{y_{igj}}[1-P_j(x_q)]^{1-y_{igj}},\quad\ell(\psi)=\sum_{g,i}\log\sum_qw_{gq}L_{igq}"),
("3 · Expected counts",r"N_{jgq}^{(t)}=\sum_i m_{igj}\tau_{igq}^{(t)},\qquad R_{jgq}^{(t)}=\sum_{i:m_{igj}=1}y_{igj}\tau_{igq}^{(t)}",
 "Each observed administration contributes posterior fractions across nodes. Missing and absent items contribute zero exposure; they never become incorrect responses. Sum counts across groups for a shared item before fitting its single coefficient vector. These are expected fractional counts, not bins of observed true abilities.",
 r"0\le R_{jgq}\le N_{jgq},\qquad\sum_qN_{jgq}=\#\{\text{observed administrations of item }j\text{ in group }g\}",
 r"N_{jq}=\sum_gN_{jgq},\quad R_{jq}=\sum_gR_{jgq},\quad\sum_qR_{jgq}=\sum_{i:m_{igj}=1}y_{igj}"),
("4 · Joint numerical item M-step",r"\beta_j^{(t+1)}\approx\arg\max_{\beta_j\in\mathcal C_j}\left\{\sum_{g,q}[R_{jgq}\log p_{jq}+(N_{jgq}-R_{jgq})\log(1-p_{jq})]+\log\pi_j(\beta_j)\right\}",
 "Keep expected counts fixed while optimizing a,b,c together numerically. There are no independent closed-form a/b/c updates. Whole bank items are locked; common IDs have one shared vector. With all item priors off the target is MMLE. Enabled priors change it to Bayes modal estimation. The raw marginal likelihood is always reported separately.",
 r"\beta_j=(a_j,b_j,c_j),\quad e_{jgq}=R_{jgq}-N_{jgq}p_{jq},\quad\mathcal C_j:\text{model and bank constraints}",
 r"S_v=\sum_{g,q}\frac{e_{jgq}}{p_{jq}(1-p_{jq})}\frac{\partial p_{jq}}{\partial v}"),
("5 · Population update",r"\widetilde w_{gq}^{(t+1)}=\frac{1}{n_g}\sum_i\tau_{igq}^{(t)},\quad\mu_g=\sum_q\widetilde w_{gq}x_q,\quad\sigma_g^2=\sum_q\widetilde w_{gq}(x_q-\mu_g)^2",
 "Histogram mode uses posterior-average masses. Reference groups rescale their histogram toward fixed moments by interpolation on the fixed grid; free bank-scale groups do not rescale. Free normal groups regenerate normalized normal masses using these moments. Fixed normal reference groups retain their initial masses. This moment rule is the reference algorithm, not a claim of exact maximization for every discretized Gaussian objective.",
 r"\widetilde w:\text{raw posterior average},\quad w^{(t+1)}:\text{masses actually used next cycle}",
 r"w_{gq}^{(t+1)}\propto\phi(x_q;\mu_g,\sigma_g^2)\quad\text{for a free normal group}"),
("6 · Stopping and reference agreement",r"\Delta_{\mathrm{item}}=\max_{j,k\,\mathrm{free}}|\beta_{jk}^{(t+1)}-\beta_{jk}^{(t)}|,\quad\Delta_w=\max_{g,q}|w_{gq}^{(t+1)}-w_{gq}^{(t)}|",
 "Item stability, density stability, finite-sample recovery and software agreement answer different questions. The default stopping rule uses item changes and optimizer success; a stricter density rule is an app option. All-fixed fits check mean/variance changes. OEM completes one cycle and makes no iterative convergence claim. Convergence never proves identification or a global optimum.",
 r"|\widehat\beta_{jk}^{\mathrm{Python}}-\widehat\beta_{jk}^{\mathrm{irtQ}}|\le\operatorname{atol}_k+\operatorname{rtol}_k|\widehat\beta_{jk}^{\mathrm{irtQ}}|",
 r"\ell_{\mathrm{display}}^{(t+1)}=\ell(\beta^{(t+1)},w^{(t+1)})\quad\text{uses the full updated state}")]
SCORES = [
r"\frac{\partial p}{\partial a}=(1-c)D(x-b)s(1-s),\quad\frac{\partial p}{\partial b}=-(1-c)Da\,s(1-s),\quad\frac{\partial p}{\partial c}=1-s",
r"S_a=\sum_{g,q}D(x_q-b)s_q\frac{R_{gq}-N_{gq}p_q}{p_q},\quad S_b=-\sum_{g,q}Da\,s_q\frac{R_{gq}-N_{gq}p_q}{p_q},\quad S_c=\sum_{g,q}\frac{R_{gq}-N_{gq}p_q}{p_q(1-c)}",
r"\log\pi_a=-\log(Da)-\frac{[\log(Da)-\mu_a]^2}{2\sigma_a^2}+C_a,\quad\log\pi_b=-\frac{(b-\mu_b)^2}{2\sigma_b^2}+C_b",
r"\log\pi_c=(\alpha_c-1)\log c+(\beta_c-1)\log(1-c)+C_c",
r"S_a^{\mathrm{pen}}=S_a-I_a\frac{1+[\log(Da)-\mu_a]/\sigma_a^2}{a},\quad S_b^{\mathrm{pen}}=S_b-I_b\frac{b-\mu_b}{\sigma_b^2},\quad S_c^{\mathrm{pen}}=S_c+I_c\left[\frac{\alpha_c-1}{c}-\frac{\beta_c-1}{1-c}\right]",
r"\theta_F=\mu_F+\frac{\sigma_F}{\sigma_T}(\theta_T-\mu_T),\quad a_F=a_T\frac{\sigma_T}{\sigma_F},\quad b_F=\mu_F+\frac{\sigma_F}{\sigma_T}(b_T-\mu_T),\quad c_F=c_T"]
NOTATION = {"g, i":"Group and person within group", "j, q":"Global item ID and quadrature node", "A, O, m":"Administration, observation, and their combined response mask", "y":"Observed 0/1 response", "θ, x":"Unobserved ability and integration node", "w":"Group population masses; nonnegative and sum to one", "τ":"One person's posterior masses", "N, R":"Expected exposure and expected correct counts", "a, b, c":"Discrimination, difficulty, lower asymptote (irtQ names c as g)", "D":"Positive logistic scale constant, not number of dimensions", "ψ(t), β":"Full input state at cycle t and item coefficients", "I_a, I_b, I_c":"Indicators that each item prior is enabled", "μ, σ":"Population mean and SD; prior hyperparameters are separately labeled"}
