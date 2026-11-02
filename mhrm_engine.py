"""Three-stage MH-RM for a confirmatory binary compensatory 2PL model.

Scope: complete 0/1 responses, fixed latent N(0, I), Q-constrained positive
slopes in [0, 10], and free item intercepts. This is a Python implementation
of the staged MH-RM/NR1 strategy, not a port of all features of R mirt.

References consulted:
Cai (2010), doi:10.1007/s11336-009-9136-x.
https://github.com/philchalmers/mirt/blob/main/R/MHRM.group.R
https://github.com/philchalmers/mirt/blob/main/R/MHRM.utils.R
https://philchalmers.github.io/mirt/reference/mirt.html
"""

from collections import deque
from itertools import product

import numpy as np
from scipy.special import expit, logsumexp, ndtri
from scipy.stats import qmc


def sigmoid(x):
    return expit(x)


def prob_m2pl(theta, a, d):
    return expit(np.asarray(theta) @ np.asarray(a).T + np.asarray(d))


def log_likelihood(y, theta, a, d):
    """Conditional response log likelihood; excludes the latent prior."""
    logits = np.asarray(theta) @ np.asarray(a).T + np.asarray(d)
    return float(np.sum(np.asarray(y) * logits - np.logaddexp(0., logits)))


def validate_inputs(data, dims, q_matrix):
    y, q = np.asarray(data, dtype=float), np.asarray(q_matrix, dtype=float)
    if y.ndim != 2 or min(y.shape) < 1:
        raise ValueError('Responses must be a nonempty N by J matrix.')
    if not isinstance(dims, (int, np.integer)) or dims < 1 or q.shape != (y.shape[1], dims):
        raise ValueError('Q must be a J by D matrix matching a positive dimension count.')
    if not np.isin(y, [0, 1]).all() or not np.isin(q, [0, 1]).all():
        raise ValueError('This estimator requires complete binary 0/1 responses and a binary Q-matrix.')
    if np.any(q.sum(axis=0) == 0) or np.any(q.sum(axis=1) == 0):
        raise ValueError('Every factor and item must have at least one active loading.')
    if np.any(y.min(axis=0) == y.max(axis=0)):
        raise ValueError('Every item needs both response categories; constant items cannot be calibrated.')
    return y.copy(), q.astype(bool)


def identification_issues(q_matrix):
    """Conservative structural checks, not a general identification proof."""
    q = np.asarray(q_matrix, dtype=bool)
    issues = []
    if q.ndim != 2:
        return ['The loading mask must be a matrix.']
    if np.any(q.sum(axis=0) < 3):
        issues.append('The supported designs require at least three indicators per factor.')
    for k in range(q.shape[1]):
        for l in range(k + 1, q.shape[1]):
            if np.array_equal(q[:, k], q[:, l]):
                issues.append(f'Factors {k+1} and {l+1} have identical loading masks and cannot have separately identified axes.')
    return issues


def simulate_data(N, J, D, q_matrix, true_a=None, true_d=None, seed=42):
    """Use a local RNG so simulation never resets the estimator's random stream."""
    q = np.asarray(q_matrix, dtype=float)
    if q.shape != (J, D) or not np.isin(q, [0, 1]).all():
        raise ValueError('Q must be a binary J by D matrix.')
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(N, D))
    a = rng.uniform(.5, 2., (J, D)) * q if true_a is None else np.asarray(true_a, dtype=float)
    d = rng.uniform(-2., 2., J) if true_d is None else np.asarray(true_d, dtype=float)
    if a.shape != (J, D) or d.shape != (J,) or not np.isfinite(a).all() or not np.isfinite(d).all():
        raise ValueError('True item coefficients have invalid shapes or nonfinite values.')
    if np.any(a[q == 0] != 0):
        raise ValueError('True slopes must respect the Q-matrix zeros.')
    y = (rng.random((N, J)) < prob_m2pl(theta, a, d)).astype(int)
    return y, theta, a.copy(), d.copy()


def marginal_loglikelihood(data, a, d, points_per_dim=31, qmc_draws=4096, seed=17):
    """Integrate out N(0,I) abilities; return value and integration description.

    Uses product Gauss-Hermite quadrature for D<=2 and scrambled Sobol QMC
    for D>2. The latter is an integration approximation, not a convergence test.
    Complete response patterns are collapsed and evaluated in bounded batches.
    """
    y, a, d = np.asarray(data), np.asarray(a), np.asarray(d)
    if a.ndim != 2 or d.shape != (a.shape[0],) or y.ndim != 2 or y.shape[1] != len(d):
        raise ValueError('Response and coefficient dimensions do not agree.')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(a).all() or not np.isfinite(d).all():
        raise ValueError('Marginal likelihood requires complete binary responses and finite coefficients.')
    dims = a.shape[1]
    if dims <= 2:
        if points_per_dim < 3:
            raise ValueError('Use at least three Gauss-Hermite points per dimension.')
        abscissas, weights = np.polynomial.hermite.hermgauss(points_per_dim)
        indices = np.array(list(product(range(points_per_dim), repeat=dims)))
        nodes = np.sqrt(2.) * abscissas[indices]
        log_weights = np.sum(np.log(weights[indices] / np.sqrt(np.pi)), axis=1)
        method = f'Gauss-Hermite ({points_per_dim} points per dimension)'
    else:
        if qmc_draws < 2:
            raise ValueError('Use at least two integration draws.')
        exponent = int(np.ceil(np.log2(qmc_draws)))
        uniform = qmc.Sobol(dims, scramble=True, seed=seed).random_base2(exponent)
        nodes = ndtri(np.clip(uniform, np.finfo(float).eps, 1 - np.finfo(float).eps))
        log_weights = np.full(len(nodes), -np.log(len(nodes)))
        method = f'Sobol QMC approximation ({len(nodes)} nodes)'
    normalization = np.logaddexp(0., nodes @ a.T + d).sum(axis=1)
    patterns, counts = np.unique(y, axis=0, return_counts=True)
    total = 0.
    for start in range(0, len(patterns), 128):
        batch = patterns[start:start+128]
        conditional = (batch @ a) @ nodes.T + (batch @ d)[:, None] - normalization
        total += float(counts[start:start+128] @ logsumexp(conditional + log_weights, axis=1))
    return total, method


class MHRM_MIRT:
    """MH sampling, constant-gain SEM averaging, then smoothed-curvature RM.

    Complete-data item Hessians are block diagonal when the latent distribution
    is fixed. Each block includes *all* free slopes and the item intercept.
    These blocks are not observed-data information or standard-error estimates.
    """

    def __init__(self, data, dims, q_matrix, seed=101, check_identification=True):
        self.Y, self.Q = validate_inputs(data, dims, q_matrix)
        self.N, self.J = self.Y.shape
        self.D = dims
        self.identification_warnings = identification_issues(self.Q)
        if check_identification and self.identification_warnings:
            raise ValueError(' '.join(self.identification_warnings))
        self.rng = np.random.default_rng(seed)
        self.est_a = self.rng.uniform(.5, 1., (self.J, self.D)) * self.Q
        self.est_d = np.zeros(self.J)
        self.current_theta = np.zeros((self.N, self.D))
        self.active = [np.flatnonzero(row) for row in self.Q]
        self.proposal_sd = .5
        self._reset_history()

    def _reset_history(self):
        self.loss_history, self.iteration_history = [], []
        self.diagnostics = []
        self.converged = False
        self.n_iter_ = self.rm_iterations_ = 0
        self.curvature_ = None
        self.sem_parameter_mean_ = None
        self.sem_curvature_mean_ = None
        self.rm_parameter_history = []
        self.rm_cycle_history = []
        self.rm_gain_history = []
        # Full-run trajectory: one (J, D+1) snapshot per total cycle, starting
        # with the random starting values at cycle 0, across all three stages.
        self.parameter_history = []
        self.total_cycle_history = []
        self.stage_history = []
        self.stage_cycle_history = []
        self.gain_history = []

    def _record(self, total_cycle, stage, stage_cycle, gain):
        self.parameter_history.append(np.column_stack((self.est_a, self.est_d)))
        self.total_cycle_history.append(total_cycle)
        self.stage_history.append(stage)
        self.stage_cycle_history.append(stage_cycle)
        self.gain_history.append(float(gain))

    def _row_log_posterior(self, theta):
        logits = theta @ self.est_a.T + self.est_d
        ll = np.sum(self.Y * logits - np.logaddexp(0., logits), axis=1)
        return ll - .5 * np.sum(theta**2, axis=1)

    def mh_step(self, n_steps=5, candidate_sd=None):
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
            raise ValueError('MH steps must be a positive integer.')
        sd = self.proposal_sd if candidate_sd is None else float(candidate_sd)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError('MH proposal SD must be positive and finite.')
        accepted = 0
        current = self._row_log_posterior(self.current_theta)
        for _ in range(n_steps):
            proposal = self.current_theta + self.rng.normal(0., sd, (self.N, self.D))
            proposed = self._row_log_posterior(proposal)
            log_uniform = np.log(np.maximum(self.rng.random(self.N), np.finfo(float).tiny))
            mask = log_uniform < np.minimum(0., proposed - current)
            self.current_theta[mask] = proposal[mask]
            current[mask] = proposed[mask]
            accepted += int(mask.sum())
        return accepted / (n_steps * self.N)

    def complete_data_derivatives(self, theta=None):
        """Return per-item score and full positive curvature, both divided by N."""
        theta = self.current_theta if theta is None else np.asarray(theta)
        probabilities = prob_m2pl(theta, self.est_a, self.est_d)
        residual = self.Y - probabilities
        scores, information = [], []
        for j, active in enumerate(self.active):
            design = np.column_stack((theta[:, active], np.ones(self.N)))
            w = probabilities[:, j] * (1 - probabilities[:, j])
            scores.append(design.T @ residual[:, j] / self.N)
            information.append((design.T * w) @ design / self.N)
        return scores, information

    def _parameters(self):
        return [np.r_[self.est_a[j, active], self.est_d[j]] for j, active in enumerate(self.active)]

    def _set_parameters(self, parameters):
        self.est_a[:] = 0.
        for j, (active, beta) in enumerate(zip(self.active, parameters)):
            self.est_a[j, active] = beta[:-1]
            self.est_d[j] = beta[-1]

    @staticmethod
    def _direction(information, score):
        # Numerical stabilization of a solve, not a parameter prior/penalty.
        smallest = np.linalg.eigvalsh(information)[0]
        stabilized = information + np.eye(len(score)) * max(0., 1e-6 - smallest)
        return np.linalg.solve(stabilized, score)

    def _update(self, scores, information, gain, clip_correction=False):
        old, new, directions = self._parameters(), [], []
        for beta, score, block in zip(old, scores, information):
            direction = self._direction(block, score)
            directions.append(direction)
            if clip_correction:
                direction = np.clip(direction, -1., 1.)
            candidate = beta + gain * direction
            # mirt-style halfway-to-bound adjustment keeps slopes inside bounds.
            for k in range(len(beta) - 1):
                if candidate[k] < 0:
                    candidate[k] = beta[k] / 2
                elif candidate[k] > 10:
                    candidate[k] = (beta[k] + 10) / 2
            new.append(candidate)
        self._set_parameters(new)
        if not np.isfinite(self.est_a).all() or not np.isfinite(self.est_d).all():
            raise FloatingPointError('The parameter update produced nonfinite coefficients.')
        return max(np.max(np.abs(b - a)) for a, b in zip(old, new)), directions

    def rm_step(self, learning_rate, scores=None, information=None):
        """One RM step using recursively smoothed full item curvature blocks."""
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError('RM gain must be in (0, 1].')
        if scores is None or information is None:
            scores, information = self.complete_data_derivatives()
        if self.curvature_ is None:
            self.curvature_ = [block.copy() for block in information]
        else:
            self.curvature_ = [old + learning_rate * (new - old)
                               for old, new in zip(self.curvature_, information)]
        return self._update(scores, self.curvature_, learning_rate)

    def fit(self, max_iter=500, burn_in=150, decay=.75, progress_bar=None,
            sem_cycles=100,
            mh_steps=5, gain_scale=.1, tolerance=.001, score_tolerance=.005,
            min_rm_cycles=50, callback=None):
        """max_iter caps stage-3 RM cycles *in addition* to the first two stages.

        Like mirt's NR1 mode: stages 1/2 use bounded constant-gain Newton
        corrections; stage 2 averages parameters and curvature; stage 3 uses
        gain=(gain_scale/t)**decay and recursively averaged curvature. Adapt
        proposal SD in burn-in only. After min_rm_cycles, require 3 consecutive
        small updates AND a small 30-cycle mean projected score for convergence.
        """
        for name, number in [('max_iter', max_iter), ('burn_in', burn_in), ('sem_cycles', sem_cycles), ('mh_steps', mh_steps), ('min_rm_cycles', min_rm_cycles)]:
            if not isinstance(number, (int, np.integer)) or number < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if not .5 < decay <= 1 or not 0 < gain_scale <= 1:
            raise ValueError('Use decay in (0.5,1] and gain_scale in (0,1].')
        if not np.isfinite([tolerance, score_tolerance]).all() or min(tolerance, score_tolerance) <= 0:
            raise ValueError('Convergence tolerances must be finite and positive.')
        self._reset_history()
        total = burn_in + sem_cycles + max_iter
        parameter_sum = [np.zeros_like(v) for v in self._parameters()]
        curvature_sum = [np.zeros((len(v), len(v))) for v in parameter_sum]
        score_window = deque(maxlen=30)
        stable = 0
        self._record(0, 'Start', 0, 0.)
        for iteration in range(1, total + 1):
            if iteration <= burn_in:
                stage, stage_cycle, gain = 'Burn-in', iteration, .2
            elif iteration <= burn_in + sem_cycles:
                stage, stage_cycle, gain = 'SEM', iteration - burn_in, .2
            else:
                stage_cycle = iteration - burn_in - sem_cycles
                stage, gain = 'RM', (gain_scale / stage_cycle)**decay
            acceptance = self.mh_step(mh_steps)
            if stage == 'Burn-in':
                adjustment = (acceptance - .3) / np.sqrt(iteration)
                self.proposal_sd = float(np.clip(self.proposal_sd * np.exp(adjustment), .05, 3.))
            scores, information = self.complete_data_derivatives()
            if stage != 'RM':
                change, _ = self._update(scores, information, gain, clip_correction=True)
                if stage == 'SEM':
                    for j, beta in enumerate(self._parameters()):
                        parameter_sum[j] += beta
                        curvature_sum[j] += information[j]
                    if stage_cycle == sem_cycles:
                        self.sem_parameter_mean_ = [v / sem_cycles for v in parameter_sum]
                        self.sem_curvature_mean_ = [v / sem_cycles for v in curvature_sum]
                        before_average = self._parameters()
                        self._set_parameters(self.sem_parameter_mean_)
                        change = max(change, max(np.max(np.abs(a-b)) for a, b in
                                     zip(before_average, self.sem_parameter_mean_)))
                        self.curvature_ = [v.copy() for v in self.sem_curvature_mean_]
                        # Cycle 0 is the SEM-averaged starting point for RM.
                        self.rm_parameter_history.append(np.column_stack((self.est_a, self.est_d)))
                        self.rm_cycle_history.append(0)
                        self.rm_gain_history.append(0.)
            else:
                change, _ = self.rm_step(gain, scores, information)
                self.rm_iterations_ = stage_cycle
                # Copies preserve the actual post-update values for every item.
                self.rm_parameter_history.append(np.column_stack((self.est_a, self.est_d)))
                self.rm_cycle_history.append(stage_cycle)
                self.rm_gain_history.append(float(gain))
                score_window.append(np.concatenate(scores))
                mean_score = np.mean(score_window, axis=0)
                beta = np.concatenate(self._parameters())
                lower = np.concatenate([np.r_[np.zeros(len(v)), -np.inf] for v in self.active])
                upper = np.concatenate([np.r_[np.full(len(v), 10.), np.inf] for v in self.active])
                # Bound-constrained stationarity: discard outward-pointing scores.
                projected = mean_score.copy()
                projected[(beta <= lower + 1e-6) & (mean_score < 0)] = 0.
                projected[(beta >= upper - 1e-6) & (mean_score > 0)] = 0.
                small = (stage_cycle >= min_rm_cycles and len(score_window) == 30
                         and change < tolerance and np.max(np.abs(projected)) < score_tolerance)
                stable = stable + 1 if small else 0
                self.converged = stable >= 3
            self.n_iter_ = iteration
            # Post-update values (for the last SEM cycle: after averaging).
            self._record(iteration, stage, stage_cycle, gain)
            conditional_ll = log_likelihood(self.Y, self.current_theta, self.est_a, self.est_d)
            diagnostic = dict(iteration=iteration, stage=stage, stage_cycle=stage_cycle,
                              gain=gain, acceptance=acceptance, proposal_sd=self.proposal_sd,
                              max_change=change, conditional_loglik=conditional_ll)
            if stage == 'RM':
                diagnostic['mean_projected_score'] = float(np.max(np.abs(projected)))
            self.diagnostics.append(diagnostic)
            if iteration == 1 or iteration % 5 == 0 or self.converged or iteration == total:
                self.loss_history.append(conditional_ll)
                self.iteration_history.append(iteration)
                if progress_bar is not None:
                    progress_bar.progress(iteration / total,
                        text=f'{stage} {stage_cycle}: Δmax={change:.4f}, MH acceptance={acceptance:.2f}, gain={gain:.4f}')
                if callback is not None:
                    callback(self, diagnostic)
            if self.converged:
                break
        self.stop_reason_ = ('Convergence criteria met' if self.converged else
                             'RM cycle limit reached without convergence')
        if progress_bar is not None:
            progress_bar.progress(1., text=self.stop_reason_)
        return self
