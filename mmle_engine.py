"""Independent, missingness-aware MMLE/Bayes-modal EM teaching engine.

Parameterization and density schedules follow irtQ 1.4.0 source review;
SciPy performs joint direct-coefficient M-steps. No R fitting is used here.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from scipy.stats import norm, lognorm, beta as beta_dist, skewnorm

VERSION = "0.1.0"
DESIGNS = ("Standard", "Concurrent", "Fixed bank anchors", "Concurrent + fixed bank anchors")


def jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def signature(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Priors:
    a: bool = False
    b: bool = False
    c: bool = False
    a_meanlog: float = 0.0
    a_sdlog: float = .5
    b_mean: float = 0.0
    b_sd: float = 1.0
    c_alpha: float = 5.0
    c_beta: float = 16.0

    def validate(self):
        vals = np.array([self.a_meanlog, self.a_sdlog, self.b_mean, self.b_sd, self.c_alpha, self.c_beta])
        if not np.isfinite(vals).all() or min(self.a_sdlog, self.b_sd, self.c_alpha, self.c_beta) <= 0:
            raise ValueError("Prior hyperparameters must be finite; SDs and Beta shapes must be positive.")


@dataclass
class FitConfig:
    design: str = "Standard"
    model: str = "2PL"
    density: str = "Normal"
    D: float = 1.0
    nodes: int = 49
    extent: float = 6.0
    mean: float = 0.0
    sd: float = 1.0
    max_cycles: int = 500
    tolerance: float = 1e-4
    density_tolerance: float = 1e-5
    strict_density_stop: bool = False
    schedule: str = "MEM"
    positive_a: bool = False
    start_a: float = 1.0
    start_b: float = 0.0
    start_c: float = .15
    priors: Priors = field(default_factory=Priors)
    # Optional common initial masses; all groups start with the same distribution,
    # matching the public irtQ weights argument. Nodes remain equally spaced.
    initial_weights: list | None = None

    @property
    def anchored(self):
        return self.design in DESIGNS[2:]

    def validate(self):
        self.priors.validate()
        if self.design not in DESIGNS or self.model not in ("2PL", "3PL") or self.density not in ("Normal", "Histogram"):
            raise ValueError("Unknown calibration design, model or density.")
        if not np.isfinite([self.D, self.extent, self.mean, self.sd, self.tolerance, self.density_tolerance, self.start_a, self.start_b, self.start_c]).all():
            raise ValueError("Fit settings must be finite.")
        if min(self.D, self.sd, self.extent, self.tolerance, self.density_tolerance) <= 0 or self.nodes < 5 or self.max_cycles < 1:
            raise ValueError("Positive scaling, SD, tolerances, extent and cycle count are required (at least five nodes).")
        if self.schedule not in ("OEM", "MEM") or self.start_a <= 0 or not 0 < self.start_c < 1:
            raise ValueError("Use positive starting a, 0 < starting c < 1, and OEM or MEM.")
        if self.model == "2PL" and self.priors.c:
            raise ValueError("Guessing priors are disabled for 2PL.")
        if self.initial_weights is not None:
            w=np.asarray(self.initial_weights,float)
            if w.shape!=(self.nodes,) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
                raise ValueError("Custom initial masses must match the node count, be finite/nonnegative and have positive total.")



@dataclass
class Dataset:
    responses: np.ndarray
    administered: np.ndarray
    group: np.ndarray
    item_ids: list[str]
    group_names: list[str]
    bank: np.ndarray  # one row per global ID, NaN for free items
    truth: np.ndarray | None = None
    theta: np.ndarray | None = None
    generation: dict = field(default_factory=dict)

    @property
    def observed(self):
        return self.administered & np.isfinite(self.responses)

    @property
    def fixed(self):
        return np.isfinite(self.bank).all(axis=1)

    @property
    def identity(self):
        return signature(dict(y=self.responses, admin=self.administered, group=self.group,
                              items=self.item_ids, groups=self.group_names))

    def validate(self):
        y, a = self.responses, self.administered
        if y.ndim != 2 or a.shape != y.shape or self.group.shape != (len(y),):
            raise ValueError("Response, administration and group arrays have inconsistent shapes.")
        if len(set(self.item_ids)) != y.shape[1] or len(self.item_ids) != y.shape[1]:
            raise ValueError("Global item IDs must be unique; common IDs use one global column.")
        if len(set(self.group_names)) != len(self.group_names) or set(self.group) != set(range(len(self.group_names))):
            raise ValueError("Every named group must have people and a unique name.")
        if self.bank.shape != (y.shape[1], 3) or np.any(np.isfinite(self.bank).any(axis=1) != self.fixed):
            raise ValueError("Supply all a,b,c bank coefficients or leave all three blank for a free item.")
        if np.any(np.isfinite(y) & ~a):
            raise ValueError("An unadministered cell cannot contain a response.")
        if np.any(~np.isin(y[self.observed], [0, 1])) or np.isinf(y).any():
            raise ValueError("Observed responses must be 0 or 1; use blank/NA for missing.")
        if np.any(self.observed.sum(axis=0) == 0):
            raise ValueError("Every item needs at least one observed administration.")
        if np.any((self.bank[self.fixed, 2] < 0) | (self.bank[self.fixed, 2] >= 1)):
            raise ValueError("Bank c must lie in [0,1).")


def normal_weights(x, mean=0., sd=1.):
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("Population SD must be positive and finite.")
    lp = norm.logpdf(x, mean, sd)
    return np.exp(lp - logsumexp(lp))


def moments(x, w):
    mu = np.sum(w * x, axis=-1)
    var = np.sum(w * (x - np.expand_dims(mu, -1)) ** 2, axis=-1)
    return np.stack([mu, np.sqrt(np.maximum(var, 0))], axis=-1)


def log_probabilities(x, params, D=1.):
    params = np.atleast_2d(params)
    a, b, c = params.T
    z = D * a[:, None] * (np.asarray(x)[None, :] - b[:, None])
    logs = -np.logaddexp(0., -z)
    log1s = -np.logaddexp(0., z)
    with np.errstate(divide="ignore"):
        lp = np.logaddexp(np.log(c[:, None]), np.log1p(-c[:, None]) + logs)
    l1p = np.log1p(-c[:, None]) + log1s
    return lp, l1p


def probabilities(x, params, D=1.):
    return np.exp(log_probabilities(x, params, D)[0])


def log_prior(params, priors, D=1., model="3PL", gradient=False):
    a, b, c = params
    value, score = 0., np.zeros(3)
    if priors.a:
        if a <= 0:
            return (-np.inf, score) if gradient else -np.inf
        value += lognorm.logpdf(D*a, s=priors.a_sdlog, scale=np.exp(priors.a_meanlog))
        score[0] -= (1 + (np.log(D*a)-priors.a_meanlog)/priors.a_sdlog**2)/a
    if priors.b:
        value += norm.logpdf(b, priors.b_mean, priors.b_sd)
        score[1] -= (b-priors.b_mean)/priors.b_sd**2
    if priors.c and model == "3PL":
        if not 0 < c < 1:
            return (-np.inf, score) if gradient else -np.inf
        value += beta_dist.logpdf(c, priors.c_alpha, priors.c_beta)
        score[2] += (priors.c_alpha-1)/c - (priors.c_beta-1)/(1-c)
    return (float(value), score) if gradient else float(value)


def item_objective(params, x, N, R, config):
    """Return negative expected log likelihood plus prior and its direct a/b/c gradient."""
    a, b, c = params
    lp, l1p = log_probabilities(x, params, config.D)
    lp, l1p = lp[0], l1p[0]
    q = np.sum(R*lp + (N-R)*l1p)
    s = expit(config.D*a*(x-b))
    if config.model == "2PL":
        e = R-N*s
        score = np.array([np.sum(config.D*(x-b)*e), np.sum(-config.D*a*e), 0.])
    else:
        # Stable simplified scores. s/p stays bounded by 1/(1-c).
        ratio = np.exp(-np.logaddexp(0., -config.D*a*(x-b))-lp)
        scaled_e = R*ratio - N*s
        score = np.array([np.sum(config.D*(x-b)*scaled_e), np.sum(-config.D*a*scaled_e),
                          np.sum(R*np.exp(np.minimum(-lp, 690.))-N)/(1-c)])
    prior, pscore = log_prior(params, config.priors, config.D, config.model, True)
    return -float(q+prior), -(score+pscore)


def estep(data, params, weights, x, D, likelihood_items=None):
    mask = data.observed.copy()
    if likelihood_items is not None:
        mask &= np.asarray(likelihood_items)[None, :]
    y = np.where(mask, data.responses, 0.)
    lp, l1p = log_probabilities(x, params, D)
    ll = y @ lp + (mask-y) @ l1p
    with np.errstate(divide="ignore"):
        joint = ll + np.log(weights[data.group])
    marginal = logsumexp(joint, axis=1)
    posterior = np.exp(joint-marginal[:, None])
    return posterior, ll, marginal


def expected_counts(data, posterior):
    G, J, Q = len(data.group_names), len(data.item_ids), posterior.shape[1]
    N, R = np.zeros((G, J, Q)), np.zeros((G, J, Q))
    for g in range(G):
        ix = data.group == g
        m = data.observed[ix]
        N[g] = m.T @ posterior[ix]
        R[g] = np.where(m, data.responses[ix], 0.).T @ posterior[ix]
    return N, R


def rescale_histogram(x, weights, target_mean, target_sd):
    """Woods interpolation/exponential-tail rule used by irtQ scale_prior.R."""
    mu, sd = moments(x, weights)
    if sd < 1e-12:
        raise ValueError("Histogram collapsed to a point; use a wider grid or informative data.")
    transformed = (x-mu)/sd*target_sd+target_mean
    freq = np.maximum(weights, 1e-20)
    delta = transformed[1]-transformed[0]
    out = np.interp(x, transformed, freq)
    left, right = x <= transformed[0], x >= transformed[-1]
    out[left] = np.exp(np.clip(np.log(freq[0])+(transformed[0]-x[left])/delta*np.log(freq[0]/freq[1]), -745, 700))
    out[right] = np.exp(np.clip(np.log(freq[-1])+(x[right]-transformed[-1])/delta*np.log(freq[-1]/freq[-2]), -745, 700))
    return out/out.sum()


def check_design(data, config):
    data.validate()
    config.validate()
    G = len(data.group_names)
    if (config.design in (DESIGNS[0], DESIGNS[2])) != (G == 1):
        raise ValueError("Use a single-group design for one group, or a concurrent design for multiple groups.")
    if config.anchored and not data.fixed.any():
        raise ValueError("An anchored design needs at least one whole-item bank anchor.")
    if not config.anchored and data.fixed.any():
        raise ValueError("Bank coefficients are present: choose an anchored design or clear the bank locks.")
    if config.anchored and not any(len(np.unique(data.responses[data.observed[:,j],j])) > 1 for j in np.flatnonzero(data.fixed)):
        raise ValueError("All bank-anchor responses are constant; these data do not provide anchor response variation for calibration.")
    if config.model == "2PL" and np.any(data.bank[data.fixed, 2] != 0):
        raise ValueError("2PL bank items must have c=0.")
    present = np.array([data.observed[data.group == g].any(axis=0) for g in range(G)])
    roots = set(np.flatnonzero((present & data.fixed).any(axis=1))) if config.anchored else {0}
    connected = set(roots)
    while True:
        more = {g for g in range(G) if any(np.any(present[g] & present[h]) for h in connected)}
        if more <= connected:
            break
        connected |= more
    if len(connected) != G:
        names = [data.group_names[g] for g in range(G) if g not in connected]
        raise ValueError("No observed item path to a reference/bank scale for: " + ", ".join(names))
    return "Necessary scale connections present; connectivity alone does not prove identification or adequate information."


@dataclass
class Cycle:
    index: int
    input_params: np.ndarray
    input_weights: np.ndarray
    params: np.ndarray
    weights: np.ndarray
    raw_weights: np.ndarray
    expected_N: np.ndarray
    expected_R: np.ndarray
    likelihood_items: np.ndarray
    loglik: float
    penalized: float
    item_change: float
    weight_change: float
    moment_change: float
    optimizer: list
    q_before: np.ndarray
    q_after: np.ndarray


@dataclass
class FitResult:
    config: FitConfig
    data_id: str
    nodes: np.ndarray
    initial_params: np.ndarray
    initial_weights: np.ndarray
    cycles: list[Cycle]
    stop_reason: str = "Not started"
    elapsed: float = 0.
    run_id: str = ""
    versions: dict = field(default_factory=dict)
    constraints_id: str = ""

    @property
    def params(self):
        return self.cycles[-1].params if self.cycles else self.initial_params

    @property
    def weights(self):
        return self.cycles[-1].weights if self.cycles else self.initial_weights


def initialize(data, config):
    check_design(data, config)
    x = np.linspace(-config.extent, config.extent, config.nodes)
    p = np.tile([config.start_a, config.start_b, config.start_c if config.model == "3PL" else 0.], (len(data.item_ids), 1))
    p[data.fixed] = data.bank[data.fixed]
    w = normal_weights(x, config.mean, config.sd)
    if config.initial_weights is not None:
        w = np.asarray(config.initial_weights, float)
        if w.shape != x.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Initial masses must match the grid, be finite/nonnegative and have positive total.")
        w = w/w.sum()
    import scipy
    return FitResult(deepcopy(config), data.identity, x, p, np.tile(w, (len(data.group_names), 1)), [],
                     run_id=signature([data.identity, config, time.time_ns()]),
                     versions=dict(engine=VERSION, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), constraints_id=signature(data.bank))


def em_cycle(data, config, old, ow, x, used, free_groups, target_mean, target_sd, index):
    """One EM cycle from item coefficients `old` and population masses `ow`: E-step on the
    `used` items, item M-steps on expected counts, then the population update."""
    fixed = data.fixed
    tau, _, _ = estep(data, old, ow, x, config.D, used)
    N, R = expected_counts(data, tau)
    p = old.copy()
    status, qb, qa = [], np.full(len(fixed), np.nan), np.full(len(fixed), np.nan)
    for j in range(len(p)):
        if fixed[j]:
            status.append(dict(item=j, success=True, message="Bank anchor locked", locked=True))
            continue
        nj, rj = N[:, j].sum(axis=0), R[:, j].sum(axis=0)
        nfree = 2 if config.model == "2PL" else 3
        def fun(v):
            candidate = np.r_[v, 0.] if nfree == 2 else v
            val, grad = item_objective(candidate, x, nj, rj, config)
            return val, grad[:nfree]
        lower_a = 1e-10 if config.positive_a or config.priors.a else None
        bounds = [(lower_a, None), (None, None)] + ([(1e-8, 1-1e-8)] if nfree == 3 else [])
        qb[j] = -fun(old[j, :nfree])[0]
        opt = minimize(fun, old[j, :nfree], method="L-BFGS-B", jac=True, bounds=bounds,
                       options=dict(maxiter=200, maxls=40, ftol=1e-12, gtol=1e-6))
        after = -fun(opt.x)[0]
        accepted = np.isfinite(after) and after >= qb[j]-1e-7
        if accepted:
            p[j, :nfree] = opt.x
        qa[j] = after if accepted else qb[j]
        status.append(dict(item=j, success=bool(opt.success and accepted), message=str(opt.message),
                           locked=False, iterations=int(opt.nit), gradient=float(np.max(np.abs(opt.jac))), accepted=bool(accepted)))
    raw = np.array([tau[data.group == g].mean(axis=0) for g in range(len(free_groups))])
    w = ow.copy()
    for g in range(len(w)):
        if config.density == "Histogram":
            w[g] = raw[g] if free_groups[g] else rescale_histogram(x, raw[g], target_mean, target_sd)
        elif free_groups[g]:
            mu, sd = moments(x, raw[g])
            w[g] = normal_weights(x, mu, max(sd, 1e-8))
    ll = float(estep(data, p, w, x, config.D)[2].sum())
    prior_total = sum(log_prior(p[j], config.priors, config.D, config.model) for j in range(len(p)) if not fixed[j])
    dp = float(np.max(np.abs(p[~fixed]-old[~fixed]))) if (~fixed).any() else 0.
    dw = float(np.max(np.abs(w-ow)))
    dm = float(np.max(np.abs(moments(x, w)-moments(x, ow))))
    return Cycle(index, old, ow, p, w, raw, N, R, used, ll, ll+prior_total, dp, dw, dm, status, qb, qa)


def stationarity_step(data, config, params, weights, x):
    """Apply one full EM cycle (all observed items) starting from an endpoint.

    A converged endpoint is (nearly) a fixed point of EM, so the returned item and
    density changes are small; a large change means the endpoint was still moving.
    """
    free_groups = np.ones(len(data.group_names), bool)
    if not config.anchored:
        free_groups[0] = False
    target_mean, target_sd = (moments(x, np.asarray(config.initial_weights)/np.sum(config.initial_weights))
                              if config.initial_weights is not None else (config.mean, config.sd))
    used = np.ones(len(data.item_ids), bool)
    return em_cycle(data, config, np.asarray(params, float), np.asarray(weights, float), x, used,
                    free_groups, target_mean, target_sd, 0)


def fit(data, config, previous=None, cycles=None, callback=None):
    """Fit or resume the same experiment. Stored cycles use input t -> posterior t -> output t+1."""
    check_design(data, config)
    result = deepcopy(previous) if previous is not None else initialize(data, config)
    if result.data_id != data.identity or result.constraints_id != signature(data.bank) or signature(result.config) != signature(config):
        raise ValueError("Resume requires identical saved responses, bank constraints and fitting settings.")
    started = time.monotonic()
    limit = 1 if config.anchored and config.schedule == "OEM" else config.max_cycles
    end = min(limit, len(result.cycles)+(cycles if cycles is not None else limit))
    x, fixed = result.nodes, data.fixed
    free_groups = np.ones(len(data.group_names), bool)
    if not config.anchored:
        free_groups[0] = False
    target_mean, target_sd = moments(x, result.initial_weights[0]) if config.initial_weights is not None else (config.mean, config.sd)
    for t in range(len(result.cycles), end):
        old, ow = result.params.copy(), result.weights.copy()
        used = fixed.copy() if config.anchored and t == 0 else np.ones(len(fixed), bool)
        cycle = em_cycle(data, config, old, ow, x, used, free_groups, target_mean, target_sd, t+1)
        p, w, status, dp, dw = cycle.params, cycle.weights, cycle.optimizer, cycle.item_change, cycle.weight_change
        result.cycles.append(cycle)
        if callback:
            callback(cycle)
        success = all(s["success"] for s in status)
        if config.anchored and config.schedule == "OEM":
            result.stop_reason = "One cycle completed (OEM; returned density not fed back)"
            break
        # irtQ uses mean and variance changes when every item is fixed.
        criterion = dp if (~fixed).any() else float(np.max(np.abs(np.c_[moments(x,w)[:,0],moments(x,w)[:,1]**2]-np.c_[moments(x,ow)[:,0],moments(x,ow)[:,1]**2])))
        if criterion <= config.tolerance and success and (not config.strict_density_stop or dw <= config.density_tolerance):
            result.stop_reason = "Item stopping rule satisfied" if (~fixed).any() else "Population moment stopping rule satisfied"
            if config.strict_density_stop:
                result.stop_reason += "; app density rule satisfied"
            break
        result.stop_reason = "Cycle limit reached" if t+1 == limit else "Paused after recorded cycle"
    result.elapsed += time.monotonic()-started
    result.run_id = signature([data.identity, config, len(result.cycles), time.time_ns()])
    return result


def replay_estep(data, result, cycle_index):
    c = result.cycles[cycle_index]
    return estep(data, c.input_params, c.input_weights, result.nodes, result.config.D, c.likelihood_items)


def align_truth(params, mu_true, sd_true, mu_fit, sd_fit):
    out = np.array(params, float, copy=True)
    out[:, 0] *= sd_true/sd_fit
    out[:, 1] = mu_fit+sd_fit/sd_true*(out[:, 1]-mu_true)
    return out


def recovery(data, result):
    if data.truth is None:
        return None
    cfg = result.config
    truth = data.truth.copy()
    true_groups = np.array([[g["mean"],g["sd"]] for g in data.generation["groups"]])
    if not cfg.anchored:
        mt, st = true_groups[0]
        mf, sf = moments(result.nodes, result.initial_weights[0]) if cfg.initial_weights is not None else (cfg.mean, cfg.sd)
        truth = align_truth(truth, mt, st, mf, sf)
        true_groups[:, 0] = mf+sf/st*(true_groups[:, 0]-mt)
        true_groups[:, 1] *= sf/st
        alignment = f"One shared reference transform: θF={mf:.4g}+({sf:.4g}/{st:.4g})(θT−{mt:.4g})"
    else:
        alignment = "Supplied bank scale retained; deliberate bank drift remains a model mismatch."
    free = ~data.fixed
    delta = result.params[free]-truth[free]
    metrics = [dict(parameter=k, bias=float(np.mean(delta[:,i])), RMSE=float(np.sqrt(np.mean(delta[:,i]**2))))
               for i,k in enumerate(("a","b","c")) if (i<2 or cfg.model=="3PL") and free.any()]
    tw=truth_population_weights(data,result)
    return dict(raw_truth=data.truth, aligned_truth=truth, metrics=metrics, group_truth=true_groups,
                group_estimates=moments(result.nodes,result.weights), alignment=alignment,
                truth_grid_weights=tw,density_TV=np.sum(np.abs(tw-result.weights),axis=1)/2 if tw is not None else None)



def truth_population_weights(data, result):
    """Generating population on the fit grid (continuous tails renormalized).

    Custom point masses are projected linearly onto neighboring fit nodes;
    out-of-grid point masses go to an endpoint. This is a grid diagnostic,
    not a claim that the generating continuous distribution is discrete.
    """
    if data.truth is None or data.generation.get("D") != result.config.D:
        return None
    x=result.nodes
    cfg=result.config
    mt,st_true=data.generation["groups"][0]["mean"],data.generation["groups"][0]["sd"]
    mf,sf=moments(x,result.initial_weights[0]) if cfg.initial_weights is not None else (cfg.mean,cfg.sd)
    slope,shift=(1.,0.) if cfg.anchored else (sf/st_true,mf-sf/st_true*mt)
    output=[]
    for spec in data.generation["groups"]:
        mu,sd=shift+slope*spec["mean"],slope*spec["sd"]
        shape=spec.get("shape","Normal")
        if shape=="Normal": w=normal_weights(x,mu,sd)
        elif shape=="Skewed":
            alpha=5.;delta=alpha/np.sqrt(1+alpha**2);center=delta*np.sqrt(2/np.pi);spread=np.sqrt(1-2*delta**2/np.pi)
            logs=skewnorm.logpdf(x,alpha,loc=mu-sd*center/spread,scale=sd/spread)
            w=np.exp(logs-logsumexp(logs))
        elif shape=="Bimodal":
            logs=np.logaddexp(norm.logpdf(x,mu-sd*1.2/1.3,sd*.5/1.3),norm.logpdf(x,mu+sd*1.2/1.3,sd*.5/1.3))
            w=np.exp(logs-logsumexp(logs))
        else:
            w=np.zeros(len(x))
            for node,mass in zip(shift+slope*np.asarray(spec["nodes"]),np.asarray(spec["weights"])/np.sum(spec["weights"])):
                hi=int(np.searchsorted(x,node))
                if hi==0: w[0]+=mass
                elif hi==len(x): w[-1]+=mass
                else:
                    fraction=(node-x[hi-1])/(x[hi]-x[hi-1]);w[hi]+=mass*fraction;w[hi-1]+=mass*(1-fraction)
        output.append(w)
    return np.array(output)

def simulate(n_per_group=1000, items=20, groups=None, model="2PL", D=1., seed=2026,
             common=8, anchors=0, missing=.0, a_range=(.5,2.), b_range=(-2.,2.), c_range=(.1,.25),
             true_params=None, form_map=None, bank_params=None, drift=0.):
    """Generate once; population presets have their declared population mean/SD."""
    groups = groups or [dict(name="G1", mean=0., sd=1., shape="Normal")]
    G = len(groups)
    if not (D>0 and np.isfinite(D)) or model not in ("2PL","3PL") or not 0<=missing<1 or n_per_group<2 or items<2:
        raise ValueError("Invalid generation model, scaling, missingness, people or item count.")
    if min(a_range)<=0 or a_range[0]>a_range[1] or b_range[0]>b_range[1] or not 0<=c_range[0]<=c_range[1]<1:
        raise ValueError("Use positive generating slopes and ordered valid parameter ranges.")
    rng = np.random.default_rng(seed)
    common = min(common, items)
    J = items if G == 1 else common+G*(items-common)
    admin_map = np.zeros((G,J),bool)
    for g in range(G):
        admin_map[g,:common] = True
        if G==1:
            admin_map[g,:] = True
        else:
            admin_map[g,common+g*(items-common):common+(g+1)*(items-common)] = True
    if form_map is not None:
        admin_map = np.asarray(form_map, bool)
        if admin_map.shape[0] != G or admin_map.ndim != 2:
            raise ValueError("Administration map needs one row per group/form.")
        J = admin_map.shape[1]
    p = np.column_stack([rng.uniform(*a_range,J),rng.uniform(*b_range,J),rng.uniform(*c_range,J) if model=="3PL" else np.zeros(J)])
    if true_params is not None:
        p = np.asarray(true_params,float)
    if p.shape!=(J,3) or not np.isfinite(p).all() or (p[:,0]<=0).any() or (p[:,2]<0).any() or (p[:,2]>=1).any() or (model=="2PL" and np.any(p[:,2]!=0)):
        raise ValueError("Truth requires finite a,b,c per item, a>0 and valid c (zero for 2PL).")
    ys, ad, gr, abilities, saved_groups = [],[],[],[],[]
    for g,spec in enumerate(groups):
        spec = dict(spec)
        mean,sd = float(spec.get("mean",0)),float(spec.get("sd",1))
        if not np.isfinite([mean,sd]).all() or sd<=0:
            raise ValueError("True group moments must be finite with positive SD.")
        shape = spec.get("shape","Normal")
        if shape=="Normal":
            z = rng.normal(size=n_per_group)
        elif shape=="Skewed":
            alpha=5.; delta=alpha/np.sqrt(1+alpha**2)
            z=(skewnorm.rvs(alpha,size=n_per_group,random_state=rng)-delta*np.sqrt(2/np.pi))/np.sqrt(1-2*delta**2/np.pi)
        elif shape=="Bimodal":
            z=(rng.choice([-1.,1.],n_per_group)*1.2+rng.normal(0,.5,n_per_group))/1.3
        elif shape=="Custom":
            nodes, mass = np.asarray(spec["nodes"],float), np.asarray(spec["weights"],float)
            if nodes.ndim!=1 or nodes.shape!=mass.shape or not np.isfinite(nodes).all() or not np.isfinite(mass).all() or (mass<0).any() or mass.sum()<=0:
                raise ValueError("Custom nodes/masses must be finite and nonnegative with positive total.")
            mass=mass/mass.sum(); cm,cs=moments(nodes,mass)
            if cs<=0: raise ValueError("Custom ability nodes need positive variance.")
            # Custom nodes are used literally; declared moments become their exact moments.
            theta=rng.choice(nodes,n_per_group,p=mass)
            mean,sd=float(cm),float(cs)
            spec.update(mean=mean,sd=sd,weights=mass.tolist())
            z=None
        else:
            raise ValueError("Unknown generating ability shape.")
        theta = mean+sd*z if z is not None else theta
        admin=np.tile(admin_map[g],(n_per_group,1))
        prob=probabilities(theta,p,D).T
        y=rng.binomial(1,prob).astype(float)
        y[~admin | (rng.random(y.shape)<missing)]=np.nan
        ys.append(y); ad.append(admin); gr.extend([g]*n_per_group); abilities.extend(theta)
        spec.update(mean=mean,sd=sd); saved_groups.append(spec)
    bank=np.full((J,3),np.nan)
    if anchors:
        bank[:min(anchors,J)]=p[:min(anchors,J)]
        bank[0,1]+=drift
    if bank_params is not None:
        bank=np.asarray(bank_params,float)
    data=Dataset(np.vstack(ys),np.vstack(ad),np.array(gr),[f"I{j+1:02}" for j in range(J)],
                 [s.get("name",f"G{g+1}") for g,s in enumerate(groups)],bank,p,np.array(abilities),
                 dict(seed=seed,model=model,D=D,groups=saved_groups,n_per_group=n_per_group,items=items,common=common,
                      anchors=anchors,missing=missing,a_range=a_range,b_range=b_range,c_range=c_range,drift=drift,
                      form_map=admin_map.tolist(),true_params=p.tolist(),bank_params=bank.tolist()))
    data.validate()
    return data


def invariant_report(data, result):
    rows=[]
    def add(name,ok,details): rows.append(dict(check=name,status="Pass" if ok else "Fail",details=details))
    maxpost=maxcounts=0.
    for t,c in enumerate(result.cycles):
        tau,_,_=replay_estep(data,result,t)
        maxpost=max(maxpost,float(np.max(np.abs(tau.sum(axis=1)-1))))
        N,R=expected_counts(data,tau)
        maxcounts=max(maxcounts,float(np.max(np.abs(N-c.expected_N))),float(np.max(np.abs(R-c.expected_R))))
    add("Posterior normalization / replay history",maxpost<1e-10 and maxcounts<1e-8,f"posterior error {maxpost:.3g}; reconstructed count error {maxcounts:.3g}")
    for c in result.cycles:
        if not (np.isfinite(c.params).all() and np.isfinite(c.weights).all() and (c.weights>=0).all()):
            add("Finite estimates and masses",False,"Nonfinite estimate or invalid mass");break
    else: add("Finite estimates and masses",True,"Every recorded state checked")
    c=result.cycles[-1]
    exposures=np.array([data.observed[data.group==g].sum(axis=0) for g in range(len(data.group_names))])
    err=max(float(np.max(np.abs(c.expected_N.sum(axis=-1)-exposures))),float(np.max(np.abs(c.weights.sum(axis=1)-1))))
    add("Expected exposure conservation / population normalization",err<1e-8 and np.all(c.expected_R<=c.expected_N+1e-10) and np.all(c.expected_R>=-1e-10),f"max error {err:.3g}; missing and absent cells excluded")
    add("Bank locks / same-ID sharing",all(np.array_equal(h.params[data.fixed],data.bank[data.fixed]) for h in result.cycles),"One vector per global item ID; bank rows tested every cycle")
    pr=probabilities(result.nodes,result.params,result.config.D)
    add("Response probabilities",bool(np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all()),"Finite and between zero and one on the saved grid")
    if not result.config.anchored:
        target=moments(result.nodes,result.initial_weights[0]) if result.config.initial_weights is not None else np.array([result.config.mean,result.config.sd])
        error=float(np.max(np.abs(moments(result.nodes,result.weights[0])-target)))
        rows.append(dict(check="Reference grid mean / SD",status="Pass" if error<=.03 else "Review",details=f"Max moment error {error:.4g}; declared numerical diagnostic tolerance 0.03 ability units. Refine/widen the grid when needed; interpolation need not match moments exactly."))

    add("M-step optimizer status",all(s["success"] for s in c.optimizer),"Final cycle: "+"; ".join(s["message"] for s in c.optimizer if not s["success"]))
    return rows
