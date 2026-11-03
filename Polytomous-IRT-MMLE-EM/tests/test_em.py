"""EM engine, designs and identification (acceptance items 6, 7, 10–14, 16–19 and the edge cases)."""
from dataclasses import replace

import numpy as np
import pytest

from core import walkthrough as W
from core.data import Dataset, validate, walkthrough_dataset
from core.diagnostics import category_fit, information_criteria, monte_carlo
from core.em import (FitConfig, build_grid, estep, expected_counts, fit, marginal_loglik, node_masses, plan,
                     standard_errors, start_items)
from core.models import Item, ItemSpec, make_item
from core.quadrature import moments
from core.simulation import SimSpec, simulate


@pytest.fixture(scope="module")
def conc():
    return simulate(SimSpec(model="GPCM", design="concurrent", n_per_group=400, n_items=6, n_common=3, ncat=3,
                            groups=2, seed=5))


def test_posterior_normalises_and_counts_reconcile(conc):
    cfg = FitConfig(model="GPCM", design="concurrent")
    nodes, w = build_grid(cfg, conc.G)
    items = start_items(conc, "GPCM", np.zeros(conc.J, bool))
    es = estep(conc, items, nodes, w)
    np.testing.assert_allclose(es["post"].sum(axis=1), 1, atol=1e-12)
    C = expected_counts(conc, es["post"])
    for j in range(conc.J):
        for g in range(conc.G):
            np.testing.assert_allclose(C[j][g].sum(axis=0), conc.counts(j, g), atol=1e-9)
    np.testing.assert_allclose(node_masses(conc, es["post"]).sum(axis=1), conc.group_sizes(), atol=1e-9)


def test_missing_by_design_is_not_a_zero_response(conc):
    """Filling unadministered cells with 0 must change the likelihood; NaN cells contribute nothing."""
    cfg = FitConfig(model="GPCM", design="concurrent")
    nodes, w = build_grid(cfg, conc.G)
    items = start_items(conc, "GPCM", np.zeros(conc.J, bool))
    ll = marginal_loglik(conc, items, nodes, w)
    sub = conc.subset_groups([0])
    ll0 = marginal_loglik(sub, [items[conc.ids.index(i)] for i in sub.ids], nodes[:1], w[:1])
    sub1 = conc.subset_groups([1])
    ll1 = marginal_loglik(sub1, [items[conc.ids.index(i)] for i in sub1.ids], nodes[:1], w[1:])
    assert ll == pytest.approx(ll0 + ll1, abs=1e-8)
    filled = Dataset("f", np.nan_to_num(conc.Y, nan=0), conc.group, conc.group_names, conc.specs)
    assert marginal_loglik(filled, items, nodes, w) < ll - 1


@pytest.mark.parametrize("model", ["PCM", "RSM", "GPCM", "GRM"])
@pytest.mark.parametrize("prior", ["fixed_normal", "estimated_normal", "estimated_empirical"])
def test_loglik_never_decreases_with_exact_updates(model, prior):
    d = simulate(SimSpec(model=model, n_per_group=300, n_items=5, ncat=4, seed=11))
    r = fit(d, FitConfig(model=model, prior=prior, n_nodes=21, max_cycles=150))
    ll = [h["loglik"] for h in r.history] + [r.loglik]
    assert np.all(np.diff(ll) >= -1e-7 * abs(ll[0]))
    assert not any(h["ll_decrease"] for h in r.history)


def test_rsm_shared_steps_are_joint_and_sum_to_zero():
    d = simulate(SimSpec(model="RSM", n_per_group=500, n_items=6, ncat=4, seed=2))
    r = fit(d, FitConfig(model="RSM", prior="estimated_normal"))
    taus = np.array([it.tau for it in r.items])
    np.testing.assert_allclose(taus, taus[0][None, :].repeat(len(taus), 0), atol=1e-12)
    assert abs(taus[0].sum()) < 1e-10
    assert any("joint b and shared τ" in s["message"] for s in r.history[-1]["statuses"])


def test_grm_thresholds_stay_ordered_after_fit():
    d = simulate(SimSpec(model="GRM", n_per_group=300, n_items=5, ncat=5, seed=3))
    r = fit(d, FitConfig(model="GRM"))
    assert all(np.all(np.diff(it.d) > 0) for it in r.items)


@pytest.mark.parametrize("model", ["PCM", "RSM", "GPCM", "GRM"])
def test_fixed_anchors_unchanged_exactly(model):
    d = simulate(SimSpec(model=model, design="fipc", n_per_group=500, n_items=4, n_anchor=4, ncat=3,
                         group_means=[0.4], group_sds=[1.1], seed=4))
    for method in ("MEM", "OEM"):
        r = fit(d, FitConfig(model=model, design="fipc", prior="estimated_normal", fipc_method=method))
        for it, f in zip(r.items, r.fixed):
            if f:
                assert np.array_equal(it.vector(), d.anchors[it.id].vector())
        for h in r.history:
            for a, f in zip(h["items_after"], r.fixed):
                if f:
                    assert np.array_equal(a.vector(), d.anchors[a.id].vector())


def test_oem_is_one_cycle_and_equals_first_mem_cycle():
    d = simulate(SimSpec(model="GPCM", design="fipc", n_per_group=600, n_items=4, n_anchor=4, ncat=3,
                         group_means=[0.5], group_sds=[1.0], seed=6))
    oem = fit(d, FitConfig(model="GPCM", design="fipc", prior="estimated_normal", fipc_method="OEM"))
    mem = fit(d, FitConfig(model="GPCM", design="fipc", prior="estimated_normal", fipc_method="MEM"))
    assert oem.cycles == 1 and not oem.converged and "OEM" in oem.reason
    assert oem.history[0]["estep_items"] == "anchors only"
    np.testing.assert_allclose(oem.weights, oem.initial_weights)        # no prior-weights updating
    assert oem.implied_weights is not None
    for a, b in zip(oem.items, mem.history[0]["items_after"]):
        np.testing.assert_allclose(a.vector(), b.vector(), atol=1e-12)
    assert mem.converged and mem.cycles > 1
    assert mem.history[1]["estep_items"] == "all items"
    assert abs(mem.group_moments[0, 0] - 0.5) < 0.15


def test_concurrent_common_items_share_one_parameter_vector(conc):
    """Pooled fixed-normal concurrent calibration equals a single-group fit of the stacked data."""
    a = fit(conc, FitConfig(model="GPCM", design="concurrent", prior="fixed_normal", pooled=True, tol=1e-7))
    single = Dataset("s", conc.Y, np.zeros(conc.N, int), ["all"], conc.specs)
    b = fit(single, FitConfig(model="GPCM", design="single", prior="fixed_normal", tol=1e-7))
    for x, y in zip(a.items, b.items):
        np.testing.assert_allclose(x.vector(), y.vector(), atol=1e-6)
    assert len(a.items) == conc.J                            # one vector per item, whatever the number of forms


def test_reference_group_identification(conc):
    r = fit(conc, FitConfig(model="GPCM", design="concurrent", prior="estimated_normal"))
    m = r.group_moments
    assert m[0, 0] == pytest.approx(0, abs=1e-10) and m[0, 1] == pytest.approx(1, abs=1e-6)
    assert abs(m[1, 0] - 0.5) < 0.25 and m[1, 1] > 0
    d = simulate(SimSpec(model="PCM", design="concurrent", n_per_group=500, n_items=6, n_common=3, ncat=3, groups=2,
                         group_sds=[1.3, 1.0], seed=8))
    r = fit(d, FitConfig(model="PCM", design="concurrent", prior="estimated_normal"))
    assert r.group_moments[0, 0] == pytest.approx(0, abs=1e-8)
    assert abs(r.group_moments[0, 1] - 1.3) < 0.2            # Rasch: reference SD is estimated


def test_constrained_empirical_meets_identification_each_cycle():
    d = simulate(SimSpec(model="GPCM", n_per_group=800, n_items=6, ncat=4, shape="skewed", seed=9))
    r = fit(d, FitConfig(model="GPCM", prior="estimated_empirical", max_cycles=60))
    for h in r.history:
        m, s = moments(r.nodes[0], h["weights_after"][0])
        assert abs(m) < 1e-8 and abs(s - 1) < 1e-8


def test_unconstrained_empirical_in_anchored_design_equals_average_posterior():
    d = simulate(SimSpec(model="GRM", design="fipc", n_per_group=600, n_items=4, n_anchor=5, ncat=3,
                         group_means=[0.3], group_sds=[1.0], seed=10))
    cfg = FitConfig(model="GRM", design="fipc", prior="estimated_empirical", max_cycles=5)
    r = fit(d, cfg)
    h = r.history[2]
    es = estep(d, h["items_before"], r.nodes, h["weights_before"])
    np.testing.assert_allclose(h["weights_after"][0], es["post"].mean(axis=0), atol=1e-12)


def test_group_specific_priors_are_not_pooled(conc):
    r = fit(conc, FitConfig(model="GPCM", design="concurrent", prior="estimated_empirical", max_cycles=30))
    assert np.max(np.abs(r.weights[0] - r.weights[1])) > 1e-3
    p = fit(conc, FitConfig(model="GPCM", design="concurrent", prior="estimated_empirical", pooled=True, max_cycles=30))
    np.testing.assert_allclose(p.weights[0], p.weights[1])


def test_estimated_normal_variances_positive():
    d = simulate(SimSpec(model="GRM", design="mgfipc", n_per_group=400, n_items=4, n_anchor=4, n_common=2, ncat=3,
                         groups=3, seed=12))
    r = fit(d, FitConfig(model="GRM", design="mgfipc", prior="estimated_normal"))
    assert np.all(r.group_moments[:, 1] > 0)
    assert r.n_parameters() > 0


def test_walkthrough_tables_reproduce_engine_values():
    d = walkthrough_dataset()
    cfg = W.walkthrough_config("PCM", "fixed_normal")
    cyc = W.cycle(d, cfg, W.default_start(d, "PCM"))
    tp = W.table_posterior(d, cyc)
    np.testing.assert_allclose(tp["posterior"].to_numpy().reshape(5, 3), cyc["estep"]["post"])
    np.testing.assert_allclose(cyc["nodes"][0], [-np.sqrt(3), 0, np.sqrt(3)])
    np.testing.assert_allclose(cyc["weights"][0], [1 / 6, 2 / 3, 1 / 6])
    assert W.reconcile_counts(d, cyc)["difference"].abs().max() < 1e-12
    tl = W.table_likelihoods(d, cyc["items"], cyc)
    np.testing.assert_allclose(tl["likelihood"].to_numpy().reshape(5, 3), np.exp(cyc["estep"]["logL"]))
    assert cyc["ll_after"] >= cyc["ll_before"]
    # GPCM, estimated empirical on three nodes with both moments fixed: masses cannot move
    c2 = W.cycle(d, W.walkthrough_config("GPCM", "estimated_empirical"), W.default_start(d, "GPCM"))
    np.testing.assert_allclose(c2["new_weights"][0], [1 / 6, 2 / 3, 1 / 6], atol=1e-9)


def test_simulation_and_monte_carlo_are_reproducible():
    s = SimSpec(model="PCM", n_per_group=200, n_items=4, ncat=3, seed=21)
    np.testing.assert_array_equal(simulate(s).Y, simulate(s).Y)
    cfg = FitConfig(model="PCM", prior="fixed_normal", n_nodes=21)
    a = monte_carlo(s, cfg, reps=2, seed0=5)
    b = monte_carlo(s, cfg, reps=2, seed0=5)
    np.testing.assert_allclose(a["items"]["estimate"], b["items"]["estimate"])


def test_prior_misspecification_detected_by_aic():
    d = simulate(SimSpec(model="GPCM", n_per_group=3000, n_items=10, ncat=4, shape="bimodal", seed=15))
    rn = fit(d, FitConfig(model="GPCM", prior="fixed_normal"), record=False)
    re = fit(d, FitConfig(model="GPCM", prior="estimated_empirical"), record=False)
    assert information_criteria(re, d.N)["AIC"] < information_criteria(rn, d.N)["AIC"]


def test_standard_errors_match_numerical_hessian():
    """Independent check: finite-difference Hessian of the marginal log-likelihood over all free parameters."""
    d = simulate(SimSpec(model="GPCM", n_per_group=400, n_items=3, ncat=3, seed=16))
    r = fit(d, FitConfig(model="GPCM", tol=1e-9, se=True))
    v0 = np.concatenate([it.vector() for it in r.items])
    sizes = [len(it.vector()) for it in r.items]

    def ll(v):
        items, k = [], 0
        for it, m in zip(r.items, sizes):
            items.append(make_item(it.id, "GPCM", v[k], tuple(v[k + 1:k + m])))
            k += m
        return marginal_loglik(d, items, r.nodes, r.weights)
    n, h = len(v0), 1e-3
    H = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            e1, e2 = np.eye(n)[a] * h, np.eye(n)[b] * h
            H[a, b] = H[b, a] = (ll(v0 + e1 + e2) - ll(v0 + e1 - e2) - ll(v0 - e1 + e2) + ll(v0 - e1 - e2)) / (4 * h * h)
    se_num = np.sqrt(np.diag(np.linalg.inv(-H)))
    se_lab = np.concatenate([[r.se["items"][it.id][lab] for lab in it.labels()] for it in r.items])
    np.testing.assert_allclose(se_lab, se_num, rtol=2e-3)


def test_fit_category_proportions_close_for_true_model():
    d = simulate(SimSpec(model="GPCM", n_per_group=2000, n_items=6, ncat=4, seed=17))
    r = fit(d, FitConfig(model="GPCM"))
    t = category_fit(d, r)
    assert np.sqrt(np.mean(t["difference"] ** 2)) < 0.01


# ---- edge cases -------------------------------------------------------------------------------

def test_duplicate_ids_and_invalid_categories_are_errors():
    Y = np.array([[0, 1], [1, 3]], float)
    d = Dataset("x", Y, np.zeros(2, int), ["g"], [ItemSpec("A", 3), ItemSpec("A", 3)])
    errs, _ = validate(d)
    assert any("Duplicated" in e for e in errs) and any("0…2" in e for e in errs)


def test_insufficient_anchors_and_unlinked_groups():
    d = simulate(SimSpec(model="GPCM", n_per_group=200, n_items=4, ncat=3, seed=1))
    assert any("anchor" in e for e in plan(d, FitConfig(model="GPCM", design="fipc"))["errors"])
    Y = np.full((4, 2), np.nan)
    Y[:2, 0] = [0, 1]
    Y[2:, 1] = [1, 0]
    d2 = Dataset("u", Y, np.array([0, 0, 1, 1]), ["a", "b"], [ItemSpec("I1", 2), ItemSpec("I2", 2)])
    assert any("share no items" in e for e in plan(d2, FitConfig(model="PCM", design="concurrent"))["errors"])
    d3 = simulate(SimSpec(model="GPCM", design="fipc", n_per_group=200, n_items=3, n_anchor=1, ncat=3, seed=2))
    assert any("weak identification" in w for w in plan(d3, FitConfig(model="GPCM", design="fipc",
                                                                      prior="estimated_normal"))["warnings"])


def test_incompatible_rating_scale_categories():
    Y = np.array([[0, 1], [1, 3], [2, 0]], float)
    d = Dataset("r", Y, np.zeros(3, int), ["g"], [ItemSpec("A", 3, "scale"), ItemSpec("B", 4, "scale")])
    errs, _ = validate(d, "RSM")
    assert any("mixes items" in e for e in errs)


def test_gauss_hermite_rejected_for_estimated_priors_and_sparse_category_warned():
    d = simulate(SimSpec(model="PCM", n_per_group=200, n_items=3, ncat=3, seed=4))
    assert plan(d, FitConfig(model="PCM", prior="estimated_normal", grid="gauss_hermite"))["errors"]
    Y = d.Y.copy()
    Y[Y[:, 0] == 2, 0] = 1                               # category 2 of item 1 never observed
    d2 = Dataset("s", Y, d.group, d.group_names, d.specs)
    _, warns = validate(d2, "PCM")
    assert any("never observed" in w for w in warns)
    r = fit(d2, FitConfig(model="PCM", max_cycles=200))
    assert np.isfinite(r.loglik) and r.items[0].d[1] <= 10.0 + 1e-9
    assert any("bound" in w for w in r.warnings)


def test_extreme_thresholds_and_nearly_empty_bins_are_stable():
    d = simulate(SimSpec(model="GRM", n_per_group=150, n_items=4, ncat=4, shape="bimodal", seed=5))
    r = fit(d, FitConfig(model="GRM", prior="estimated_empirical", n_nodes=61, max_cycles=300), record=False)
    assert np.isfinite(r.loglik) and np.all(np.isfinite(r.weights)) and np.all(r.weights >= 0)
    it = make_item("x", "GRM", 8.0, (-9.9, 9.9))
    from core.models import log_probs
    assert np.all(np.isfinite(log_probs(it, np.linspace(-50, 50, 11))))


def test_rsm_anchors_fix_shared_steps():
    d = simulate(SimSpec(model="RSM", design="fipc", n_per_group=600, n_items=4, n_anchor=3, ncat=4,
                         group_means=[0.3], group_sds=[1.0], seed=7))
    r = fit(d, FitConfig(model="RSM", design="fipc", prior="estimated_normal"))
    tau = np.array(next(iter(d.anchors.values())).tau)
    for it in r.items:
        np.testing.assert_allclose(it.tau, tau, atol=1e-12)
