"""Linking designs: fixed items in every estimator, multigroup MML, the four designs, R references, page."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT, requires_erm, requires_tam
from rasch.cmle import _Data, cmle
from rasch.jmle import JMLESettings, jmle
from rasch.linking import (DESIGNS, anchor_stability, evaluate, monte_carlo, reference_rows, run_all, run_design,
                           simulate_forms)
from rasch.mml_em import MMLSettings, mml_em, mml_em_groups
from rasch.simulation import simulate


@pytest.fixture(scope="module")
def forms():
    return simulate_forms(n_a=400, n_b=400, n_anchor=5, n_common=3, n_unique_a=6, n_unique_b=6, mean_b=0.6, seed=11)


# ---------------- data ----------------
def test_simulated_forms_follow_the_design(forms):
    ld = forms
    a_rows, b_rows = ld.form_rows("A"), ld.form_rows("B")
    observed = ~np.isnan(ld.X)
    assert observed[np.ix_(a_rows, ld.form_items("A"))].all() and observed[np.ix_(b_rows, ld.form_items("B"))].all()
    assert not observed[np.ix_(a_rows, np.flatnonzero(ld.roles == "B"))].any()
    assert not observed[np.ix_(b_rows, np.flatnonzero(ld.roles == "A"))].any()
    np.testing.assert_allclose(ld.bank[ld.anchors], ld.true_b[ld.anchors])          # no bank error by default
    assert np.isnan(ld.bank[ld.roles != "anchor"]).all()
    drift = simulate_forms(n_anchor=4, drift_items=2, drift=0.8, seed=3)
    np.testing.assert_allclose(drift.true_b_form_b - drift.true_b, [0.8, 0.8] + [0.0] * (len(drift.true_b) - 2))
    with pytest.raises(ValueError):
        simulate_forms(n_anchor=3, drift_items=4)


def test_cmle_groups_persons_by_item_set_and_score(forms):
    d = _Data(forms.X)
    n_scores = forms.X.shape[1]
    assert len(d.w) <= 2 * n_scores                     # two forms x raw scores, not one row per pattern
    assert d.w.sum() == d.n_informative


# ---------------- fixed items in each estimator ----------------
def test_fixed_anchors_shift_the_whole_cmle_solution_exactly():
    d = simulate(400, 12, seed=3)
    free = cmle(d.X)
    fx = np.full(12, np.nan)
    fx[:4] = free.b[:4] + 0.7
    fixed = cmle(d.X, fixed_b=fx)
    np.testing.assert_allclose(fixed.b, free.b + 0.7, atol=1e-6)
    assert np.isnan(fixed.se_b[:4]).all() and np.isfinite(fixed.se_b[4:]).all()


def test_fixed_anchors_shift_jmle_exactly_without_extreme_scores():
    d = simulate(400, 12, seed=3)
    r = np.nansum(d.X, axis=1)
    X = d.X[(r > 0) & (r < 12)]
    s = JMLESettings(bias=False, tol=1e-8)
    free = jmle(X, s)
    fx = np.full(12, np.nan)
    fx[:4] = free.b[:4] + 0.7
    fixed = jmle(X, s, fixed_b=fx)
    np.testing.assert_allclose(fixed.b, free.b + 0.7, atol=1e-6)
    np.testing.assert_allclose(fixed.theta, free.theta + 0.7, atol=1e-6)
    # With bias correction only the free items are rescaled (as tam.jml does).
    corrected = jmle(X, JMLESettings(tol=1e-8), fixed_b=fx)
    np.testing.assert_allclose(corrected.b[:4], fx[:4])
    np.testing.assert_allclose(corrected.b[4:], corrected.b_uncorrected[4:] * 11 / 12)


def test_multigroup_mml_reduces_to_single_group_and_shifts_with_anchors():
    d = simulate(400, 12, seed=3)
    single = mml_em(d.X)
    groups = mml_em_groups(d.X)
    np.testing.assert_allclose(groups.b, single.b, atol=1e-12)
    assert groups.sds[0] == pytest.approx(single.sd)
    np.testing.assert_allclose(groups.se_b, single.se_b_complete, atol=1e-12)
    s = MMLSettings(tol=1e-8)
    base = mml_em_groups(d.X, settings=s)
    fx = np.full(12, np.nan)
    fx[:4] = base.b[:4] + 0.7
    fixed = mml_em_groups(d.X, fixed_b=fx, settings=s)
    np.testing.assert_allclose(fixed.b, base.b + 0.7, atol=1e-5)
    assert fixed.means[0] == pytest.approx(0.7, abs=1e-5)       # the mean is free once anchors set the scale
    assert fixed.sds[0] == pytest.approx(base.sds[0], abs=1e-5)


def test_multigroup_mml_recovers_group_difference(forms):
    r = mml_em_groups(forms.X, forms.groups)
    assert r.converged and r.means[0] == 0.0
    assert r.means[1] == pytest.approx(forms.true_mean_difference, abs=0.15)


# ---------------- designs ----------------
def test_every_design_puts_items_on_the_bank_scale(forms):
    for method in ("JMLE", "CMLE", "MML"):
        res = run_all(method, forms)
        for d in DESIGNS:
            e = evaluate(forms, res[d])
            assert e["rmse"] < 0.3, (method, d, e)
            assert abs(e["mean_difference"] - e["true_difference"]) < 0.25, (method, d, e)
        # Fixed designs hold anchors at the bank values; free designs' anchor mean equals the bank mean.
        for d in ("fixed", "fixed_concurrent"):
            np.testing.assert_allclose(res[d].b[forms.anchors], forms.bank[forms.anchors])
        assert res["concurrent"].b[forms.anchors].mean() == pytest.approx(forms.bank[forms.anchors].mean())
        sep = res["separate"]
        for form in "AB":
            assert sep.by_form[form][forms.anchors].mean() == pytest.approx(forms.bank[forms.anchors].mean())


def test_unlinked_form_b_absorbs_the_group_difference(forms):
    sep = run_design("separate", "MML", forms)
    gap = np.mean(sep.unlinked["A"][forms.anchors] - sep.unlinked["B"][forms.anchors])
    assert gap == pytest.approx(forms.true_mean_difference, abs=0.15)
    assert sep.shifts["Form B"] - sep.shifts["Form A"] == pytest.approx(gap)


def test_fixed_and_fixed_concurrent_coincide_without_common_items():
    ld = simulate_forms(n_a=300, n_b=300, n_anchor=5, n_common=0, n_unique_a=6, n_unique_b=6, seed=4)
    for method, kw, tol in (("CMLE", {}, 1e-5), ("JMLE", dict(jmle_settings=JMLESettings(bias=False, tol=1e-8)), 1e-5),
                            ("MML", dict(mml_settings=MMLSettings(tol=1e-8)), 1e-4)):
        a = run_design("fixed", method, ld, **kw)
        b = run_design("fixed_concurrent", method, ld, **kw)
        np.testing.assert_allclose(a.b, b.b, atol=tol, err_msg=method)
        np.testing.assert_allclose(a.group_means, b.group_means, atol=tol, err_msg=method)


def test_drifted_anchor_is_flagged_and_dropping_it_removes_the_bias():
    ld = simulate_forms(n_a=800, n_b=800, n_anchor=6, n_common=3, drift_items=1, drift=1.2, seed=21)
    sep = run_design("separate", "MML", ld)
    stab = pd.DataFrame(anchor_stability(ld, sep, threshold=0.5))
    assert stab.flag.tolist() == ["Unstable"] + ["Stable"] * 5
    kept = ld.anchors[1:]
    for design in DESIGNS:
        with_drift = evaluate(ld, run_design(design, "MML", ld))
        clean = evaluate(ld, run_design(design, "MML", ld, kept))
        err = clean["mean_difference"] - clean["true_difference"]
        assert abs(err) < abs(with_drift["mean_difference"] - with_drift["true_difference"]), design
        assert abs(err) < 0.1, (design, err)


def test_stability_check_is_not_fooled_by_several_drifting_anchors():
    # Two of six anchors drift by +1: mean centring would push the stable anchors about -0.33 and flag them.
    ld = simulate_forms(n_a=500, n_b=500, n_anchor=6, n_common=4, drift_items=2, drift=1.0, seed=2026)
    stab = pd.DataFrame(anchor_stability(ld, run_design("separate", "MML", ld), threshold=0.5))
    assert stab.flag.tolist() == ["Unstable"] * 2 + ["Stable"] * 4


def test_dropped_anchor_is_split_by_form_in_concurrent_designs():
    ld = simulate_forms(n_a=600, n_b=600, n_anchor=5, n_common=2, drift_items=1, drift=1.0, seed=5)
    for design in ("concurrent", "fixed_concurrent"):
        r = run_design(design, "MML", ld, ld.anchors[1:])
        run = r.runs[0]
        assert run.X.shape[1] == len(ld.item_ids) + 1 and run.column_ids[-1] == "K01 (Form B)"
        assert np.isnan(run.X[ld.groups == 1, 0]).all() and np.isnan(run.X[ld.groups == 0, -1]).all()
        # The Form B copy carries the drift.
        assert r.by_form["B"][0] - r.by_form["A"][0] == pytest.approx(1.0, abs=0.3)


def test_fixed_anchor_displacement_points_to_the_drifted_anchor():
    ld = simulate_forms(n_a=800, n_b=800, n_anchor=6, n_common=0, drift_items=1, drift=1.0, seed=8)
    for method in ("JMLE", "CMLE", "MML"):
        r = run_design("fixed", method, ld)
        disp = r.anchor_displacement
        assert int(np.argmax(np.abs(disp))) == 0 and disp[0] > 0.3, (method, disp)


def test_monte_carlo_holds_items_and_bank_fixed(forms):
    rows, done = monte_carlo("JMLE", forms, replications=3)
    assert done == 3 and len(rows) == 3 * len(DESIGNS)
    assert {r["design"] for r in rows} == {"Separate + link", "Concurrent", "Fixed anchor", "Fixed + concurrent"}


# ---------------- reference packages ----------------
@requires_tam
@pytest.mark.parametrize("method", ["JMLE", "MML"])
def test_tam_agrees_on_every_calibration_run(forms, method):
    rows = []
    for res in run_all(method, forms).values():
        rows += reference_rows(method, forms, res)
    frame = pd.DataFrame(rows)
    assert (frame.status == "Pass").all(), frame[frame.status != "Pass"]
    checks = set(frame.check)
    assert "Anchors held at bank values" in checks
    if method == "MML":
        assert {"Group means", "Group variances"} <= checks


@requires_erm
def test_erm_agrees_on_free_cmle_runs(forms):
    rows = []
    for res in run_all("CMLE", forms).values():
        rows += reference_rows("CMLE", forms, res)
    frame = pd.DataFrame(rows)
    assert (frame.loc[frame.status != "No R reference", "status"] == "Pass").all()
    assert set(frame.loc[frame.status == "No R reference", "design"]) == {"Fixed anchor", "Fixed + concurrent"}


# ---------------- page ----------------
def test_linking_page_runs_every_design_and_repeated_samples():
    at = AppTest.from_file(str(ROOT / "pages/9_Linking_and_Anchoring.py"), default_timeout=180).run()
    assert not at.exception
    next(n for n in at.number_input if n.label == "Group A persons (Form A)").set_value(200).run()
    next(n for n in at.number_input if n.label == "Group B persons (Form B)").set_value(200).run()
    for method in ("MML", "JMLE", "CMLE"):
        next(r for r in at.radio if r.label == "Estimator").set_value(method).run()
        next(b for b in at.button if b.label == "Run the four designs").click().run()
        assert not at.exception, [e.message for e in at.exception]
        assert any("Compare the designs" in s.value for s in at.subheader)
    next(n for n in at.number_input if n.label == "Replications").set_value(5).run()
    next(b for b in at.button if b.label == "Run repeated samples").click().run()
    assert not at.exception
    assert any("replications in" in c.value for c in at.caption)
    # Changing the data marks the results stale.
    next(s for s in at.slider if s.label == "Group B ability mean (group A = 0)").set_value(1.0).run()
    assert any("settings changed" in w.value.lower() for w in at.warning)


def test_drop_flagged_anchors_button_updates_the_anchor_set():
    at = AppTest.from_file(str(ROOT / "pages/9_Linking_and_Anchoring.py"), default_timeout=180).run()
    next(s for s in at.slider if s.label == "Anchors that drift in Form B").set_value(1).run()
    next(s for s in at.slider if s.label == "Drift size (logits, harder in Form B)").set_value(1.5).run()
    next(b for b in at.button if b.label == "Run the four designs").click().run()
    drop = next(b for b in at.button if b.label.startswith("Drop flagged anchors"))
    assert "K01" in drop.label
    drop.click().run()
    assert not at.exception
    assert at.session_state["link_anchors"] == ["K02", "K03", "K04", "K05", "K06"]
