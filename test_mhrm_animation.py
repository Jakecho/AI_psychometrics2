"""Recorded RM paths, bounded-size frames, and persistent item selection."""

from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from mhrm_engine import MHRM_MIRT, simulate_data
from mhrm_animation import item_convergence_figure, item_history_table, item_path


@pytest.fixture
def fitted():
    q = np.r_[np.repeat(np.eye(2), 3, axis=0), [[1, 1]]]
    y, _, a, d = simulate_data(150, len(q), 2, q, seed=91)
    model = MHRM_MIRT(y, 2, q).fit(max_iter=12, burn_in=3, sem_cycles=4)
    return model, a, d


def test_recorded_endpoints_and_copies(fitted):
    model, _, _ = fitted
    assert model.rm_cycle_history == list(range(13))
    assert len(model.rm_parameter_history) == 13
    expected_start = np.zeros((model.J, model.D+1))
    for j, (active, beta) in enumerate(zip(model.active, model.sem_parameter_mean_)):
        expected_start[j, active] = beta[:-1]
        expected_start[j, -1] = beta[-1]
    np.testing.assert_array_equal(model.rm_parameter_history[0], expected_start)
    np.testing.assert_array_equal(model.rm_parameter_history[-1], np.column_stack([model.est_a, model.est_d]))
    snapshots = np.asarray(model.rm_parameter_history)
    assert np.all(snapshots[:, :, :model.D][:, ~model.Q] == 0)
    last = model.rm_parameter_history[-1].copy()
    model.est_d[:] += 100
    np.testing.assert_array_equal(model.rm_parameter_history[-1], last)


def test_every_snapshot_matches_the_post_update_parameters(fitted):
    model, _, _ = fitted
    observed = []
    original = model.rm_step
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(np.column_stack((model.est_a, model.est_d)))
        return result
    model.rm_step = record
    model.fit(max_iter=8, burn_in=2, sem_cycles=3)
    assert len(model.rm_parameter_history) == 9  # previous fit's history is reset
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(model.rm_parameter_history[1:]))


def test_csv_columns_include_only_active_parameters_and_actual_deltas(fitted):
    model, _, _ = fitted
    pure = item_history_table(model, 0)
    assert 'a1' in pure and 'a2' not in pure and 'd' in pure
    cross = item_history_table(model, 6)
    assert {'a1', 'a2', 'd'} <= set(cross.columns)
    assert len(cross) == model.rm_iterations_+1
    np.testing.assert_allclose(cross['Change in a1'].iloc[1:], np.diff(cross.a1))
    assert cross['RM cycle'].iloc[0] == cross.Gain.iloc[0] == 0
    np.testing.assert_allclose(cross.Gain.iloc[1:], (.1/np.arange(1, 13))**.75)


def test_optional_factor_labels_rename_slopes_only(fitted):
    model, _, _ = fitted
    table = item_history_table(model, 6, factor_labels=['G', 'S_1'])
    assert {'a(G)', 'a(S_1)', 'd', 'Change in a(G)'} <= set(table.columns)
    np.testing.assert_array_equal(table['a(G)'], item_history_table(model, 6).a1)
    figure = item_convergence_figure(model, 6, factor_labels=['G', 'S_1'])
    assert [a.text for a in figure.layout.annotations][:3] == ['a(G)', 'a(S_1)', 'd']
    with pytest.raises(ValueError, match='one factor label'):
        item_path(model, 0, factor_labels=['G'])


def test_full_run_history_covers_every_stage_and_matches_rm_history(fitted):
    model, _, _ = fitted   # burn_in=3, sem_cycles=4, 12 RM cycles
    assert model.total_cycle_history == list(range(model.n_iter_ + 1))
    assert model.stage_history == ['Start'] + ['Burn-in']*3 + ['SEM']*4 + ['RM']*12
    assert model.stage_cycle_history == [0, 1, 2, 3, 1, 2, 3, 4] + list(range(1, 13))
    # The last SEM snapshot is the SEM-averaged RM start; RM snapshots are identical.
    np.testing.assert_array_equal(np.asarray(model.parameter_history[7:]), np.asarray(model.rm_parameter_history))
    np.testing.assert_array_equal(model.gain_history[8:], model.rm_gain_history[1:])
    assert np.all(np.asarray(model.parameter_history)[:, :, :model.D][:, ~model.Q] == 0)


def test_all_stage_playback_table_and_shading(fitted):
    model, a, d = fitted
    table = item_history_table(model, 6, stages='all')
    assert list(table.columns[:5]) == ['Item', 'Total cycle', 'Stage', 'Stage cycle', 'Gain']
    assert len(table) == model.n_iter_ + 1 and table.Stage.iloc[0] == 'Start'
    np.testing.assert_allclose(table['Change in a1'].iloc[1:], np.diff(table.a1))
    figure = item_convergence_figure(model, 6, true_a=a, true_d=d, stages='all')
    assert len(figure.frames) == model.n_iter_ + 1
    assert 'random starting values' in figure.layout.title.text
    assert len([s for s in figure.layout.shapes if s.type == 'rect']) == 3 * 3  # 3 stages x 3 panels
    assert figure.layout.xaxis.range[1] > model.n_iter_ + .5   # room for the end label
    with pytest.raises(ValueError, match="'rm' or 'all'"):
        item_path(model, 0, stages='burn-in')


def test_animation_frames_use_recorded_values_and_fixed_axes(fitted):
    model, a, d = fitted
    figure = item_convergence_figure(model, 6, true_a=a, true_d=d,
                                    reference_a=model.est_a, reference_d=model.est_d)
    assert len(figure.frames) == 13
    assert figure.frames[0].name == '0' and figure.frames[-1].name == '12'
    path = item_path(model, 6)
    for frame in figure.frames:
        cycle = int(frame.name)
        for panel in range(3):
            np.testing.assert_allclose(frame.data[2*panel].y, [path['values'][cycle, panel]])
            np.testing.assert_allclose(frame.data[2*panel].customdata, [[path['changes'][cycle, panel]]])
    assert [b.label for b in figure.layout.updatemenus[0].buttons] == ['Play', 'Pause', 'Restart']
    assert figure.layout.xaxis.autorange is False
    assert figure.layout.yaxis.autorange is False
    assert figure.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] == 0
    assert len(figure.layout.shapes) == 6  # truth and R reference for each parameter


def test_live_chart_has_no_animation_frames(fitted):
    model, _, _ = fitted
    figure = item_convergence_figure(model, 0, animate=False)
    assert len(figure.frames) == 0
    assert 'RM cycle 12' in figure.layout.title.text
    assert list(figure.data[1].y) == [model.est_a[0, 0]]


def test_long_run_payload_is_bounded_and_retains_full_csv():
    cycles = 2001
    snapshots = [np.tile([1+.1*np.sin(k/10), 1+.1*np.cos(k/10), -.2], (2, 1)) for k in range(cycles)]
    model = SimpleNamespace(J=2, D=2, Q=np.ones((2, 2), dtype=bool),
                            rm_parameter_history=snapshots,
                            rm_cycle_history=list(range(cycles)), rm_gain_history=[0.]+[.01]*2000)
    figure = item_convergence_figure(model, 1)
    assert len(figure.frames) <= 240
    assert figure.frames[-1].name == '2000'
    assert len(figure.data[0].x) == cycles
    assert len(item_history_table(model, 1)) == cycles
    assert len(figure.to_json()) < 1_000_000


def test_no_trajectory_and_invalid_item_are_friendly(fitted):
    model, _, _ = fitted
    with pytest.raises(ValueError, match='Select an item'):
        item_path(model, 99)
    model.rm_parameter_history = []
    with pytest.raises(ValueError, match='No RM trajectory'):
        item_path(model, 0)


def test_dashboard_embeds_player_and_item_selection_reuses_fit():
    app = AppTest.from_file('MHRM_2D.py').run(timeout=30)
    def widget(kind, label):
        return next(w for w in getattr(app, kind) if w.label == label)
    widget('slider', 'Sample Size (N)').set_value(500)
    widget('slider', 'Number of Items (J)').set_value(10)
    widget('slider', 'Maximum RM Cycles (Stage 3)').set_value(50)
    widget('slider', 'Burn-in Cycles (Stage 1)').set_value(10)
    widget('slider', 'SEM Averaging Cycles (Stage 2)').set_value(20)
    widget('checkbox', 'Cross-check with R mirt').uncheck()
    app.run()
    app.button[0].click().run(timeout=60)
    assert not app.exception and not app.error
    assert any(h.value == 'Step 3 · Parameter convergence playback' for h in app.subheader)
    player = app.get('iframe')[0].proto.srcdoc
    assert 'Plotly.addFrames' in player and 'Play' in player and 'Restart' in player
    original = app.session_state['mhrm_saved_run']['model'].est_a.copy()
    widget('selectbox', 'Item to inspect').select(10).run(timeout=30)
    assert not app.exception and not app.error
    np.testing.assert_array_equal(app.session_state['mhrm_saved_run']['model'].est_a, original)
    assert 'Item 10' in app.get('iframe')[0].proto.srcdoc
    def trajectory_table():
        return next(df.value for df in app.dataframe if 'Gain' in df.value.columns and 'Item' in df.value.columns)
    trajectory = trajectory_table()
    assert trajectory.Item.eq(10).all()
    assert len(trajectory) == 81  # cycle 0 + 10 burn-in + 20 SEM + 50 RM (default: all stages)
    assert list(trajectory.Stage.unique()) == ['Start', 'Burn-in', 'SEM', 'RM']
    widget('radio', 'Playback range').set_value('RM stage only').run(timeout=30)
    assert len(trajectory_table()) == 51
    widget('selectbox', 'Playback speed').select('Fast').run(timeout=30)
    np.testing.assert_array_equal(app.session_state['mhrm_saved_run']['model'].est_a, original)
