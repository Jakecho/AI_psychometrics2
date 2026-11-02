"""Bounded-size playback of actual recorded item parameter trajectories.

stages='rm' (default) shows the stage-3 Robbins-Monro path, starting from the
SEM-averaged values. stages='all' shows the full run: random starting values,
burn-in, SEM averaging and RM, so the early movement toward the estimate is visible.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STAGE_SHADING = {'Burn-in': 'rgba(148,163,184,0.16)', 'SEM': 'rgba(37,99,235,0.07)',
                 'RM': 'rgba(22,163,74,0.07)'}


def _slope_label(k, factor_labels):
    return f'a{k+1}' if factor_labels is None else f'a({factor_labels[k]})'


def item_path(model, item_index, factor_labels=None, stages='rm'):
    if not 0 <= item_index < model.J:
        raise ValueError('Select an item in the calibrated bank.')
    if stages not in ('rm', 'all'):
        raise ValueError("stages must be 'rm' or 'all'.")
    if stages == 'rm':
        if not model.rm_parameter_history:
            raise ValueError('No RM trajectory is available until SEM averaging finishes.')
        history = model.rm_parameter_history
        cycles = np.asarray(model.rm_cycle_history)
        gains = np.asarray(model.rm_gain_history)
        stage = np.array(['RM'] * len(cycles), dtype=object)
        stage_cycles = cycles
    else:
        if not getattr(model, 'parameter_history', None):
            raise ValueError('No trajectory is available until calibration starts.')
        history = model.parameter_history
        cycles = np.asarray(model.total_cycle_history)
        gains = np.asarray(model.gain_history)
        stage = np.asarray(model.stage_history, dtype=object)
        stage_cycles = np.asarray(model.stage_cycle_history)
    if len(history) != len(cycles) or len(gains) != len(cycles) or len(stage) != len(cycles) or any(
        np.shape(snapshot) != (model.J, model.D+1) for snapshot in history
    ):
        raise ValueError('Recorded trajectory dimensions do not agree.')
    if factor_labels is not None and len(factor_labels) != model.D:
        raise ValueError('Provide one factor label per dimension.')
    parameters = list(np.flatnonzero(model.Q[item_index])) + [model.D]
    labels = [_slope_label(k, factor_labels) if k < model.D else 'd' for k in parameters]
    # Select one item before stacking so live plots don't repeatedly copy the
    # complete bank's trajectory while calibration is still running.
    values = np.stack([snapshot[item_index, parameters] for snapshot in history])
    differences = np.vstack((np.zeros(len(parameters)), np.diff(values, axis=0)))
    return dict(cycles=cycles, gains=gains, parameters=parameters, labels=labels,
                values=values, changes=differences, stage=stage, stage_cycles=stage_cycles,
                axis_label='RM cycle' if stages == 'rm' else 'Total cycle', stages=stages)


def item_history_table(model, item_index, factor_labels=None, stages='rm'):
    path = item_path(model, item_index, factor_labels, stages)
    result = pd.DataFrame({path['axis_label']: path['cycles']})
    if stages == 'all':
        result['Stage'] = path['stage']
        result['Stage cycle'] = path['stage_cycles']
    result['Gain'] = path['gains']
    for k, label in enumerate(path['labels']):
        result[label] = path['values'][:, k]
        result[f'Change in {label}'] = path['changes'][:, k]
    result.insert(0, 'Item', item_index+1)
    return result


def _stage_spans(cycles, stage):
    """Contiguous (stage, first cycle, last cycle) runs, excluding the start point."""
    spans = []
    for c, s in zip(cycles, stage):
        if s == 'Start':
            continue
        if spans and spans[-1][0] == s:
            spans[-1][2] = c
        else:
            spans.append([s, c, c])
    return spans


def item_convergence_figure(model, item_index, *, true_a=None, true_d=None,
                            reference_a=None, reference_d=None, animate=True,
                            frame_duration=180, max_frames=240, factor_labels=None,
                            stages='rm'):
    """Full traces stay static; only current markers/cursors change per frame.

    This keeps JSON size linear in trajectory length, rather than copying
    growing traces into every frame. Long runs sample playback frames but
    retain every recorded cycle in the trace and CSV. Transitions are zero:
    the playback does not invent intermediate parameter values.
    """
    path = item_path(model, item_index, factor_labels, stages)
    cycles, gains, values, changes = (path[k] for k in ('cycles', 'gains', 'values', 'changes'))
    stage, stage_cycles, axis_label = path['stage'], path['stage_cycles'], path['axis_label']
    count = len(path['parameters'])
    cols = min(3, count)
    rows = math.ceil(count/cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=path['labels'],
                        horizontal_spacing=.12, vertical_spacing=.2)
    selected = 0 if animate else len(cycles)-1
    last = max(int(cycles[-1]), 1)
    # Room on the right so the current-value label is not clipped at the final cycle.
    x_range = [-max(.5, .03*last), last + max(.5, .03*last)]   # keep end markers off the axes
    spans = _stage_spans(cycles, stage) if stages == 'all' else []
    indices, ranges = [], []
    for panel, (parameter, label) in enumerate(zip(path['parameters'], path['labels'])):
        row, col = divmod(panel, cols)
        row, col = row+1, col+1
        y = values[:, panel]
        for name, first, final in spans:
            fig.add_vrect(x0=first-.5, x1=final+.5, fillcolor=STAGE_SHADING.get(name, 'rgba(0,0,0,0)'),
                          line_width=0, layer='below', row=row, col=col, exclude_empty_subplots=False,
                          annotation_text=name, annotation_position='top left',
                          annotation_font=dict(size=10, color='#64748b'))
        reference_values = []
        if true_a is not None and true_d is not None:
            target = float(true_a[item_index, parameter] if parameter < model.D else true_d[item_index])
            reference_values.append(target)
            fig.add_hline(y=target, line_color='#16a34a', line_dash='dot',
                          exclude_empty_subplots=False, row=row, col=col)
        if reference_a is not None and reference_d is not None:
            target = float(reference_a[item_index, parameter] if parameter < model.D else reference_d[item_index])
            reference_values.append(target)
            fig.add_hline(y=target, line_color='#9333ea', line_dash='dash',
                          exclude_empty_subplots=False, row=row, col=col)
        all_values = np.r_[y, reference_values]
        lower, upper = float(np.min(all_values)), float(np.max(all_values))
        padding = max((upper-lower)*.15, .05)
        # Extra headroom for the stage labels when stages are shaded.
        y_range = [lower-padding, upper+padding*(2.2 if spans else 1)]
        ranges.append(y_range)
        fig.add_trace(go.Scatter(x=cycles, y=y, mode='lines', name='Full recorded path',
            line=dict(color='#94a3b8', width=2), showlegend=panel==0,
            hovertemplate=f'{label}: %{{y:.6f}}<br>{axis_label} %{{x}}<extra></extra>'), row=row, col=col)
        indices.append(len(fig.data))
        fig.add_trace(go.Scatter(x=[cycles[selected]], y=[y[selected]],
            mode='markers', name='Current recorded value', marker=dict(color='#2563eb', size=12),
            customdata=[[changes[selected, panel]]], showlegend=panel==0, cliponaxis=False,
            hovertemplate=f'{label}: %{{y:.6f}}<br>{axis_label} %{{x}}<br>Actual update: %{{customdata[0]:+.6f}}<extra></extra>'), row=row, col=col)
        indices.append(len(fig.data))
        fig.add_trace(go.Scatter(x=[cycles[selected], cycles[selected]], y=y_range,
            mode='lines', line=dict(color='#2563eb', width=1, dash='dot'),
            hoverinfo='skip', showlegend=False), row=row, col=col)
        fig.update_xaxes(title_text=axis_label, range=x_range, autorange=False, row=row, col=col)
        fig.update_yaxes(title_text=label, title_standoff=4, range=y_range, autorange=False, row=row, col=col)

    # Reference lines are named in the legend (in-plot labels collided with the path and each other).
    # These legend-only traces are appended after the animated traces so frame indices are unchanged.
    if true_a is not None and true_d is not None:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='True (simulation)',
                                 line=dict(color='#16a34a', dash='dot')))
    if reference_a is not None and reference_d is not None:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='R mirt endpoint',
                                 line=dict(color='#9333ea', dash='dash')))

    def title(position):
        # Current values live in the title (floating labels collided with the axes in narrow panels).
        current = '  ·  '.join(f'{label} = {values[position, k]:.3f}' for k, label in enumerate(path['labels']))
        return f'{_title_head(position)}<br><span style="font-size:13px">{current}</span>'

    def _title_head(position):
        cycle, name = int(cycles[position]), stage[position]
        if stages == 'rm':
            status = 'SEM-averaged start' if cycle == 0 else f'gain {gains[position]:.5f}'
            return f'Item {item_index+1} · RM cycle {cycle} · {status}'
        if name == 'Start':
            return f'Item {item_index+1} · cycle 0 · random starting values'
        return (f'Item {item_index+1} · total cycle {cycle} · {name} cycle {int(stage_cycles[position])}'
                f' · gain {gains[position]:.5f}')

    # The animated player needs room below for its buttons and slider; the live chart does not.
    height = rows*300 + (200 if animate else 110)
    bottom = 110 if animate else 45
    # Title on top, then the legend, then the subplot titles: fixed pixel gaps so they never overlap.
    top = 130
    height += top - 95
    legend_y = 1 + 58 / (height - top - bottom)
    fig.update_layout(title=dict(text=title(selected), y=1 - 18 / height, yanchor='top'), height=height,
                      margin=dict(t=top, b=bottom, l=60, r=35),
                      legend=dict(orientation='h', y=legend_y), template='plotly_white')
    if animate:
        if max_frames < 2 or frame_duration <= 0:
            raise ValueError('Playback needs at least two frames and a positive frame duration.')
        playback_indices = np.unique(np.linspace(0, len(cycles)-1, min(max_frames, len(cycles)), dtype=int))
        frames = []
        for position in playback_indices:
            updates = []
            for panel in range(count):
                updates.append(go.Scatter(x=[cycles[position]], y=[values[position, panel]],
                    customdata=[[changes[position, panel]]]))
                updates.append(go.Scatter(x=[cycles[position], cycles[position]], y=ranges[panel]))
            frames.append(go.Frame(name=str(int(cycles[position])), data=updates,
                                   traces=indices, layout=go.Layout(title_text=title(position))))
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(type='buttons', direction='left', x=0, y=-.22,
                buttons=[dict(label='Play', method='animate', args=[None, dict(
                    frame=dict(duration=frame_duration, redraw=False), transition=dict(duration=0), fromcurrent=True)]),
                    dict(label='Pause', method='animate', args=[[None], dict(mode='immediate',
                        frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
                    dict(label='Restart', method='animate', args=[[frames[0].name], dict(mode='immediate',
                        frame=dict(duration=0, redraw=False), transition=dict(duration=0))])])],
            sliders=[dict(active=0, x=0, y=-.06, len=1,
                currentvalue=dict(prefix=f'Recorded {axis_label}: '),
                steps=[dict(label=frame.name, method='animate', args=[[frame.name], dict(mode='immediate',
                    frame=dict(duration=0, redraw=False), transition=dict(duration=0))]) for frame in frames])])
    return fig
