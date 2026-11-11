"""Family medicine Item Studio — three complete manual authoring modes."""
from __future__ import annotations

from copy import deepcopy
import json

import streamlit as st

import core
import storage
from ai_adapter import get_config
from ui_editor import refresh_widgets, render_editor
from ui_assistant import render_assistant

st.set_page_config(page_title='Item Studio · Family medicine', page_icon='◈', layout='wide', initial_sidebar_state='auto')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#173c3b; --muted:#67807b; --accent:#26776e; }
html, body, p, label, input, textarea {font-family:'DM Sans',sans-serif;}
h1,h2,h3 {font-family:'Manrope',sans-serif !important; letter-spacing:-.035em;}
.stApp {background:#fafbf8; color:#203b39;}
[data-testid="stSidebar"] {background:#eef3ef; border-right:1px solid #dce6dd;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#19433b;}
.block-container {padding-top:2.2rem; padding-bottom:4rem; max-width:1600px;}
.eyebrow {font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; color:#648178; font-weight:700; margin-bottom:10px;}
.hero-title {font-family:'Manrope',sans-serif; font-size:2.25rem; font-weight:800; letter-spacing:-.055em; color:#193e36; margin:0;}
.hero-sub {color:#74867f; font-size:.95rem; margin:.4rem 0 1.5rem;}
.quiet-pill {border:1px solid #cdded5; color:#2f6252; padding:6px 10px; border-radius:6px; font-size:.72rem; display:inline-block; background:#f0f6f1; margin-right:7px;}
[data-testid="stMetric"] {background:white;border:1px solid #e0e8df;border-radius:10px;padding:12px 18px;}
[data-testid="stMetricValue"] {font-size:1.4rem;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:18px; border-bottom:1px solid #dbe5dc;}
[data-testid="stTabs"] [data-baseweb="tab"] {padding:12px 3px; font-weight:600;}
div[data-testid="stExpander"] {background:#fff;border:1px solid #e0e8df;border-radius:9px;}
button[kind="primary"] {background:#26776e;border:0;}
[data-testid="stCaptionContainer"] {color:#6f817a;}
@media(max-width:850px){.hero-title{font-size:1.8rem}.block-container{padding-top:1rem}}
</style>''', unsafe_allow_html=True)


reset_widgets = refresh_widgets


def commit(case, reason='Edit draft'):
    """Persist a case. A copy edited outside the form widgets replaces the open draft."""
    current = st.session_state['itemdev_cases'].get(case['id'])
    saved = storage.save_case(case, reason)
    case.clear()
    case.update(saved)
    st.session_state['itemdev_cases'][case['id']] = case
    if current is not None and current is not case:
        # A command, proposal or dictation changed the saved case: rebuild the
        # form from it so stale on-screen values cannot overwrite the change.
        refresh_widgets(case['id'])


def add_cases(cases):
    created = [storage.import_case(case, suffix='') for case in cases]
    for case in created:
        st.session_state['itemdev_cases'][case['id']] = case
    if created:
        activate(created[-1])


def activate(case):
    st.session_state['itemdev_cases'][case['id']] = case
    st.session_state['itemdev_active'][case['mode']] = case['id']
    st.session_state.setdefault('itemdev_pending_selection', {})[case['mode']] = case['id']


try:
    if 'itemdev_cases' not in st.session_state:
        initial = storage.load_library()
        st.session_state['itemdev_cases'] = initial['cases']
        st.session_state['itemdev_active'] = {
            mode: next((cid for cid, c in initial['cases'].items() if c['mode'] == mode), None)
            for mode in core.MODES
        }
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

action = st.session_state.pop('itemdev_requested_action', None)
if action:
    cid = action.get('case_id')
    if action['action'] in ('undo', 'redo') and cid:
        recovered = storage.recover(cid, action['action'])
        if recovered:
            reset_widgets(cid)
            activate(recovered)
            st.session_state['itemdev_notice'] = 'Revision recovered. Content checks and review status were refreshed.'
    elif action['action'] == 'open_mode':
        if action.get('mode') in core.MODES:
            st.session_state['itemdev_mode_tab'] = core.MODES[action['mode']]
    elif action['action'] in ('preview', 'validate'):
        if cid in st.session_state['itemdev_cases']:
            target_mode = st.session_state['itemdev_cases'][cid]['mode']
            st.session_state['itemdev_mode_tab'] = core.MODES[target_mode]
            st.session_state[cid + '::editor_tab'] = '▷ Learner preview' if action['action'] == 'preview' else '✓ Validate & review'

with st.sidebar:
    st.markdown('<div class="eyebrow">Authoring workspace</div>', unsafe_allow_html=True)
    st.markdown('## ◈ Item Studio')
    st.caption('Family medicine · ' + ('cloud session' if storage.hosted_mode() else 'local workspace'))
    st.divider()
    ai_enabled = st.toggle('AI assistance', value=False, key='itemdev_ai_enabled', help='Optional. All manual authoring, guidance, and structural validation work with AI off.')
    if ai_enabled:
        config = get_config()
        st.caption('Configured · connection not yet verified' if config['configured'] else 'Not configured · manual tools remain available')
    else:
        st.caption('Off · no AI-provider requests')
    st.divider()
    st.markdown('**A guide for every decision**')
    st.caption('Blueprint → objective → clinical reasoning → options → review')
    with st.expander('Clinical reasoning guide'):
        for phase in core.REASONING_PHASES:
            st.markdown('**' + phase + '**')
            st.caption(core.REASONING_GUIDANCE[phase])
        st.caption('Proposed teaching synthesis; phases can recur and do not prescribe the item count.')
    with st.expander('Clinical appraisal guide'):
        for dimension in core.APPRAISAL_DIMENSIONS:
            st.markdown('**' + dimension + '**')
            st.caption(core.APPRAISAL_GUIDANCE[dimension])
        st.caption('Local adaptation informed by IMPACT and RACCCA; not validated learner subscales.')
    with st.expander('Misconception guide'):
        selected_error = st.selectbox('Reasoning error to explore', list(core.MISCONCEPTIONS), key='itemdev_guide_error')
        st.write(core.MISCONCEPTIONS[selected_error])
        st.caption('An option selection alone does not establish the cause of an error.')
    st.divider()
    st.markdown('**Import an existing draft**')
    uploaded = st.file_uploader('Case JSON', type=['json'], key='itemdev_import')
    st.caption('A single case, a downloaded variant set, or a library backup.')
    if st.button('Import as new drafts', disabled=uploaded is None, width="stretch"):
        try:
            imported = storage.import_cases(uploaded.getvalue())
            for entry in imported:
                activate(entry)
            st.session_state['itemdev_mode_tab'] = core.MODES[imported[-1]['mode']]
            st.session_state['itemdev_notice'] = f'Imported {len(imported)} case(s) as new drafts. Imported approvals are kept as history only and need local reassessment.'
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f'Nothing was imported. {exc}')
    st.download_button('Back up all drafts (JSON)', json.dumps({'schema_version': core.SCHEMA_VERSION, 'cases': list(st.session_state['itemdev_cases'].values())}, indent=2, ensure_ascii=False), file_name='item-studio-library.json', mime='application/json', key='itemdev_backup', width="stretch")
    st.caption(storage.persistence_description())
    st.caption('No automatic publication or clinical approval.')

st.markdown('<div class="eyebrow">From clinical expertise to better questions</div><div class="hero-title">A thoughtful space to develop items.</div><div class="hero-sub">Shape the clinical decision, make the reasoning visible, and review every option.</div>', unsafe_allow_html=True)
st.markdown('<span class="quiet-pill">3 authoring modes</span><span class="quiet-pill">Single best answer + multiple selection</span><span class="quiet-pill">Manual first</span>', unsafe_allow_html=True)
if storage.hosted_mode():
    st.info(storage.persistence_description())
if notice := st.session_state.pop('itemdev_notice', None):
    st.info(notice)
with st.expander('Library dashboard · all drafts and review states'):
    rows = []
    for entry in st.session_state['itemdev_cases'].values():
        findings = core.validate_case(entry)
        outdated = sum(bool(r.get('requires_reassessment')) for r in entry.get('reviews', []))
        rows.append({
            'Type': core.MODES[entry['mode']], 'Title': entry['title'], 'Review state': entry.get('review_status', 'Draft'),
            'Revision': entry.get('version', 1), 'Items': len(entry['items']), 'Stages': len(entry['stages']),
            'Formats': ' + '.join(sorted({i['format'] for i in entry['items']})),
            'Blueprint': ' → '.join(x for x in (entry['blueprint'].get('domain'), entry['blueprint'].get('subdomain')) if x) or 'Not mapped',
            'Structural errors': sum(f['severity'] == 'error' for f in findings),
            'Missing input': sum(f['severity'] == 'warning' for f in findings),
            'Recorded reviews': len(entry.get('reviews', [])), 'Reviews needing reassessment': outdated,
            'Provenance': entry.get('provenance', ''), 'Updated (UTC)': entry.get('updated_at', ''),
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption('Review states record decisions made by named reviewers. No learner response data are stored, so no item statistics are shown.')
    archived = storage.archived_cases()
    if archived:
        restore_col, button_col = st.columns([4, 1])
        chosen = restore_col.selectbox('Archived drafts', list(archived), format_func=lambda cid: f"{core.MODES.get(archived[cid].get('mode'), '')} · {archived[cid].get('title', cid)}", key='itemdev_archived_pick')
        button_col.write('')
        if button_col.button('Restore', key='itemdev_restore', width="stretch"):
            restored = storage.restore_case(chosen)
            if restored:
                activate(restored)
                st.rerun()
st.write('')

mode_descriptions = {
    'standalone': 'Build a focused clinical question or a set under the same unchanged vignette.',
    'unfolding': 'Develop a consistent patient story with linked decisions and controlled disclosure.',
    'appraisal': 'Ask the learner to evaluate a preliminary clinical assessment or care recommendation.',
}


def render_mode(mode):
    cases = st.session_state['itemdev_cases']
    matching = [cid for cid, c in cases.items() if c['mode'] == mode]
    if not matching:
        created = storage.save_case(core.new_case(mode), 'Create blank draft')
        activate(created)
        matching = [created['id']]
    active_id = st.session_state['itemdev_active'].get(mode)
    pending = st.session_state.get('itemdev_pending_selection', {}).pop(mode, None)
    if pending in matching:
        active_id = pending
        st.session_state[f'library:{mode}'] = pending
    if active_id not in matching:
        active_id = matching[0]
    if st.session_state.get(f'library:{mode}') not in matching:
        st.session_state[f'library:{mode}'] = active_id
    st.caption(mode_descriptions[mode])
    selector, blank_col, example_col = st.columns([6, 1.6, 1.6])
    with selector:
        selected_id = st.selectbox('Draft library', matching, format_func=lambda cid: cases[cid]['title'], key=f'library:{mode}')
    st.session_state['itemdev_active'][mode] = selected_id
    with blank_col:
        st.write('')
        if st.button('＋ Blank draft', key=f'new:{mode}', width="stretch"):
            created = storage.save_case(core.new_case(mode), 'Create blank draft')
            activate(created)
            st.rerun()
    with example_col:
        st.write('')
        if st.button('Load example', key=f'example:{mode}', width="stretch"):
            created = storage.save_case(core.new_case(mode, example=True), 'Load educational example')
            activate(created)
            st.rerun()
    case = cases[selected_id]
    tools = st.columns([1, 1, 1, 1, 4])
    if tools[0].button('↶ Undo', key=f'undo:{mode}', width="stretch"):
        recovered = storage.recover(case['id'])
        if recovered:
            reset_widgets(case['id'])
            activate(recovered)
            st.rerun()
        else:
            st.toast('No earlier revision in this draft.')
    if tools[1].button('↷ Redo', key=f'redo:{mode}', width="stretch"):
        recovered = storage.recover(case['id'], 'redo')
        if recovered:
            reset_widgets(case['id'])
            activate(recovered)
            st.rerun()
        else:
            st.toast('No revision to redo.')
    if tools[2].button('Duplicate', key=f'clone:{mode}', width="stretch"):
        created = storage.clone_case(case)
        activate(created)
        st.rerun()
    if tools[3].button('Archive', key=f'archive:{mode}', width="stretch", help='Removes this draft from the working library. It stays recoverable under Library dashboard → Archived drafts.'):
        storage.archive_case(case['id'])
        st.session_state['itemdev_cases'].pop(case['id'], None)
        st.session_state['itemdev_active'][mode] = None
        st.session_state.pop(f'library:{mode}', None)
        st.session_state['itemdev_notice'] = 'Draft archived. Restore it from the Library dashboard.'
        st.rerun()
    tools[4].caption(f"Revision {case.get('version', 1)} · {case.get('review_status', 'Draft')} · {len(case['items'])} items / {len(case['stages'])} stages")
    editor_col, guide_col = st.columns([3.5, 1.25], gap='large')
    with editor_col:
        render_editor(case, commit, add_cases)
    with guide_col:
        render_assistant(case, ai_enabled, commit)
    commit(case, 'Edit draft')
    st.divider()
    exports = st.columns(3)
    exports[0].download_button('Export complete JSON', json.dumps(case, indent=2, ensure_ascii=False), file_name=f"{mode}-case.json", mime='application/json', key=f'json:{mode}', width="stretch")
    exports[1].download_button('Author copy · Markdown', core.export_markdown(case), file_name=f'{mode}-author.md', mime='text/markdown', key=f'md:{mode}', width="stretch")
    exports[2].download_button('Learner copy · Markdown', core.export_markdown(case, include_answers=False), file_name=f'{mode}-learner.md', mime='text/markdown', key=f'learner:{mode}', width="stretch")
    st.caption('Text fields save when committed by Enter or leaving the field. Tables use their Apply button. Sources and clinical content remain subject to SME review.')
    with st.expander('Version history'):
        events = storage.load_library().get('events', {}).get(case['id'], [])
        st.dataframe(list(reversed(events)), hide_index=True, width="stretch")


for mode, tab in zip(core.MODES, st.tabs(list(core.MODES.values()), key='itemdev_mode_tab', on_change='rerun')):
    with tab:
        render_mode(mode)
