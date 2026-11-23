"""Local atomic draft persistence and recoverable revision history."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import tempfile

import core


def hosted_mode():
    return os.environ.get('ITEMDEV_HOSTED') == '1'


def persistence_description():
    if hosted_mode():
        return 'Cloud workspace · drafts are isolated to this browser session. Export JSON to keep a backup before refreshing, disconnecting, or redeployment.'
    return 'Local workspace · committed changes are saved to the project library with recoverable history.'


def data_dir():
    return Path(os.environ.get('ITEMDEV_DATA_DIR', Path(__file__).parent / 'data'))


def library_path():
    return data_dir() / 'library.json'


@contextmanager
def _lock():
    if hosted_mode():
        # Streamlit gives every connected browser session its own state object.
        yield
        return
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.library.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read():
    if hosted_mode():
        import streamlit as st
        return deepcopy(st.session_state.get('itemdev_hosted_library', {
            'schema_version': 1, 'cases': {}, 'history': {}, 'redo': {}, 'events': {},
        }))
    if not library_path().exists():
        return {'schema_version': 1, 'cases': {}, 'history': {}, 'redo': {}, 'events': {}}
    try:
        result = json.loads(library_path().read_text())
        if not isinstance(result, dict) or not isinstance(result.get('cases'), dict):
            raise ValueError('Invalid local library structure')
        for key in ['history', 'redo', 'events']:
            result.setdefault(key, {})
        return result
    except (ValueError, OSError) as exc:
        raise ValueError('The local library could not be read. The original file has been preserved; restore a valid JSON backup before continuing.') from exc


def _write(library):
    if hosted_mode():
        import streamlit as st
        st.session_state['itemdev_hosted_library'] = deepcopy(library)
        return
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.library-', suffix='.json', dir=root)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(library, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, library_path())
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_library():
    with _lock():
        result = _read()
        if not result['cases']:
            for mode in core.MODES:
                case = core.new_case(mode, example=True)
                result['cases'][case['id']] = case
            _write(result)
        return deepcopy(result)


def _content(case):
    ignored = {'version', 'updated_at', 'review_status', 'reviews', 'last_change', 'specialist_runs'}
    value = {k: deepcopy(v) for k, v in case.items() if k not in ignored}
    for item in value.get('items', []):
        item.pop('review_status', None)
    return value


def verify_specialist_revision(case_id, version):
    """Check the saved permissions and content are still at the inspected revision."""
    with _lock():
        current = _read()['cases'].get(case_id)
        if not current or current['version'] != version:
            raise ValueError("The saved draft or its source permissions changed. Reopen the case and inspect the new request.")


def save_case(case, reason='Edit draft', *, expected_revision=None):
    """Save a document, preserving reviews only when clinical content is unchanged."""
    with _lock():
        library = _read()
        saved = deepcopy(case)
        previous = library['cases'].get(saved['id'])
        if expected_revision is not None and (previous is None or previous['version'] != expected_revision):
            raise ValueError("The saved draft advanced to a newer revision. Reopen it before accepting changes.")
        if (previous and saved.get('version') != previous['version']
                and previous.get('retrieval_permissions') != saved.get('retrieval_permissions')):
            raise ValueError("Source permissions changed in another revision. Reopen this case before saving.")
        if expected_revision is not None:
            saved['reviews'] = deepcopy(previous.get('reviews', []))
            saved['review_status'] = previous['review_status']
            if 'empirical_records' in previous:
                saved['empirical_records'] = deepcopy(previous['empirical_records'])
        if previous is not None and 'specialist_runs' in previous:
            saved['specialist_runs'] = deepcopy(previous['specialist_runs'])
        if previous == saved:
            return deepcopy(previous)
        content_changed = previous is not None and _content(previous) != _content(saved)
        if previous is not None:
            library['history'].setdefault(saved['id'], []).append(deepcopy(previous))
            library['history'][saved['id']] = library['history'][saved['id']][-30:]
            library['redo'][saved['id']] = []
        if content_changed:
            saved['version'] = previous.get('version', 1)
            core.touch_case(saved, reason)
            if any(i.get('decision_workflow') for i in saved['items']):
                from specialist_workflow import mark_reassessment
                mark_reassessment(previous, saved, reason)
        elif previous:
            saved['version'] = previous.get('version', 1)
        saved['updated_at'] = core.timestamp()
        library['cases'][saved['id']] = saved
        library['events'].setdefault(saved['id'], []).append({
            'time': saved['updated_at'], 'reason': reason, 'version': saved.get('version', 1),
        })
        library['events'][saved['id']] = library['events'][saved['id']][-100:]
        _write(library)
        return deepcopy(saved)


def record_specialist_run(case_id, run):
    """Append a result to its original draft without changing content or undo history."""
    with _lock():
        library = _read()
        current = library['cases'].get(case_id) or library.get('archived_cases', {}).get(case_id)
        if current is None:
            raise ValueError("The source draft is unavailable.")
        history = current.setdefault('specialist_runs', [])
        if not any(entry['request_id'] == run['request_id'] for entry in history):
            history.append(deepcopy(run))
            _write(library)
        return deepcopy(current)


def recover(case_id, direction='undo'):
    with _lock():
        library = _read()
        source = 'history' if direction == 'undo' else 'redo'
        target = 'redo' if direction == 'undo' else 'history'
        stack = library[source].setdefault(case_id, [])
        if not stack:
            return None
        current = library['cases'][case_id]
        library[target].setdefault(case_id, []).append(deepcopy(current))
        restored = stack.pop()
        restored['version'] = current.get('version', 1)
        # Undo restores authored snapshots under the most recent source-sharing approvals.
        if 'retrieval_permissions' in current:
            from specialist_workflow import permitted_retrieval_providers
            permissions = deepcopy(current['retrieval_permissions'])
            restored['retrieval_permissions'] = permissions
            for item in restored['items'] + restored.get('archived_items', []):
                by_source = permissions.get(item['id'], {})
                for row in item.get('decision_workflow', {}).get('retrieved_sources', []):
                    row['providers'] = permitted_retrieval_providers(row, by_source.get(row['source_id']))
        if 'specialist_runs' in current:
            restored['specialist_runs'] = deepcopy(current['specialist_runs'])
        if 'specialist_runs' in current or any(i.get('decision_workflow') for i in restored['items'] + current['items']):
            restored['specialist_epoch'] = core.uid('epoch')
        present = {r.get('id') for r in restored.get('reviews', [])}
        for record in current.get('reviews', []):
            if record.get('id') not in present:
                restored.setdefault('reviews', []).append(deepcopy(record) | {'valid': False, 'requires_reassessment': True})
        if 'empirical_records' in current:
            restored['empirical_records'] = deepcopy(current['empirical_records'])
        core.touch_case(restored, f'{direction.title()} revision')
        library['cases'][case_id] = restored
        library['events'].setdefault(case_id, []).append({
            'time': core.timestamp(), 'reason': f'{direction.title()} revision', 'version': restored['version'],
        })
        _write(library)
        return deepcopy(restored)


def clone_case(case):
    cloned = deepcopy(case)
    cloned.update(id=core.uid('case'), title=case['title'] + ' · copy', reviews=[], version=1, review_status='Draft')
    for item in cloned.get('items', []):
        item['review_status'] = 'Draft'
    return save_case(cloned, 'Duplicate draft')


def import_cases(payload):
    """Import a single case, an assembled variant set, or an exported library.

    Every case is validated before any is saved, so a bad file changes nothing.
    """
    normalized = []
    for index, entry in enumerate(core.split_import(payload)):
        try:
            normalized.append(core.normalize_import(entry))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Case {index + 1} in the import: {exc}") from exc
    return [import_case(entry, suffix='') if entry.get('generated_from') else import_case(entry) for entry in normalized]


def archive_case(case_id):
    """Move a draft out of the active library into a recoverable archive."""
    with _lock():
        library = _read()
        case = library['cases'].pop(case_id, None)
        if case is None:
            return None
        library.setdefault('archived_cases', {})[case_id] = case
        library['events'].setdefault(case_id, []).append({
            'time': core.timestamp(), 'reason': 'Archive draft', 'version': case.get('version', 1),
        })
        _write(library)
        return deepcopy(case)


def archived_cases():
    with _lock():
        return deepcopy(_read().get('archived_cases', {}))


def restore_case(case_id):
    with _lock():
        library = _read()
        case = library.get('archived_cases', {}).pop(case_id, None)
        if case is None:
            return None
        if 'specialist_runs' in case or any(i.get('decision_workflow') for i in case['items']):
            case['specialist_epoch'] = core.uid('epoch')
            core.touch_case(case, 'Restore archived revision')
        library['cases'][case_id] = case
        library['events'].setdefault(case_id, []).append({
            'time': core.timestamp(), 'reason': 'Restore archived draft', 'version': case.get('version', 1),
        })
        _write(library)
        return deepcopy(case)


def import_case(payload, suffix=' · imported'):
    imported = core.normalize_import(payload)
    # Imported approvals are retained only as provenance, never activated locally.
    original_id = imported['id']
    imported['id'] = core.uid('case')
    imported['title'] += suffix
    imported['review_status'] = 'Draft'
    for record in imported.get('reviews', []):
        record['imported'] = True
        record['valid'] = False
        record['requires_reassessment'] = True
    for record in imported.get('empirical_records', []):
        record.setdefault('source_case_id', original_id)
        record['imported'] = True
    for item in imported.get('items', []):
        item['review_status'] = 'Draft'
    return save_case(imported, 'Import draft as new case')
