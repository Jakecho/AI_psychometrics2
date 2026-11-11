"""Assistant panel integration checks using Streamlit's in-process app runner."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


HARNESS = """
import streamlit as st
from core import new_case
from ui_assistant import render_assistant
case = st.session_state.setdefault('case', new_case('standalone', example=True))
enabled = st.checkbox('Enable AI', key='enabled')
def commit(candidate, reason):
    st.session_state['case'] = candidate
    st.session_state['reason'] = reason
render_assistant(case, enabled, commit)
"""


class AssistantPanelTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def app(self):
        at = AppTest.from_string(HARNESS, default_timeout=15)
        at.run()
        self.assertEqual(len(at.exception), 0)
        return at

    def button(self, at, label):
        return next(value for value in at.button if value.label == label)

    def test_ai_off_has_manual_guidance_and_no_recording(self):
        with patch('ui_assistant.request_assistance') as call:
            at = self.app()
        call.assert_not_called()
        self.assertEqual(len(at.get('audio_input')), 0)
        self.assertTrue(any('AI is off' in value.value for value in at.info))
        self.assertTrue(any(value.label == 'Explore a proposed reasoning error' for value in at.selectbox))

    def test_local_add_item_commits_without_ai(self):
        at = self.app()
        count = len(at.session_state['case']['items'])
        next(value for value in at.text_input if value.label == 'Local command').input('Add another question')
        self.button(at, 'Run local command').click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state['case']['items']), count + 1)
        self.assertIn('without changing the shared vignette', at.session_state['reason'])

    def test_local_format_retains_key_for_review(self):
        at = self.app()
        before = deepcopy(at.session_state['case']['items'][0]['options'])
        next(value for value in at.text_input if value.label == 'Local command').input('Set this item to multiple-selection')
        self.button(at, 'Run local command').click().run()
        self.assertEqual(len(at.exception), 0)
        item = at.session_state['case']['items'][0]
        self.assertEqual(item['format'], 'MS')
        self.assertEqual(item['options'], before)
        self.assertEqual(item['selection_instruction'], 'Select all applicable options.')

    def test_unconfigured_ai_is_explicitly_unavailable(self):
        with patch('ui_assistant.request_assistance') as call:
            at = self.app()
            at.checkbox[0].check().run()
        call.assert_not_called()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(self.button(at, 'Ask AI').disabled)
        self.assertTrue(self.button(at, 'Propose item').disabled)
        self.assertEqual(len(at.get('audio_input')), 0)
        self.assertTrue(any('not configured' in value.value for value in at.info))

    def test_stale_proposal_cannot_be_applied(self):
        at = self.app()
        case = at.session_state['case']
        prefix = case['id'] + '::ai:'
        at.session_state[prefix + 'proposal'] = deepcopy(case['items'][0])
        at.session_state[prefix + 'proposal_item_id'] = case['items'][0]['id']
        at.session_state[prefix + 'proposal_base'] = json.dumps(case, sort_keys=True, ensure_ascii=False)
        changed = deepcopy(case)
        changed['patient'] += ' New information was added.'
        at.session_state['case'] = changed
        at.checkbox[0].check().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(self.button(at, 'Apply candidate to draft').disabled)
        self.assertTrue(any('case changed' in value.value for value in at.warning))

    def test_switching_off_discards_pending_proposal(self):
        at = self.app()
        at.checkbox[0].check().run()
        case = at.session_state['case']
        prefix = case['id'] + '::ai:'
        at.session_state[prefix + 'proposal'] = deepcopy(case['items'][0])
        at.checkbox[0].uncheck().run()
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn(prefix + 'proposal', at.session_state)


if __name__ == '__main__':
    unittest.main()
