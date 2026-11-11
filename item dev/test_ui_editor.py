"""Focused sequential-preview privacy and navigation regressions."""
import unittest
from pathlib import Path
import pandas  # Preload before AppTest's render deadline on a cold local environment.
from streamlit.testing.v1 import AppTest

ROOT = str(Path(__file__).resolve().parent)


class PreviewTests(unittest.TestCase):
    def app(self):
        code = f'''import sys
sys.path.insert(0, {ROOT!r})
import streamlit as st
from core import new_case
from ui_editor import _preview
if "case" not in st.session_state:
 c = new_case("unfolding", example=True)
 c["facts"] = [
  {{"id":"shared_fact","name":"Shared fact","value":"VISIBLE_SHARED_CONTEXT","disclosed_at":"shared"}},
  {{"id":"later_fact","name":"Later result","value":"HIDDEN_FUTURE_RESULT","disclosed_at":c["stages"][1]["id"]}},
 ]
 c["stages"][0]["transition"] = "AUTHORED_TRANSITION_TO_STAGE_TWO"
 c["stages"][1]["new_information"] = "NEW_INFORMATION_STAGE_TWO"
 c["stages"][1]["elapsed"] = "AUTHOR_TIME_INTERVAL"
 st.session_state.case = c
c = st.session_state.case
_preview(c, c["id"] + "::")
'''
        return AppTest.from_string(code, default_timeout=20).run()

    @staticmethod
    def visible(app):
        types = [app.markdown, app.caption, app.info, app.success]
        return "\n".join(str(e.value) for group in types for e in group)

    def test_initial_preview_shows_shared_facts_and_hides_future(self):
        app = self.app()
        self.assertFalse(app.exception)
        visible = self.visible(app)
        self.assertIn("VISIBLE_SHARED_CONTEXT", visible)
        self.assertNotIn("HIDDEN_FUTURE_RESULT", visible)
        self.assertNotIn("NEW_INFORMATION_STAGE_TWO", visible)
        self.assertNotIn("AUTHORED_TRANSITION_TO_STAGE_TWO", visible)
        self.assertNotIn("Keyed ·", visible)
        advance = next(b for b in app.button if b.label == "Advance and lock this stage")
        self.assertTrue(advance.disabled)

    def test_advance_reveals_new_evidence_and_locks_prior_response(self):
        app = self.app()
        case = app.session_state["case"]
        first = case["items"][0]
        key = next(o["id"] for o in first["options"] if o["correct"])
        app.radio[0].set_value(key).run()
        next(b for b in app.button if b.label == "Submit response").click().run()
        self.assertIn("Preview score", self.visible(app))
        next(b for b in app.button if b.label == "Advance and lock this stage").click().run()
        visible = self.visible(app)
        self.assertIn("HIDDEN_FUTURE_RESULT", visible)
        self.assertIn("NEW_INFORMATION_STAGE_TWO", visible)
        self.assertIn("AUTHORED_TRANSITION_TO_STAGE_TWO", visible)
        self.assertIn("AUTHOR_TIME_INTERVAL", visible)
        self.assertTrue(app.radio[0].disabled)
        self.assertFalse(app.radio[1].disabled)
        self.assertNotIn(case["stages"][2]["title"] + " FUTURE_SECRET", visible)
        next(b for b in app.button if b.label == "Restart learner preview").click().run()
        self.assertFalse(app.radio[0].disabled)
        self.assertNotIn("HIDDEN_FUTURE_RESULT", self.visible(app))
        self.assertNotIn("Preview score", self.visible(app))


if __name__ == "__main__":
    unittest.main()
