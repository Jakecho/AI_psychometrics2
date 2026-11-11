"""Focused UI integration checks using Streamlit's simulated AppTest runtime.

Each test owns a temporary data directory and forbids outbound AI requests.
These checks exercise committed edits and persistence; they do not certify
clinical content or replace browser layout/accessibility review.
"""
from __future__ import annotations

import os
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Hydrate the existing environment's data dependencies before timing a UI rerun.
# The actual app imports these through ui_editor, which can be slow on first use.
import pandas as _pandas
from streamlit.testing.v1 import AppTest

import core
import storage


APP_DIR = Path(__file__).resolve().parent


class ItemDevelopmentAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="itemdev-apptest-")
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {
            "ITEMDEV_DATA_DIR": self.temp.name,
            "ITEMDEV_HOSTED": "0",
            "ITEMDEV_AI_PROVIDER": "",
            "ITEMDEV_AI_BASE_URL": "",
            "ITEMDEV_AI_MODEL": "",
            "ITEMDEV_AI_API_KEY": "",
            "ITEMDEV_TRANSCRIBE_URL": "",
            "ITEMDEV_TRANSCRIBE_MODEL": "",
            "ITEMDEV_TRANSCRIBE_API_KEY": "",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Manual mode must not call an AI provider"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def start_app(self):
        app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30).run()
        self.assert_no_exception(app)
        return app

    def assert_no_exception(self, app):
        self.assertEqual([], [str(element.value) for element in app.exception])

    def by_label(self, app, element_type, label):
        matches = [element for element in getattr(app, element_type) if element.label == label]
        self.assertEqual(1, len(matches), f"Expected one {element_type} labeled {label!r}; got {[getattr(e, 'label', None) for e in getattr(app, element_type)]}")
        return matches[0]

    def test_starts_with_manual_authoring_without_network(self):
        app = self.start_app()
        labels = [tab.label for tab in app.tabs]
        for expected in ("Standalone MCQ", "Unfolding Case", "Clinical Appraisal MCQ"):
            self.assertIn(expected, labels)
        self.assertFalse(app.toggle(key="itemdev_ai_enabled").value)
        for mode in core.MODES:
            cid = app.session_state["itemdev_active"][mode]
            self.assertTrue(app.text_area(key=cid + "::patient"))
            self.assertTrue(app.button(key=cid + "::apply_counts"))

    def test_new_blank_and_example_selection_work_in_all_three_modes(self):
        app = self.start_app()
        for mode in core.MODES:
            with self.subTest(mode=mode):
                before = set(app.session_state["itemdev_cases"])
                app.button(key=f"new:{mode}").click().run()
                self.assert_no_exception(app)
                cases = app.session_state["itemdev_cases"]
                active = app.session_state["itemdev_active"][mode]
                self.assertNotIn(active, before)
                self.assertEqual(mode, cases[active]["mode"])
                self.assertEqual("", cases[active]["patient"])
                self.assertEqual(active, app.selectbox(key=f"library:{mode}").value)
                app.button(key=f"example:{mode}").click().run()
                self.assert_no_exception(app)
                active_example = app.session_state["itemdev_active"][mode]
                self.assertNotEqual(active, active_example)
                self.assertTrue(app.session_state["itemdev_cases"][active_example]["patient"])
                self.assertEqual(active_example, app.selectbox(key=f"library:{mode}").value)

    def test_both_response_formats_and_manual_edits_persist_in_all_modes(self):
        app = self.start_app()
        edited_ids = {}
        for mode in core.MODES:
            cid = app.session_state["itemdev_active"][mode]
            edited_ids[mode] = cid
            item_id = app.session_state["itemdev_cases"][cid]["items"][0]["id"]
            original_provenance = app.session_state["itemdev_cases"][cid]["items"][0].get("provenance")
            prefix = cid + "::"
            for response_format in ("SBA", "MS"):
                with self.subTest(mode=mode, response_format=response_format):
                    app.radio(key=prefix + item_id + "::format").set_value(response_format).run()
                    self.assert_no_exception(app)
                    vignette = f"Author-written {mode} vignette retained after selecting {response_format}."
                    app.text_area(key=prefix + "patient").set_value(vignette).run()
                    self.assert_no_exception(app)
                    persisted = storage.load_library()["cases"][cid]
                    self.assertEqual(vignette, persisted["patient"])
                    self.assertEqual(response_format, persisted["items"][0]["format"])
                    self.assertEqual(original_provenance, persisted["items"][0].get("provenance"))
                    core.normalize_import(persisted)
        reloaded = self.start_app()
        for mode, cid in edited_ids.items():
            self.assertEqual(f"Author-written {mode} vignette retained after selecting MS.", reloaded.session_state["itemdev_cases"][cid]["patient"])
            self.assertEqual("MS", reloaded.session_state["itemdev_cases"][cid]["items"][0]["format"])

    def test_item_counts_are_independent_and_removed_content_is_recoverable(self):
        app = self.start_app()
        cid = app.session_state["itemdev_active"]["unfolding"]
        prefix = cid + "::"
        initial = deepcopy(app.session_state["itemdev_cases"][cid])
        stages = [stage["id"] for stage in initial["stages"]]
        first_item_id = initial["items"][0]["id"]
        for count in (1, 2, 5, 1):
            with self.subTest(count=count):
                app.number_input(key=prefix + "count_items").set_value(count)
                app.button(key=prefix + "apply_counts").click().run()
                self.assert_no_exception(app)
                case = app.session_state["itemdev_cases"][cid]
                self.assertEqual(count, len(case["items"]))
                self.assertEqual(stages, [stage["id"] for stage in case["stages"]])
                self.assertEqual(first_item_id, case["items"][0]["id"])
                self.assertEqual(initial["patient"], case["patient"])
        app.button(key=prefix + "add_item").click().run()
        self.assert_no_exception(app)
        self.assertEqual(2, len(app.session_state["itemdev_cases"][cid]["items"]))
        self.assertEqual(2, app.number_input(key=prefix + "count_items").value)
        self.assertTrue(app.session_state["itemdev_cases"][cid]["archived_items"])

    def test_undo_redo_and_duplicate_refresh_visible_content(self):
        app = self.start_app()
        cid = app.session_state["itemdev_active"]["standalone"]
        prefix = cid + "::"
        prior = app.text_area(key=prefix + "patient").value
        changed = "A deliberate manual revision that must survive redo."
        app.text_area(key=prefix + "patient").set_value(changed).run()
        app.button(key="undo:standalone").click().run()
        self.assert_no_exception(app)
        # Recovery rebuilds the form under a new widget generation, so a real
        # browser cannot keep (and later re-save) the superseded field values.
        self.assertEqual(1, app.session_state["itemdev_widget_generation"][cid])
        self.assertEqual(prior, app.text_area(key=cid + "::r1::patient").value)
        self.assertEqual(prior, storage.load_library()["cases"][cid]["patient"])
        app.button(key="redo:standalone").click().run()
        self.assert_no_exception(app)
        self.assertEqual(changed, app.text_area(key=cid + "::r2::patient").value)
        app.button(key="clone:standalone").click().run()
        self.assert_no_exception(app)
        cloned_id = app.session_state["itemdev_active"]["standalone"]
        self.assertNotEqual(cid, cloned_id)
        self.assertEqual(changed, app.text_area(key=cloned_id + "::patient").value)
        self.assertEqual([], app.session_state["itemdev_cases"][cloned_id]["reviews"])

    def test_preview_submission_scores_both_formats_in_every_mode(self):
        app = self.start_app()
        for mode in core.MODES:
            cid = app.session_state["itemdev_active"][mode]
            item = app.session_state["itemdev_cases"][cid]["items"][0]
            item_id = item["id"]
            option_ids = [option["id"] for option in item["options"]]
            ip = cid + "::" + item_id + "::"
            pp = cid + "::preview::" + item_id + "::"
            for response_format in ("SBA", "MS"):
                with self.subTest(mode=mode, response_format=response_format):
                    app.radio(key=ip + "format").set_value(response_format).run()
                    keyed = option_ids[:1] if response_format == "SBA" else option_ids[:2]
                    for option_id in option_ids:
                        app.checkbox(key=ip + option_id + "::correct").set_value(option_id in keyed)
                    instruction = "Select the single best answer." if response_format == "SBA" else "Select all applicable options."
                    app.text_input(key=ip + "instruction").set_value(instruction).run()
                    self.assert_no_exception(app)
                    if response_format == "SBA":
                        app.radio(key=pp + "choice").set_value(keyed[0])
                    else:
                        for option_id in option_ids:
                            app.checkbox(key=pp + option_id).set_value(option_id in keyed)
                    app.button(key="FormSubmitter:" + pp + "response-Submit response").click().run()
                    self.assert_no_exception(app)
                    result = app.session_state[cid + "::preview_state"]["submitted"][item_id]
                    self.assertEqual(1.0, result["score"])

    def test_unfolding_preview_hides_future_information_and_locks_prior_answers(self):
        app = self.start_app()
        cid = app.session_state["itemdev_active"]["unfolding"]
        case = app.session_state["itemdev_cases"][cid]
        stage_two = case["stages"][1]["id"]
        first_item = case["items"][0]
        first_key = next(option["id"] for option in first_item["options"] if option["correct"])
        marker = "New evidence that belongs only to the second disclosure stage."
        app.selectbox(key=cid + "::stage_picker").set_value(stage_two).run()
        app.text_area(key=cid + "::" + stage_two + "::information").set_value(marker).run()
        self.assert_no_exception(app)

        def preview_text():
            mode_tab = next(tab for tab in app.tabs if tab.label == "Unfolding Case")
            preview = next(tab for tab in mode_tab.tabs if tab.label == "▷ Learner preview")
            return "\n".join(element.value for element in preview.markdown)

        self.assertNotIn(marker, preview_text())
        pp = cid + "::preview::" + first_item["id"] + "::"
        app.radio(key=pp + "choice").set_value(first_key)
        app.button(key="FormSubmitter:" + pp + "response-Submit response").click().run()
        self.assert_no_exception(app)
        app.button(key=cid + "::preview_next").click().run()
        self.assert_no_exception(app)
        self.assertIn(marker, preview_text())
        self.assertTrue(app.radio(key=pp + "choice").disabled)

    def test_cloud_entry_starts_without_disk_storage_or_ai_requests(self):
        app = AppTest.from_file(str(APP_DIR / "cloud_entry.py"), default_timeout=30).run()
        self.assert_no_exception(app)
        self.assertEqual(3, len(app.session_state["itemdev_cases"]))
        self.assertFalse(app.toggle(key="itemdev_ai_enabled").value)
        self.assertFalse((Path(self.temp.name) / "library.json").exists())
        self.assertIn("itemdev_hosted_library", app.session_state)
        self.assertTrue(any("isolated" in element.value.lower() and "session" in element.value.lower() for element in app.caption))

    def test_cloud_browser_sessions_keep_drafts_separate(self):
        first = AppTest.from_file(str(APP_DIR / "cloud_entry.py"), default_timeout=30).run()
        self.assert_no_exception(first)
        cid = first.session_state["itemdev_active"]["standalone"]
        private_text = "A draft unique to the first simulated browser session."
        first.text_area(key=cid + "::patient").set_value(private_text).run()
        self.assert_no_exception(first)
        self.assertEqual(private_text, first.session_state["itemdev_hosted_library"]["cases"][cid]["patient"])
        second = AppTest.from_file(str(APP_DIR / "cloud_entry.py"), default_timeout=30).run()
        self.assert_no_exception(second)
        self.assertNotIn(cid, second.session_state["itemdev_cases"])
        self.assertTrue(all(case["patient"] != private_text for case in second.session_state["itemdev_cases"].values()))
        first.run()
        self.assert_no_exception(first)
        self.assertEqual(private_text, first.text_area(key=cid + "::patient").value)
        self.assertFalse((Path(self.temp.name) / "library.json").exists())


class DraftPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="itemdev-storage-")
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"ITEMDEV_DATA_DIR": self.temp.name, "ITEMDEV_HOSTED": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_clinical_edit_persists_and_invalidates_approval(self):
        original = storage.save_case(core.new_case("standalone", example=True), "Create")
        original["reviews"] = [{"decision": "Approved", "reviewer": "Independent reviewer", "version": original["version"]}]
        original["review_status"] = "Approved for use"
        reviewed = storage.save_case(original, "Record review")
        edited = deepcopy(reviewed)
        edited["patient"] += " Additional patient context supplied by the author."
        saved = storage.save_case(edited, "Edit patient context")
        self.assertEqual(edited["patient"], storage.load_library()["cases"][saved["id"]]["patient"])
        self.assertEqual("Draft", saved["review_status"])
        self.assertGreater(saved["version"], reviewed["version"])
        self.assertTrue(saved["reviews"][0]["requires_reassessment"])

    def test_undo_and_redo_preserve_text_and_require_review(self):
        first = storage.save_case(core.new_case("standalone"), "Create")
        edited = deepcopy(first)
        edited["patient"] = "A distinct revised patient vignette."
        second = storage.save_case(edited, "Write vignette")
        restored = storage.recover(first["id"])
        self.assertEqual(first["patient"], restored["patient"])
        self.assertEqual("Draft", restored["review_status"])
        self.assertGreater(restored["version"], second["version"])
        reapplied = storage.recover(first["id"], "redo")
        self.assertEqual(second["patient"], reapplied["patient"])
        self.assertGreater(reapplied["version"], restored["version"])

    def test_corrupt_library_is_preserved_instead_of_overwritten(self):
        path = Path(self.temp.name) / "library.json"
        damaged = "{this is not valid JSON"
        path.write_text(damaged)
        with self.assertRaisesRegex(ValueError, "preserved"):
            storage.load_library()
        self.assertEqual(damaged, path.read_text())

    def test_failed_atomic_write_keeps_the_prior_library(self):
        case = storage.save_case(core.new_case("appraisal", example=True), "Create")
        path = Path(self.temp.name) / "library.json"
        before = path.read_text()
        changed = deepcopy(case)
        changed["patient"] = "Replacement text should not be committed after a disk error."
        with patch("storage.os.replace", side_effect=OSError("Simulated atomic replace failure")):
            with self.assertRaises(OSError):
                storage.save_case(changed, "Attempt failing write")
        self.assertEqual(before, path.read_text())
        self.assertEqual(case["patient"], json.loads(path.read_text())["cases"][case["id"]]["patient"])

    def test_import_and_duplicate_do_not_activate_external_approvals(self):
        source = core.new_case("appraisal", example=True)
        source["reviews"] = [{"decision": "Approved", "reviewer": "External reviewer", "version": 1}]
        source["review_status"] = "Approved for use"
        imported = storage.import_case(source)
        self.assertNotEqual(source["id"], imported["id"])
        self.assertEqual("Draft", imported["review_status"])
        self.assertTrue(imported["reviews"][0]["imported"])
        self.assertFalse(imported["reviews"][0]["valid"])
        clone = storage.clone_case(imported)
        self.assertEqual([], clone["reviews"])
        self.assertEqual("Draft", clone["review_status"])


if __name__ == "__main__":
    unittest.main()
