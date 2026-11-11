"""End-to-end authoring workflow checks added while completing the app.

These cover behavior the earlier suites did not: edits made outside the form
widgets (local commands, AI proposals), stage tools and presets, cueing prompts,
scoring previews, variant-selection reasons, multi-case import, and archiving.

AppTest is a simulated runtime. It cannot show that a browser field refreshes;
that was checked separately in a real browser. These tests assert the mechanism
(a new widget generation) and that saved data is not overwritten afterwards.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as _pandas  # noqa: F401  (preload before AppTest's render deadline)
from streamlit.testing.v1 import AppTest

import core
import storage
from ui_editor import widget_prefix  # noqa: F401  (import check)


APP_DIR = Path(__file__).resolve().parent
AI_ENV = ("ITEMDEV_AI_PROVIDER", "ITEMDEV_AI_BASE_URL", "ITEMDEV_AI_MODEL", "ITEMDEV_AI_API_KEY",
          "ITEMDEV_TRANSCRIBE_URL", "ITEMDEV_TRANSCRIBE_MODEL", "ITEMDEV_TRANSCRIBE_API_KEY")


class IsolatedLibrary(unittest.TestCase):
    hosted = "0"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="itemdev-workflow-")
        self.addCleanup(self.temp.cleanup)
        env = {name: "" for name in AI_ENV} | {"ITEMDEV_DATA_DIR": self.temp.name, "ITEMDEV_HOSTED": self.hosted}
        self.env = patch.dict(os.environ, env)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("No provider request is expected"))
        self.network.start()
        self.addCleanup(self.network.stop)


class CoreCompletionTests(unittest.TestCase):
    def test_stage_presets_keep_content_and_only_rename_default_titles(self):
        case = core.new_case("standalone", example=True)
        item_ids = [i["id"] for i in case["items"]]
        authored = case["stages"][0]["title"]
        core.apply_stage_preset(case, 4)
        self.assertEqual(4, len(case["stages"]))
        self.assertEqual(authored, case["stages"][0]["title"])
        self.assertEqual(core.STAGE_PRESETS[4][1:], [s["title"] for s in case["stages"][1:]])
        self.assertEqual(item_ids, [i["id"] for i in case["items"]])
        core.apply_stage_preset(case, 3)
        self.assertEqual(3, len(case["stages"]))
        self.assertEqual(1, len(case["archived_stages"]))
        with self.assertRaises(ValueError):
            core.apply_stage_preset(case, 5)

    def test_duplicate_and_remove_stage_never_delete_items(self):
        case = core.new_case("unfolding", example=True)
        second = case["stages"][1]["id"]
        core.duplicate_stage(case, second)
        self.assertEqual(5, len(case["stages"]))
        self.assertEqual(4, len(case["items"]))
        self.assertNotEqual(second, case["stages"][2]["id"])
        core.remove_stage(case, second)
        self.assertEqual(4, len(case["items"]))
        self.assertEqual(second, case["archived_stages"][0]["id"])
        self.assertIn("ITEM_STAGE", {f["code"] for f in core.validate_case(case)})
        single = core.new_case("standalone")
        with self.assertRaisesRegex(ValueError, "at least one"):
            core.remove_stage(single, single["stages"][0]["id"])

    def test_scoring_patterns_expose_policy_differences(self):
        case = core.new_case("unfolding", example=True)
        item = next(i for i in case["items"] if i["format"] == "MS")
        exact = {row["Response pattern"]: row["Score"] for row in core.scoring_patterns(item)}
        self.assertEqual(1.0, exact["Exact keyed set"])
        self.assertEqual(0.0, exact["One keyed option only"])
        self.assertEqual(0.0, exact["No options selected"])
        self.assertEqual(0.0, exact["All options selected"])
        item["scoring"] = "partial"
        partial = {row["Response pattern"]: row["Score"] for row in core.scoring_patterns(item)}
        self.assertEqual(0.5, partial["One keyed option only"])
        self.assertEqual(0.5, partial["Keyed set plus one unkeyed option"])
        self.assertEqual(0.0, partial["All options selected"])
        self.assertEqual(0.0, partial["One unkeyed option only"])
        sba = {row["Response pattern"]: row["Score"] for row in core.scoring_patterns(case["items"][0])}
        self.assertEqual({"No options selected": 0.0, "Keyed option": 1.0, "An unkeyed option": 0.0}, sba)

    def test_evidence_map_respects_disclosure_and_flags_missing_stage(self):
        case = core.new_case("unfolding", example=True)
        case["facts"].append({"id": "late", "name": "Later result", "value": "LATE_VALUE", "disclosed_at": case["stages"][2]["id"]})
        rows = core.evidence_map(case)
        self.assertEqual(4, len(rows))
        self.assertNotIn("LATE_VALUE", rows[0]["Visible information"])
        self.assertNotIn("LATE_VALUE", rows[1]["Visible information"])
        self.assertIn("LATE_VALUE", rows[2]["Visible information"])
        self.assertEqual("End of case", rows[3]["What happens next"])
        case["items"][0]["stage_id"] = "gone"
        self.assertEqual("Missing stage", core.evidence_map(case)[0]["Stage"])

    def test_cueing_repetition_and_placeholder_prompts(self):
        case = core.new_case("unfolding", example=True)
        self.assertFalse({"REPEATED_OPTION", "ANSWER_CUE", "REPEATED_DECISION", "PLACEHOLDER", "MS_ALL_KEYED"}
                         & {f["code"] for f in core.validate_case(case)})
        key_text = next(o["text"] for o in case["items"][0]["options"] if o["correct"])
        case["items"][1]["stem"] = "The clinician has established: " + key_text.upper()
        case["items"][2]["options"][1]["text"] = case["items"][0]["options"][1]["text"]
        case["items"][3]["stage_id"] = case["items"][2]["stage_id"]
        case["items"][3]["lead_in"] = case["items"][2]["lead_in"]
        case["patient"] += " Aged {{age}}."
        for option in case["items"][3]["options"]:
            option["correct"] = True
        findings = core.validate_case(case)
        codes = {f["code"] for f in findings}
        self.assertTrue({"REPEATED_OPTION", "ANSWER_CUE", "REPEATED_DECISION", "PLACEHOLDER", "MS_ALL_KEYED"} <= codes)
        self.assertEqual("review", next(f["severity"] for f in findings if f["code"] == "ANSWER_CUE"))
        # An earlier item mentioning a later key is not a cue for the later item.
        clean = core.new_case("unfolding", example=True)
        later_key = next(o["text"] for o in clean["items"][2]["options"] if o["correct"])
        clean["items"][0]["stem"] = later_key
        self.assertNotIn("ANSWER_CUE", {f["code"] for f in core.validate_case(clean)})

    def feature_case(self):
        case = core.new_case("standalone", example=True)
        case["patient"] = "A {{age}}-year-old adult with a {{cough}} cough."
        case["features"] = [{"id": "f_age", "name": "age", "values": [25, 40, 70]},
                            {"id": "f_cough", "name": "cough", "values": ["mild", "hacking"]}]
        case["constraints"] = [{"feature": "age", "operator": "lt", "value": 70}]
        return case

    def test_combination_table_counts_permitted_combinations(self):
        table = core.combination_table(self.feature_case())
        self.assertEqual((6, 4, False), (table["total"], table["permitted"], table["truncated"]))
        self.assertEqual(4, sum(row["Permitted"] == "Yes" for row in table["rows"]))
        self.assertEqual(0, core.combination_table(core.new_case("standalone"))["total"])

    def test_variant_records_why_each_option_was_included_or_excluded(self):
        case = self.feature_case()
        options = case["items"][0]["options"]
        options[1]["role"] = "essential"
        options[2]["role"] = "random"
        options[3]["role"] = "random"
        options[3]["conditions"] = [{"feature": "cough", "value": "hacking"}]
        case["items"][0]["option_count"] = 3
        variants = core.generate_variants(case, 4, seed=3)
        for variant in variants:
            origin = variant["generated_from"]
            self.assertEqual(3, len(variant["items"][0]["options"]))
            self.assertEqual(1, sum(o["correct"] for o in variant["items"][0]["options"]))
            reasons = {r["option"]: r for r in origin["option_reasons"]}
            self.assertTrue(reasons[options[0]["text"]]["included"])
            self.assertIn("Essential", reasons[options[1]["text"]]["reason"])
            if origin["assignment"]["cough"] == "mild":
                self.assertFalse(reasons[options[3]["text"]]["included"])
                self.assertIn("conditions not met", reasons[options[3]["text"]]["reason"])
            self.assertEqual(3, sum(r["included"] for r in origin["option_reasons"]))
        self.assertEqual([v["items"][0]["options"] for v in core.generate_variants(case, 4, seed=3)],
                         [v["items"][0]["options"] for v in variants])
        self.assertFalse(any(f["severity"] == "error" for v in variants for f in core.validate_case(v)))

    def test_option_count_is_validated_on_import(self):
        case = self.feature_case()
        case["items"][0]["option_count"] = 1
        with self.assertRaisesRegex(ValueError, "option_count"):
            core.normalize_import(case)

    def test_split_import_accepts_case_variant_set_and_library(self):
        case = core.new_case("standalone", example=True)
        self.assertEqual(1, len(core.split_import(case)))
        self.assertEqual(1, len(core.split_import({"case": case})))
        self.assertEqual(2, len(core.split_import({"schema_version": 1, "cases": [case, case]})))
        self.assertEqual(1, len(core.split_import(json.dumps({"cases": {case["id"]: case}}))))
        with self.assertRaisesRegex(ValueError, "no cases"):
            core.split_import({"cases": []})
        with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
            core.split_import(b"{not json")


class StorageCompletionTests(IsolatedLibrary):
    def test_variant_set_round_trips_through_import_with_origin(self):
        case = CoreCompletionTests.feature_case(self)
        variants = core.generate_variants(case, 3, seed=5)
        imported = storage.import_cases(json.dumps({"schema_version": 1, "cases": variants}).encode())
        self.assertEqual(3, len(imported))
        saved = storage.load_library()["cases"]
        for original, copy in zip(variants, imported):
            self.assertIn(copy["id"], saved)
            self.assertNotEqual(original["id"], copy["id"])
            self.assertEqual(original["title"], copy["title"])
            self.assertEqual(original["patient"], copy["patient"])
            self.assertEqual(original["generated_from"], copy["generated_from"])
            self.assertEqual(case["constraints"], copy["constraints"])

    def test_bad_case_in_a_set_imports_nothing(self):
        good = core.new_case("standalone", example=True)
        bad = deepcopy(good)
        bad["items"][0]["format"] = "essay"
        before = set(storage.load_library()["cases"])
        with self.assertRaisesRegex(ValueError, "Case 2 in the import.*format"):
            storage.import_cases({"cases": [good, bad]})
        self.assertEqual(before, set(storage.load_library()["cases"]))

    def test_archive_and_restore_keep_the_draft_and_its_history(self):
        case = storage.save_case(core.new_case("appraisal", example=True), "Create")
        edited = deepcopy(case)
        edited["patient"] += " More context."
        storage.save_case(edited, "Edit")
        storage.archive_case(case["id"])
        self.assertNotIn(case["id"], storage.load_library()["cases"])
        self.assertIn(case["id"], storage.archived_cases())
        restored = storage.restore_case(case["id"])
        self.assertEqual(edited["patient"], restored["patient"])
        self.assertEqual({}, storage.archived_cases())
        self.assertEqual(case["patient"], storage.recover(case["id"])["patient"])
        self.assertIsNone(storage.archive_case("missing"))
        self.assertIsNone(storage.restore_case("missing"))


class AppWorkflowTests(IsolatedLibrary):
    def start(self, entry="app.py"):
        app = AppTest.from_file(str(APP_DIR / entry), default_timeout=60).run()
        self.ok(app)
        return app

    def ok(self, app):
        self.assertEqual([], [str(e.value) for e in app.exception])

    def case(self, app, cid):
        return app.session_state["itemdev_cases"][cid]

    def prefix(self, app, cid):
        generation = app.session_state["itemdev_widget_generation"].get(cid, 0) if "itemdev_widget_generation" in app.session_state else 0
        return f"{cid}::" + (f"r{generation}::" if generation else "")

    def command(self, app, cid, text):
        app.text_input(key=cid + "::ai:local_command").set_value(text)
        app.button(key=cid + "::ai:run_command").click().run()
        self.ok(app)

    def test_local_command_edits_are_not_reverted_by_form_widgets(self):
        app = self.start()
        cid = app.session_state["itemdev_active"]["standalone"]
        item_id = self.case(app, cid)["items"][0]["id"]
        self.command(app, cid, "Set this item to multiple-selection")
        for _ in range(2):
            self.assertEqual("MS", self.case(app, cid)["items"][0]["format"])
            self.assertEqual("MS", storage.load_library()["cases"][cid]["items"][0]["format"])
            app.run()
            self.ok(app)
        ip = self.prefix(app, cid) + item_id + "::"
        self.assertEqual("MS", app.radio(key=ip + "format").value)
        self.assertEqual("Select all applicable options.", app.text_input(key=ip + "instruction").value)
        # A later manual edit keeps the command's change.
        app.text_area(key=self.prefix(app, cid) + "patient").set_value("Edited after the command.").run()
        self.ok(app)
        saved = storage.load_library()["cases"][cid]
        self.assertEqual(("MS", "Edited after the command."), (saved["items"][0]["format"], saved["patient"]))

    def test_command_and_button_count_changes_agree_with_the_count_control(self):
        app = self.start()
        cid = app.session_state["itemdev_active"]["standalone"]
        self.command(app, cid, "create 3 items")
        p = self.prefix(app, cid)
        self.assertEqual(3, len(self.case(app, cid)["items"]))
        self.assertEqual(3, app.number_input(key=p + "count_items").value)
        app.button(key=p + "add_item").click().run()
        self.ok(app)
        self.assertEqual(4, app.number_input(key=p + "count_items").value)
        app.button(key=p + "apply_counts").click().run()
        self.ok(app)
        self.assertEqual(4, len(self.case(app, cid)["items"]))
        self.command(app, cid, "undo")
        self.assertEqual(3, len(self.case(app, cid)["items"]))
        self.assertEqual(3, app.number_input(key=self.prefix(app, cid) + "count_items").value)
        self.assertEqual(1, len(self.case(app, cid)["stages"]))

    def test_applied_ai_proposal_persists_and_is_marked(self):
        app = self.start()
        cid = app.session_state["itemdev_active"]["standalone"]
        case = self.case(app, cid)
        proposal = deepcopy(case["items"][0])
        proposal.update(title="Proposed title", lead_in="Proposed lead-in?", provenance="AI-proposed")
        prefix = cid + "::ai:"
        app.session_state[prefix + "proposal"] = proposal
        app.session_state[prefix + "proposal_item_id"] = proposal["id"]
        app.session_state[prefix + "proposal_base"] = json.dumps(case, sort_keys=True, ensure_ascii=False)
        app.toggle(key="itemdev_ai_enabled").set_value(True).run()
        self.ok(app)
        app.button(key=prefix + "apply").click().run()
        self.ok(app)
        app.run()
        self.ok(app)
        saved = storage.load_library()["cases"][cid]["items"][0]
        self.assertEqual(("Proposed title", "Proposed lead-in?", "AI-proposed"), (saved["title"], saved["lead_in"], saved["provenance"]))
        self.assertEqual("Proposed title", app.text_input(key=self.prefix(app, cid) + proposal["id"] + "::title").value)
        app.button(key="undo:standalone").click().run()
        self.ok(app)
        self.assertEqual(case["items"][0]["lead_in"], storage.load_library()["cases"][cid]["items"][0]["lead_in"])

    def test_stage_tools_and_presets_keep_items(self):
        app = self.start()
        cid = app.session_state["itemdev_active"]["standalone"]
        p = cid + "::"
        app.button(key=p + "preset_4").click().run()
        self.ok(app)
        case = self.case(app, cid)
        self.assertEqual((4, 1), (len(case["stages"]), len(case["items"])))
        self.assertEqual("Follow-up", case["stages"][3]["title"])
        self.assertEqual(4, app.number_input(key=p + "count_stages").value)
        target = case["stages"][1]["id"]
        app.selectbox(key=p + "stage_picker").set_value(target).run()
        app.button(key=p + target + "::duplicate_stage").click().run()
        self.ok(app)
        self.assertEqual(5, len(self.case(app, cid)["stages"]))
        app.button(key=p + target + "::add_stage").click().run()
        self.ok(app)
        self.assertEqual(6, len(self.case(app, cid)["stages"]))
        item_id = self.case(app, cid)["items"][0]["id"]
        app.selectbox(key=p + item_id + "::stage").set_value(target).run()
        self.ok(app)
        app.selectbox(key=p + "stage_picker").set_value(target).run()
        app.button(key=p + target + "::remove_stage").click().run()
        self.ok(app)
        case = self.case(app, cid)
        self.assertEqual((5, 1), (len(case["stages"]), len(case["items"])))
        self.assertEqual(target, case["items"][0]["stage_id"])
        self.assertTrue(any("disclosure stage is missing" in e.value for e in app.error))
        self.assertTrue(any("Not shown because" in w.value for w in app.warning))
        app.button(key="undo:standalone").click().run()
        self.ok(app)
        self.assertEqual(6, len(self.case(app, cid)["stages"]))

    def test_variants_assemble_report_shortage_and_join_the_library(self):
        source = CoreCompletionTests.feature_case(self)
        saved = storage.save_case(source, "Create")
        app = self.start()
        cid = saved["id"]
        app.selectbox(key="library:standalone").set_value(cid).run()
        self.ok(app)
        p = cid + "::"
        app.number_input(key=p + "variant_count").set_value(4)
        app.button(key=p + "assemble").click().run()
        self.ok(app)
        self.assertEqual(4, len(app.session_state[p + "assembled_variants"]))
        app.number_input(key=p + "variant_count").set_value(5)
        app.button(key=p + "assemble").click().run()
        self.ok(app)
        self.assertTrue(any("Only 4 distinct" in e.value for e in app.error))
        self.assertNotIn(p + "assembled_variants", app.session_state)
        app.number_input(key=p + "variant_count").set_value(2)
        app.button(key=p + "assemble").click().run()
        before = set(storage.load_library()["cases"])
        app.button(key=p + "variants_to_library").click().run()
        self.ok(app)
        added = [c for k, c in storage.load_library()["cases"].items() if k not in before]
        self.assertEqual(2, len(added))
        self.assertTrue(all(c["generated_from"]["case_id"] == cid and c["review_status"] == "Draft" and "{{" not in c["patient"] for c in added))
        self.assertEqual(source["patient"], storage.load_library()["cases"][cid]["patient"])

    def test_archive_and_restore_from_the_workspace(self):
        app = self.start()
        cid = app.session_state["itemdev_active"]["unfolding"]
        app.button(key="archive:unfolding").click().run()
        self.ok(app)
        self.assertNotIn(cid, app.session_state["itemdev_cases"])
        self.assertIn(cid, storage.archived_cases())
        self.assertEqual("unfolding", self.case(app, app.session_state["itemdev_active"]["unfolding"])["mode"])
        app.selectbox(key="itemdev_archived_pick").set_value(cid)
        app.button(key="itemdev_restore").click().run()
        self.ok(app)
        self.assertIn(cid, app.session_state["itemdev_cases"])
        self.assertEqual(cid, app.selectbox(key="library:unfolding").value)

    def test_any_number_of_items_under_one_vignette_survives_reload_and_export(self):
        app = self.start()
        for mode in ("standalone", "unfolding"):
            cid = app.session_state["itemdev_active"][mode]
            p = cid + "::"
            vignette = self.case(app, cid)["patient"]
            stages = len(self.case(app, cid)["stages"])
            for count in (1, 2, 3, 4, 6):
                with self.subTest(mode=mode, count=count):
                    app.number_input(key=p + "count_items").set_value(count)
                    app.button(key=p + "apply_counts").click().run()
                    self.ok(app)
                    saved = storage.load_library()["cases"][cid]
                    self.assertEqual((count, stages, vignette), (len(saved["items"]), len(saved["stages"]), saved["patient"]))
            saved = storage.load_library()["cases"][cid]
            first_stage = saved["stages"][0]["id"]
            self.assertGreater(sum(i["stage_id"] == first_stage for i in saved["items"]), 1)
            exported = json.loads(json.dumps(saved))
            copy = storage.import_cases(exported)[0]
            for field in ("patient", "stages", "items", "facts", "features", "constraints", "sources", "blueprint", "archived_items"):
                self.assertEqual(saved[field], copy[field])
            self.assertIn(saved["id"], core.export_markdown(saved) + saved["id"])
        reloaded = self.start()
        for mode in ("standalone", "unfolding"):
            cid = app.session_state["itemdev_active"][mode]
            self.assertEqual(6, len(reloaded.session_state["itemdev_cases"][cid]["items"]))

    def test_case_map_and_scoring_preview_render_in_every_mode(self):
        app = self.start()
        text = "\n".join(str(e.value) for e in app.markdown)
        self.assertEqual(3, text.count("Evidence-to-decision map"))
        self.assertEqual(3, text.count("Reasoning-phase coverage"))
        self.assertGreaterEqual(len(app.dataframe), 9)


class HostedWorkflowTests(AppWorkflowTests):
    """The same session workflow on the cloud entrypoint must not touch disk."""
    hosted = "1"

    def test_hosted_command_archive_and_restore_stay_in_the_session(self):
        app = self.start("cloud_entry.py")
        cid = app.session_state["itemdev_active"]["standalone"]
        self.command(app, cid, "Set this item to multiple-selection")
        app.run()
        self.assertEqual("MS", app.session_state["itemdev_hosted_library"]["cases"][cid]["items"][0]["format"])
        app.button(key="archive:standalone").click().run()
        self.ok(app)
        self.assertIn(cid, app.session_state["itemdev_hosted_library"]["archived_cases"])
        app.button(key="itemdev_restore").click().run()
        self.ok(app)
        self.assertIn(cid, app.session_state["itemdev_hosted_library"]["cases"])
        self.assertFalse((Path(self.temp.name) / "library.json").exists())


# Only the hosted-specific test should run from the subclass.
for _name in [n for n in dir(AppWorkflowTests) if n.startswith("test_")]:
    setattr(HostedWorkflowTests, _name, None)


if __name__ == "__main__":
    unittest.main()
