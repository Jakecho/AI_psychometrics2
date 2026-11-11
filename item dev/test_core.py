"""Domain regressions: scoring, disclosure, import safety and constrained assembly."""
from copy import deepcopy
import json
import unittest

from core import (
    APPRAISAL_DIMENSIONS, MODES, export_markdown, generate_variants, new_case,
    new_item, new_option, normalize_import, resize_items, resize_stages,
    score_item, touch_case, validate_case, visible_facts,
)


class DomainTests(unittest.TestCase):
    def item(self, multiple=False):
        item = new_item("stage1")
        item["options"] = [new_option("A", True, "A evidence"), new_option("B", multiple, "B evidence"),
                           new_option("C", False, "C evidence"), new_option("D", False, "D evidence")]
        item["format"] = "MS" if multiple else "SBA"
        return item

    def test_example_modes_have_valid_shape_and_no_structural_errors(self):
        for mode in MODES:
            case = normalize_import(new_case(mode, example=True))
            self.assertTrue(case["items"])
            self.assertFalse([f for f in validate_case(case) if f["severity"] == "error"], mode)
            self.assertTrue(any(f["code"] == "EXPERT_REVIEW" for f in validate_case(case)))

    def test_exact_ms_requires_exact_set(self):
        item = self.item(multiple=True)
        keys = [o["id"] for o in item["options"][:2]]
        self.assertEqual(score_item(item, keys)["score"], 1)
        self.assertEqual(score_item(item, keys[:1])["score"], 0)
        self.assertEqual(score_item(item, [o["id"] for o in item["options"]])["score"], 0)
        self.assertEqual(score_item(item, [])["score"], 0)

    def test_partial_ms_explicit_penalty_and_bounds(self):
        item = self.item(multiple=True)
        item["scoring"] = "partial"
        opts = item["options"]
        self.assertEqual(score_item(item, [opts[0]["id"]])["score"], 0.5)
        self.assertEqual(score_item(item, [opts[0]["id"], opts[2]["id"]])["score"], 0)
        self.assertEqual(score_item(item, [opts[0]["id"], opts[1]["id"], opts[2]["id"]])["score"], 0.5)
        self.assertEqual(score_item(item, [o["id"] for o in opts])["score"], 0)
        self.assertEqual(score_item(item, [opts[3]["id"]])["score"], 0)
        self.assertEqual(score_item(item, [])["score"], 0)

    def test_sba_rejects_multiple_selected_and_unknown_options(self):
        item = self.item()
        with self.assertRaisesRegex(ValueError, "more than one"):
            score_item(item, [o["id"] for o in item["options"][:2]])
        with self.assertRaisesRegex(ValueError, "unknown option"):
            score_item(item, ["foreign"])

    def test_duplicate_selected_identifier_does_not_increase_credit(self):
        item = self.item()
        self.assertEqual(score_item(item, [item["options"][0]["id"]] * 3)["score"], 1)

    def test_key_count_differs_by_response_format(self):
        case = new_case(example=True)
        case["items"][0]["options"][1]["correct"] = True
        self.assertIn("SBA_KEY", {f["code"] for f in validate_case(case)})
        case["items"][0]["format"] = "MS"
        self.assertNotIn("SBA_KEY", {f["code"] for f in validate_case(case)})
        case["items"][0]["select_count"] = 1
        self.assertIn("SELECT_COUNT", {f["code"] for f in validate_case(case)})

    def test_item_resize_preserves_history_ids_and_does_not_add_stages(self):
        case = new_case("unfolding", example=True)
        original = deepcopy(case["items"])
        resize_items(case, 2)
        self.assertEqual(len(case["stages"]), 4)
        self.assertEqual(case["archived_items"], original[2:])
        resize_items(case, 4)
        self.assertEqual([i["id"] for i in case["items"]], [i["id"] for i in original])
        resize_items(case, 7)
        self.assertEqual(len(case["items"]), 7)
        self.assertEqual(len(case["stages"]), 4)
        self.assertEqual(case["patient"], new_case("unfolding", example=True)["patient"])

    def test_stage_shrink_retains_dependencies_for_explicit_repair(self):
        case = new_case("unfolding", example=True)
        original_ids = [s["id"] for s in case["stages"]]
        resize_stages(case, 1)
        self.assertEqual(len(case["items"]), 4)
        self.assertIn("ITEM_STAGE", {f["code"] for f in validate_case(case)})
        resize_stages(case, 4)
        self.assertEqual([s["id"] for s in case["stages"]], original_ids)
        self.assertNotIn("ITEM_STAGE", {f["code"] for f in validate_case(case)})

    def test_resize_invalidates_current_approval_without_erasing_review(self):
        case = new_case()
        case["review_status"] = "Approved for use"
        case["reviews"] = [{"reviewer": "Independent reviewer", "decision": "Approved", "version": 1}]
        resize_items(case, 2)
        self.assertEqual(case["review_status"], "Draft")
        self.assertTrue(case["reviews"][0]["requires_reassessment"])
        self.assertFalse(case["reviews"][0]["valid"])
        self.assertEqual(case["reviews"][0]["reviewer"], "Independent reviewer")

    def test_invalid_counts_are_rejected(self):
        for value in (0, -2, True, 2.5):
            with self.assertRaises(ValueError):
                resize_items(new_case(), value)

    def test_future_fact_does_not_become_visible_to_earlier_item(self):
        case = new_case("unfolding")
        fact = {"id": "future", "name": "New result", "value": "result", "disclosed_at": case["stages"][2]["id"]}
        case["facts"] = [fact]
        case["items"][0]["required_facts"] = ["future"]
        self.assertEqual(visible_facts(case, case["stages"][0]["id"]), [])
        self.assertEqual(visible_facts(case, case["stages"][2]["id"]), [fact])
        self.assertIn("FUTURE_FACT", {f["code"] for f in validate_case(case)})

    def test_bad_nested_imports_report_field_path(self):
        for path, replacement in [("options", "bad"), ("claims", ["bad"]), ("ratings", []), ("required_facts", [5])]:
            case = new_case()
            case["items"][0][path] = replacement
            with self.assertRaisesRegex(ValueError, path):
                normalize_import(case)
        case = new_case()
        case["items"][0]["options"][0]["correct"] = "false"
        with self.assertRaisesRegex(ValueError, "correct"):
            normalize_import(case)

    def test_import_never_allows_nonfinite_or_nonjson_data(self):
        case = new_case()
        case["unexpected"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_import(case)
        case["unexpected"] = {1, 2}
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            normalize_import(case)

    def test_import_preserves_extensions_roundtrip(self):
        case = new_case("appraisal", example=True)
        case["writer_extension"] = {"notes": ["kept"]}
        loaded = normalize_import(json.dumps(case))
        self.assertEqual(loaded, case)
        self.assertEqual(normalize_import({"case": case}), case)

    def feature_case(self):
        case = new_case(example=True)
        case["patient"] += " Authored name: {{name}}."
        case["features"] = [{"id": "feature_name", "name": "name", "values": ["A", "B", "C"]}]
        return case

    def test_variants_respect_compound_constraints_and_trace_model(self):
        case = self.feature_case()
        case["constraints"] = [{"all": [{"feature": "name", "operator": "in", "value": ["A", "B"]}, {"not": {"feature": "name", "value": "C"}}]}]
        variants = generate_variants(case, 2, seed=42)
        self.assertEqual({v["generated_from"]["assignment"]["name"] for v in variants}, {"A", "B"})
        self.assertTrue(all("{{name}}" not in v["patient"] for v in variants))
        self.assertTrue(all(v["generated_from"]["version"] == case["version"] for v in variants))
        self.assertEqual([v["patient"] for v in generate_variants(case, 2, 42)], [v["patient"] for v in variants])

    def test_infeasible_constraints_report_shortage_and_no_filler(self):
        case = self.feature_case()
        case["constraints"] = [{"feature": "name", "value": "never authored"}]
        with self.assertRaisesRegex(ValueError, "Only 0.*No filler"):
            generate_variants(case, 1)

    def test_unused_feature_does_not_create_fake_distinct_content(self):
        case = self.feature_case()
        case["patient"] = "Unchanged vignette"
        with self.assertRaisesRegex(ValueError, "Only 1"):
            generate_variants(case, 2)

    def test_ineligible_correct_option_cannot_be_used_in_variant(self):
        case = self.feature_case()
        case["items"][0]["options"][0]["conditions"] = [{"feature": "name", "value": "A"}]
        variant = generate_variants(case, 1)[0]
        self.assertEqual(variant["generated_from"]["assignment"]["name"], "A")
        with self.assertRaisesRegex(ValueError, "Only 1"):
            generate_variants(case, 2)

    def test_missing_distractors_are_not_filled(self):
        case = self.feature_case()
        case["items"][0]["options"][1]["eligible"] = False
        with self.assertRaisesRegex(ValueError, "eligible option pools"):
            generate_variants(case, 1)

    def test_examinee_markdown_omits_answers_and_private_review(self):
        case = new_case("appraisal", example=True)
        case["reviews"] = [{"reviewer": "PRIVATE REVIEWER", "comment": "PRIVATE COMMENT"}]
        case["items"][0]["options"][0]["rationale"] = "SECRET RATIONALE"
        learner = export_markdown(case, include_answers=False)
        self.assertNotIn("PRIVATE", learner)
        self.assertNotIn("SECRET RATIONALE", learner)
        self.assertNotIn("Keyed option", learner)
        self.assertIn(case["items"][0]["recommendation"], learner)
        author = export_markdown(case)
        self.assertIn("PRIVATE REVIEWER", author)
        self.assertIn("SECRET RATIONALE", author)

    def test_appraisal_has_all_six_local_dimensions(self):
        self.assertEqual(len(APPRAISAL_DIMENSIONS), 6)
        case = new_case("appraisal")
        codes = {f["code"] for f in validate_case(case)}
        self.assertIn("APPRAISAL_RECOMMENDATION", codes)
        self.assertIn("CLAIM_EVIDENCE", codes)


if __name__ == "__main__":
    unittest.main()
