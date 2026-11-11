"""Focused tests for explicit AI gating and validated, non-mutating proposals."""

from copy import deepcopy
import io
import json
import os
import unittest
from unittest.mock import patch
import urllib.error
import wave

import ai_adapter as adapter


def original_item():
    return {
        "id": "item_1", "stage_id": "stage_1", "title": "A clinical decision",
        "phase": "Decide", "format": "SBA", "stem": "Case details",
        "lead_in": "Which next step is best?", "scoring": "exact", "select_count": None,
        "sources": [{"title": "Authored reference"}], "review_status": "Reviewed",
        "options": [
            {"id": "a", "text": "Option A", "correct": True, "rationale": "Supported by the case.", "role": "essential", "eligible": True, "conditions": [{"feature": "risk", "value": "low"}]},
            {"id": "b", "text": "Option B", "correct": False, "rationale": "Contradicted by the case.", "role": "conditional", "eligible": True},
            {"id": "c", "text": "Option C", "correct": False, "rationale": "An unnecessary intervention.", "role": "random", "eligible": True},
        ],
    }


def wav_recording(seconds=0.1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return buffer.getvalue()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def configure(self, provider="ollama"):
        os.environ.update({
            "ITEMDEV_AI_PROVIDER": provider,
            "ITEMDEV_AI_MODEL": "test-model",
            "ITEMDEV_AI_BASE_URL": "http://127.0.0.1:11434" if provider == "ollama" else "https://ai.example.test/v1",
        })

    def test_configuration_does_not_connect_or_return_credentials(self):
        self.configure()
        os.environ["ITEMDEV_AI_API_KEY"] = "server-secret"
        with patch.object(adapter, "_http") as http:
            config = adapter.get_config()
        http.assert_not_called()
        self.assertTrue(config["configured"])
        self.assertFalse(config["voice_configured"])
        self.assertTrue(config["api_key_configured"])
        self.assertNotIn("server-secret", json.dumps(config))
        self.assertNotIn("api_key", config)

    def test_unconfigured_default_is_honest(self):
        config = adapter.get_config()
        self.assertFalse(config["configured"])
        self.assertEqual(config["base_url"], "http://127.0.0.1:11434")
        self.assertFalse(config["voice_configured"])

    def test_endpoint_credentials_are_not_exposed(self):
        self.configure()
        os.environ["ITEMDEV_AI_BASE_URL"] = "https://name:private-key@example.test"
        config = adapter.get_config()
        self.assertFalse(config["configured"])
        self.assertNotIn("private-key", json.dumps(config))

    def test_disabled_requests_never_reach_network(self):
        self.configure()
        item = original_item()
        with patch.object(adapter, "_http") as http:
            for action in (
                lambda: adapter.request_assistance({}, "Review this.", False),
                lambda: adapter.propose_item({}, item, "Revise this.", False),
                lambda: adapter.transcribe_audio(b"recording", False),
            ):
                with self.assertRaisesRegex(adapter.AdapterError, "off"):
                    action()
        http.assert_not_called()

    def test_truthy_values_do_not_enable_requests(self):
        self.configure()
        with patch.object(adapter, "_http") as http:
            with self.assertRaises(adapter.AdapterError):
                adapter.request_assistance({}, "Review this.", "false")
        http.assert_not_called()

    def test_ollama_request_contract(self):
        self.configure()
        case = {"title": "Example", "patient": "Authored vignette", "credentials": "must-not-send"}
        before = deepcopy(case)
        with patch.object(adapter, "_http", return_value=(b'{"message":{"content":"Draft review."}}', "application/json")) as http:
            answer = adapter.request_assistance(case, "Review the lead-in.", True)
        request = http.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertFalse(body["stream"])
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertNotIn("must-not-send", request.data.decode())
        self.assertEqual(answer, "Draft review.")
        self.assertEqual(case, before)

    def test_compatible_request_contract_and_authorization(self):
        self.configure("compatible")
        os.environ["ITEMDEV_AI_API_KEY"] = "server-secret"
        response = {"choices": [{"message": {"content": "Draft response."}}]}
        with patch.object(adapter, "_http", return_value=(json.dumps(response).encode(), "application/json")) as http:
            adapter.request_assistance({}, "Explain.", True)
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, "https://ai.example.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer server-secret")

    def test_compatible_root_adds_v1_once(self):
        self.configure("compatible")
        os.environ["ITEMDEV_AI_BASE_URL"] = "http://localhost:9000/"
        with patch.object(adapter, "_http", return_value=(b'{"choices":[{"message":{"content":"Text"}}]}', "application/json")) as http:
            adapter.request_assistance({}, "Explain.", True)
        self.assertEqual(http.call_args.args[0].full_url, "http://localhost:9000/v1/chat/completions")

    def test_context_limit_prevents_network(self):
        self.configure()
        with patch.object(adapter, "_http") as http:
            with self.assertRaisesRegex(adapter.AdapterError, "too large"):
                adapter.request_assistance({"patient": "x" * adapter.MAX_CONTEXT_BYTES}, "Explain.", True)
        http.assert_not_called()

    def test_error_body_is_never_reflected(self):
        self.configure()
        error = urllib.error.HTTPError("https://private-endpoint", 403, "server-secret", {}, io.BytesIO(b"sensitive body"))
        with patch("urllib.request.build_opener") as build:
            build.return_value.open.side_effect = error
            with self.assertRaises(adapter.AdapterError) as caught:
                adapter.request_assistance({}, "Explain.", True)
        self.assertNotIn("server-secret", str(caught.exception))
        self.assertNotIn("private-endpoint", str(caught.exception))
        self.assertNotIn("sensitive body", str(caught.exception))

    def test_json_response_shape_failure_is_actionable(self):
        self.configure()
        with patch.object(adapter, "_http", return_value=(b'{"unexpected":true}', "application/json")):
            with self.assertRaisesRegex(adapter.AdapterError, "unexpected"):
                adapter.request_assistance({}, "Explain.", True)

    def test_proposal_preserves_scope_sources_and_constraints(self):
        item = original_item()
        before = deepcopy(item)
        proposal = deepcopy(item)
        proposal.update({"id": "hijack", "stage_id": "future", "sources": ["invented"], "action": "delete", "review_status": "Approved"})
        proposal["options"][0]["conditions"] = [{"feature": "risk", "value": "unsafe"}]
        result = adapter.validate_item_proposal(proposal, item)
        self.assertEqual(result["id"], "item_1")
        self.assertEqual(result["stage_id"], "stage_1")
        self.assertEqual(result["sources"], before["sources"])
        self.assertEqual(result["options"][0]["conditions"], before["options"][0]["conditions"])
        self.assertNotIn("action", result)
        self.assertEqual(result["review_status"], "Draft")
        self.assertEqual(result["provenance"], "AI-proposed")
        self.assertEqual(item, before)

    def test_invalid_boolean_key_rejected(self):
        item = original_item()
        item["options"][0]["correct"] = "false"
        with self.assertRaisesRegex(adapter.AdapterError, "true or false"):
            adapter.validate_item_proposal(item, original_item())

    def test_sba_second_correct_rejected(self):
        item = original_item()
        item["options"][1]["correct"] = True
        with self.assertRaisesRegex(adapter.AdapterError, "key"):
            adapter.validate_item_proposal(item, original_item())

    def test_ms_multiple_keys_and_count(self):
        item = original_item()
        item["format"] = "MS"
        item["options"][1]["correct"] = True
        item["select_count"] = 2
        result = adapter.validate_item_proposal(item, original_item())
        self.assertEqual(result["format"], "MS")
        self.assertEqual(result["selection_instruction"], "Select 2 options.")
        item["select_count"] = 1
        with self.assertRaisesRegex(adapter.AdapterError, "selection count"):
            adapter.validate_item_proposal(item, original_item())

    def test_duplicate_option_and_missing_rationale_rejected(self):
        item = original_item()
        item["options"][1]["text"] = " OPTION  A "
        with self.assertRaisesRegex(adapter.AdapterError, "duplicate"):
            adapter.validate_item_proposal(item, original_item())
        item = original_item()
        item["options"][1]["rationale"] = ""
        with self.assertRaisesRegex(adapter.AdapterError, "rationale"):
            adapter.validate_item_proposal(item, original_item())

    def test_ineligible_key_rejected(self):
        item = original_item()
        item["options"][0]["eligible"] = False
        with self.assertRaisesRegex(adapter.AdapterError, "eligible"):
            adapter.validate_item_proposal(item, original_item())

    def test_unrecognized_phase_rejected(self):
        item = original_item()
        item["phase"] = "Publish without review"
        with self.assertRaisesRegex(adapter.AdapterError, "phase"):
            adapter.validate_item_proposal(item, original_item())

    def test_propose_item_uses_structured_request_and_fenced_response(self):
        self.configure()
        item = original_item()
        with patch.object(adapter, "_chat", return_value="```json\n" + json.dumps(item) + "\n```") as chat:
            result = adapter.propose_item({"items": [item]}, item, "Refine options.", True)
        self.assertTrue(chat.call_args.kwargs["json_output"])
        self.assertEqual(result["id"], item["id"])
        self.assertEqual(result["provenance"], "AI-proposed")

    def test_transcription_configuration_independent(self):
        os.environ.update({"ITEMDEV_TRANSCRIBE_URL": "http://localhost:8000/transcribe", "ITEMDEV_TRANSCRIBE_MODEL": "local-speech"})
        config = adapter.get_config()
        self.assertFalse(config["configured"])
        self.assertTrue(config["voice_configured"])

    def test_transcription_multipart_and_result(self):
        os.environ.update({"ITEMDEV_TRANSCRIBE_URL": "http://localhost:8000/transcribe", "ITEMDEV_TRANSCRIBE_MODEL": "local-speech", "ITEMDEV_TRANSCRIBE_API_KEY": "voice-secret"})
        recording = wav_recording()
        with patch.object(adapter, "_http", return_value=(b'{"text":"Add another question."}', "application/json")) as http:
            answer = adapter.transcribe_audio(recording, True)
        request = http.call_args.args[0]
        self.assertIn(recording, request.data)
        self.assertIn(b'name="model"', request.data)
        self.assertIn(b"local-speech", request.data)
        self.assertIn(b"audio/wav", request.data)
        self.assertEqual(request.get_header("Authorization"), "Bearer voice-secret")
        self.assertEqual(answer, "Add another question.")

    def test_invalid_or_large_audio_does_not_connect(self):
        os.environ.update({"ITEMDEV_TRANSCRIBE_URL": "http://localhost:8000/transcribe", "ITEMDEV_TRANSCRIBE_MODEL": "local-speech"})
        with patch.object(adapter, "_http") as http:
            with self.assertRaisesRegex(adapter.AdapterError, "valid WAV"):
                adapter.transcribe_audio(b"not-a-wav", True)
            with self.assertRaisesRegex(adapter.AdapterError, "ten-megabyte"):
                adapter.transcribe_audio(b"x" * (adapter.MAX_AUDIO_BYTES + 1), True)
        http.assert_not_called()

    def test_exact_commands_never_guess_ambiguous_intent(self):
        expected = {
            "Undo that change.": {"action": "undo"},
            "Add another question": {"action": "add_item"},
            "Set this item to multiple-selection": {"action": "set_format", "format": "MS"},
            "Make this item single-best-answer": {"action": "set_format", "format": "SBA"},
            "Create two items under this same vignette": {"action": "resize_items", "count": 2},
            "Create 7 items": {"action": "resize_items", "count": 7},
            "Open the clinical appraisal tab": {"action": "open_mode", "mode": "appraisal"},
        }
        for command, action in expected.items():
            with self.subTest(command=command):
                self.assertEqual(adapter.parse_command(command), action)
        for command in ("delete all cases", "change to multiple choice", "create 0 items", "create 500 items", "set this to the right answer", "maybe undo later"):
            with self.subTest(command=command):
                self.assertIsNone(adapter.parse_command(command))


if __name__ == "__main__":
    unittest.main()
