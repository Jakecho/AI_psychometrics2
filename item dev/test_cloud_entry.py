"""Hosted entrypoint tests; no deployment secrets or live endpoints required."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import cloud_entry


class MissingSecrets(dict):
    def __getitem__(self, name):
        raise FileNotFoundError("No secrets configured")


class CloudEntrypointTests(unittest.TestCase):
    def test_hosted_isolation_is_enabled_without_secrets(self):
        environ = {}
        names = cloud_entry.configure_hosted_environment({}, environ)
        self.assertEqual(environ, {"ITEMDEV_HOSTED": "1"})
        self.assertEqual(names, ())

    def test_lazy_missing_secrets_is_supported(self):
        environ = {"ITEMDEV_AI_MODEL": "existing-config"}
        names = cloud_entry.configure_hosted_environment(MissingSecrets(), environ)
        self.assertEqual(environ["ITEMDEV_HOSTED"], "1")
        self.assertEqual(environ["ITEMDEV_AI_MODEL"], "existing-config")
        self.assertEqual(names, ())

    def test_only_allowlisted_scalar_strings_are_mapped(self):
        environ = {}
        secrets = {
            "ITEMDEV_AI_PROVIDER": "compatible",
            "ITEMDEV_AI_BASE_URL": "https://ai.example.test/v1",
            "ITEMDEV_AI_MODEL": "configured-model",
            "ITEMDEV_AI_API_KEY": "test-value-not-a-live-key",
            "ITEMDEV_TRANSCRIBE_URL": "https://speech.example.test/transcribe",
            "ITEMDEV_TRANSCRIBE_MODEL": "speech-model",
            "ITEMDEV_TRANSCRIBE_API_KEY": "test-voice-value",
            "ITEMDEV_HOSTED": "0",
            "ITEMDEV_STORAGE_PATH": "/unintended/shared/path",
            "PATH": "/unintended/path",
            "HOME": "/unintended/home",
            "other_section": {"some_key": "some_value"},
        }
        names = cloud_entry.configure_hosted_environment(secrets, environ)
        self.assertEqual(set(names), set(cloud_entry.ALLOWED_SECRET_ENV))
        self.assertEqual(set(environ), set(cloud_entry.ALLOWED_SECRET_ENV) | {"ITEMDEV_HOSTED"})
        self.assertEqual(environ["ITEMDEV_HOSTED"], "1")

    def test_nested_numeric_boolean_and_list_values_are_ignored(self):
        environ = {}
        secrets = {
            "ITEMDEV_AI_PROVIDER": {"nested": "not-scalar"},
            "ITEMDEV_AI_MODEL": 12,
            "ITEMDEV_AI_API_KEY": False,
            "ITEMDEV_TRANSCRIBE_MODEL": ["not", "scalar"],
            "ITEMDEV_AI_BASE_URL": "http://localhost:11434",
        }
        names = cloud_entry.configure_hosted_environment(secrets, environ)
        self.assertEqual(names, ("ITEMDEV_AI_BASE_URL",))
        self.assertEqual(environ, {"ITEMDEV_HOSTED": "1", "ITEMDEV_AI_BASE_URL": "http://localhost:11434"})

    def test_no_configuration_values_are_printed(self):
        with patch("builtins.print") as output:
            cloud_entry.configure_hosted_environment({"ITEMDEV_AI_API_KEY": "test-private-value"}, {})
        output.assert_not_called()

    def test_hosted_flag_survives_lazy_parser_environment_side_effects(self):
        environ = {}
        class LazySecretMapping(dict):
            def __getitem__(self, name):
                environ["ITEMDEV_HOSTED"] = "0"
                return super().__getitem__(name)
        cloud_entry.configure_hosted_environment(LazySecretMapping({"ITEMDEV_AI_MODEL": "example-model"}), environ)
        self.assertEqual(environ["ITEMDEV_HOSTED"], "1")

    def test_main_configures_before_absolute_app_execution(self):
        calls = []
        def configure():
            calls.append("configured")
        def execute(path, run_name):
            self.assertEqual(calls, ["configured"])
            self.assertEqual(Path(path), cloud_entry.APP_DIRECTORY / "app.py")
            self.assertTrue(Path(path).is_absolute())
            self.assertEqual(run_name, "__main__")
            self.assertIn(str(cloud_entry.APP_DIRECTORY), __import__('sys').path)
        with patch.object(cloud_entry, "configure_hosted_environment", side_effect=configure), patch.object(cloud_entry.runpy, "run_path", side_effect=execute), patch.object(cloud_entry.sys, "path", ["test-path"]):
            cloud_entry.main()


if __name__ == "__main__":
    unittest.main()
