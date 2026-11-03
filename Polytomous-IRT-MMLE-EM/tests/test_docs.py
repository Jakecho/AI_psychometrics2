"""Documented numerical examples must equal what the engine produces now (acceptance item 16)."""
import importlib.util

from conftest import ROOT


def test_numerical_examples_document_matches_engine():
    spec = importlib.util.spec_from_file_location("mk", ROOT / "docs" / "make_numerical_examples.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    assert (ROOT / "docs" / "NUMERICAL_EXAMPLES.md").read_text() == mk.build()
