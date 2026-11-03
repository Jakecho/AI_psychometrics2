"""Alias entry point: `streamlit run app.py` runs the same app as Polytomous_MMLE_EM.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("Polytomous_MMLE_EM.py")), run_name="__main__")
