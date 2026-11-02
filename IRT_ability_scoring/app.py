"""Alias entry point: `streamlit run app.py` runs the same app as IRT_ability_scoring.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("IRT_ability_scoring.py")), run_name="__main__")
