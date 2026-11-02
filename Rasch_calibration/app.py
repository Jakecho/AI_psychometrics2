"""Alias entry point: `streamlit run app.py` runs the same app as Rasch_calibration.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("Rasch_calibration.py")), run_name="__main__")
