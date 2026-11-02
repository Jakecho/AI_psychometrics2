"""Streamlit Community Cloud entry point for the MMLE-EM Learning Lab.

The app lives at the repository root (MMLE_EM.py). This folder exists so Community
Cloud installs only the small requirements.txt next to this file, instead of the
root requirements.txt shared by the repository's other apps.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "MMLE_EM.py"), run_name="__main__")
