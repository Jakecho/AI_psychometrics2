"""Streamlit Community Cloud entry point for LOFT Sequential ATA — OR-Tools CP-SAT (V2).

The app lives at the repository root ("LOFT Sequential ATA -OR-Tools CP-SAT_V2.py"). This folder exists so
Community Cloud installs only the small requirements.txt next to this file, instead of the root
requirements.txt shared by the repository's other apps.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "LOFT Sequential ATA -OR-Tools CP-SAT_V2.py"), run_name="__main__")
