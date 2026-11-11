"""Space-free Streamlit Community Cloud entrypoint for Item Studio.

The editable application remains in ``item dev``. Keeping this entrypoint and
its minimal requirements file in a path without spaces avoids installer command
quoting issues while reusing the hosted isolation and settings configuration.
"""
from pathlib import Path
import runpy
import sys


APPLICATION_DIRECTORY = Path(__file__).resolve().parents[2] / "item dev"
HOSTED_ENTRYPOINT = APPLICATION_DIRECTORY / "cloud_entry.py"


def main():
    if not HOSTED_ENTRYPOINT.is_file():
        raise FileNotFoundError("The Item Studio application folder is missing its hosted entrypoint.")
    application_path = str(APPLICATION_DIRECTORY)
    if application_path not in sys.path:
        sys.path.insert(0, application_path)
    runpy.run_path(str(HOSTED_ENTRYPOINT), run_name="__main__")


if __name__ == "__main__":
    main()
