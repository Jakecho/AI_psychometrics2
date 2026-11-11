"""Streamlit Community Cloud entrypoint with per-session draft storage.

Deploy this file, rather than app.py, for a shared hosted instance. Provider
configuration comes only from allowlisted Streamlit deployment secrets. The
standalone app.py entrypoint retains its existing local workspace behavior.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import os
from pathlib import Path
import runpy
import sys


ALLOWED_SECRET_ENV = (
    "ITEMDEV_AI_PROVIDER",
    "ITEMDEV_AI_BASE_URL",
    "ITEMDEV_AI_MODEL",
    "ITEMDEV_AI_API_KEY",
    "ITEMDEV_TRANSCRIBE_URL",
    "ITEMDEV_TRANSCRIBE_MODEL",
    "ITEMDEV_TRANSCRIBE_API_KEY",
)
APP_DIRECTORY = Path(__file__).resolve().parent


def configure_hosted_environment(
    secrets: Mapping | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Enable hosted isolation and copy only scalar string provider settings.

    The returned names contain no values and are useful for testing. No settings
    or credentials are printed. Missing secrets are normal for a manual-only app.
    ``ITEMDEV_HOSTED`` is always set here and cannot be overridden by secrets.
    """
    target = os.environ if environ is None else environ
    target["ITEMDEV_HOSTED"] = "1"
    if secrets is None:
        import streamlit as st

        secrets = st.secrets
    mapped = []
    try:
        for name in ALLOWED_SECRET_ENV:
            try:
                value = secrets[name]
            except FileNotFoundError:
                # StreamlitSecretNotFoundError subclasses FileNotFoundError.
                # st.secrets is lazy: the missing-file error occurs on lookup.
                break
            except KeyError:
                continue
            if isinstance(value, str):
                target[name] = value
                mapped.append(name)
    finally:
        # Streamlit can itself populate root-level secret strings in os.environ
        # while lazily parsing secrets. Reassert isolation after those side
        # effects, including when a deployment secret uses this flag's name.
        target["ITEMDEV_HOSTED"] = "1"
    return tuple(mapped)


def main() -> None:
    """Configure the hosted process before importing the application's storage."""
    configure_hosted_environment()
    directory = str(APP_DIRECTORY)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    runpy.run_path(str(APP_DIRECTORY / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()
