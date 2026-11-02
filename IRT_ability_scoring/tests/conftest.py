import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r_bridge.bridge import check_r  # noqa: E402

_R = check_r()


def requires(pkg):
    return pytest.mark.skipif(not (_R["available"] and _R.get(pkg)), reason=f"R/{pkg} not available")
