"""Run the reference comparison on the three validation datasets and save docs/validation_record.json.

Run from Rasch_calibration/:  python -m r_bridge.record_validation

Page 8 shows this file as the recorded local run, so the evidence is visible where R is not installed
(e.g. Streamlit Community Cloud). The file is only ever written by this script, never by hand.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

from r_bridge.bridge import check_r
from r_bridge.suite import JML_REFERENCE, MML_REFERENCE, reference_suite
from rasch.simulation import simulate

RECORD = Path(__file__).resolve().parents[1] / "docs" / "validation_record.json"

DATASETS = [
    ("A: N = 500, J = 20, complete", dict(n_persons=500, n_items=20, seed=2026)),
    ("B: N = 500, J = 20, 10 % missing", dict(n_persons=500, n_items=20, missing=0.10, seed=2027)),
    ("C: N = 300, J = 8, ability SD 2", dict(n_persons=300, n_items=8, theta_sd=2.0, seed=31)),
]


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    return v


def main():
    info = check_r()
    if not (info.get("available") and info.get("TAM") and info.get("eRm")):
        raise SystemExit(f"R/TAM/eRm unavailable ({info.get('reason')}); nothing recorded.")
    rows = []
    for name, spec in DATASETS:
        d = simulate(**spec)
        rows += [dict(dataset=name, **{k: clean(v) for k, v in r.items()})
                 for r in reference_suite(d.X, d.item_ids, d.n_persons)]
    record = dict(date=date.today().isoformat(), environment={k: info.get(k) for k in ("R", "TAM", "eRm")},
                  reference_settings=dict(JMLE=JML_REFERENCE, MML=MML_REFERENCE),
                  datasets=[dict(name=n, simulation=s) for n, s in DATASETS], rows=rows)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    n_pass = sum(r["status"] == "Pass" for r in rows)
    print(f"Wrote {RECORD} · {n_pass} of {len(rows)} checks pass")


if __name__ == "__main__":
    main()
