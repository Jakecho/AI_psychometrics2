"""Run the full reference comparison and save it as docs/validation_record.json.

Run from IRT_ability_scoring/:  python -m r_bridge.record_validation

Page 10 ("Validation" tab) shows this file as the recorded local run, so the evidence is visible where R is not
installed (e.g. Streamlit Community Cloud). The file is only ever written by this script, never by hand.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np

from r_bridge.bridge import check_r
from r_bridge.compare import compare_all
from scoring import ScoringSettings
from simulation import example_a, example_b, example_c, example_d, simulate

RECORD = Path(__file__).resolve().parents[1] / "docs" / "validation_record.json"

# The four worked examples plus four simulated mixed-format tests (fixed seeds, so the record is reproducible).
SIMULATED = [
    ("Rasch+PCM, 10% missing, N=200", dict(n_persons=200, seed=11, n_dich=10, dich_model="Rasch", n_poly=4,
                                           poly_model="PCM", missing_rate=0.1)),
    ("3PL+GRM, N=200", dict(n_persons=200, seed=3, n_dich=12, dich_model="3PL", n_poly=4, poly_model="GRM")),
    ("2PL+GPCM(5 cat), N=200", dict(n_persons=200, seed=8, n_dich=10, dich_model="2PL", n_poly=4,
                                    poly_model="GPCM", n_cat=5)),
    ("4PL+RSM, N=200", dict(n_persons=200, seed=21, n_dich=10, dich_model="4PL", n_poly=4, poly_model="RSM")),
]


def datasets():
    out = [(fn().name, fn(), None) for fn in (example_a, example_b, example_c, example_d)]
    out += [(name, simulate(**spec), spec) for name, spec in SIMULATED]
    return out


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def main():
    info = check_r()
    if not info.get("available"):
        raise SystemExit(f"R is unavailable ({info.get('reason')}); nothing recorded.")
    settings = ScoringSettings()
    rows, used = [], []
    for name, d, spec in datasets():
        _, r = compare_all(d.items, d.Y, settings)
        rows += [dict(dataset=name, **{k: clean(v) for k, v in row.items()}) for row in r]
        used.append(dict(name=name, persons=int(d.Y.shape[0]), items=int(d.Y.shape[1]), simulation=spec))
    record = dict(date=date.today().isoformat(),
                  environment={k: info.get(k) for k in ("R", "irtQ", "mirt", "TAM")},
                  settings=settings.as_dict(), datasets=used, rows=rows)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    counts = {}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    print(f"Wrote {RECORD} · {len(rows)} comparisons · {counts}")


if __name__ == "__main__":
    main()
