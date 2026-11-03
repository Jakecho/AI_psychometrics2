"""Run the irtQ endpoint baselines and save them as docs/mmle_validation_record.json.

Run from the repository root:  python mmle_record_validation.py

The cases are the ones in test_mmle_engine.test_installed_irtq_endpoint_baselines: 4 designs x 2PL/3PL x
normal/histogram population, plus an OEM fixed-bank run. The Validation tab of MMLE_EM.py shows this file as the
recorded local run, so the evidence is visible where R is not installed (e.g. Streamlit Community Cloud).
The file is only ever written by this script, never by hand.
"""
import json
from datetime import date
from pathlib import Path

from mmle_engine import DESIGNS, FitConfig, Priors, fit, jsonable, simulate
from mmle_reference import DEFAULT_TOLERANCES, check_irtq, compare_endpoints, fit_irtq

RECORD = Path(__file__).resolve().parent / "docs" / "mmle_validation_record.json"


def cases():
    for design in DESIGNS:
        for model in ("2PL", "3PL"):
            for density in ("Normal", "Histogram"):
                mg = design in (DESIGNS[1], DESIGNS[3])
                anchor = design in DESIGNS[2:]
                d = simulate(500, 12, [dict(name=f"G{g}", mean=g * .4, sd=1.) for g in range(2 if mg else 1)],
                             model=model, common=8, anchors=5 if anchor else 0)
                yield design, model, density, "MEM", d, FitConfig(design=design, model=model, density=density,
                                                                  priors=Priors(c=model == "3PL"))
    yield DESIGNS[2], "2PL", "Normal", "OEM", simulate(500, 12, anchors=5), FitConfig(design=DESIGNS[2], schedule="OEM")


def main():
    info = check_irtq()
    if not info.get("available"):
        raise SystemExit(f"irtQ unavailable ({info.get('reason')}); nothing recorded.")
    rows = []
    for design, model, density, schedule, d, cfg in cases():
        r = fit(d, cfg)
        report = compare_endpoints(d, r, fit_irtq(d, r))
        checks = report.get("executed", [])
        rows.append(dict(design=design, model=model, population=density,
                         schedule=schedule if cfg.anchored else "—", status=report["status"],
                         python_cycles=len(r.cycles), python_stop=r.stop_reason,
                         checks=[dict(check=c["check"], value=c["value"], criterion=c["criterion"], status=c["status"])
                                 for c in checks],
                         reason=report.get("reason", "")))
    record = dict(date=date.today().isoformat(), environment=dict(R=info.get("R"), irtQ=info.get("version")),
                  tolerances=DEFAULT_TOLERANCES, rows=rows)
    RECORD.write_text(json.dumps(jsonable(record), indent=2) + "\n")
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {RECORD} · {len(rows)} cases · {counts}")


if __name__ == "__main__":
    main()
