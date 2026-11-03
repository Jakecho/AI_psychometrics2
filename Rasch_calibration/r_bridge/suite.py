"""The full Python-vs-R comparison for one dataset (used by page 8 and by record_validation)."""
from rasch.cmle import cmle
from rasch.jmle import JMLESettings, jmle
from rasch.mml_em import MMLSettings, mml_em
from rasch.scoring import score_persons
from rasch.validation import compare, max_abs

from .bridge import run_reference

MML_REFERENCE = dict(n_nodes=21, extent=6.0, estimate_variance=True, fixed_sd=1.0, tol=1e-4, max_iter=1000)
JML_REFERENCE = dict(adj=.3, bias=True, constraint="cases", tol=1e-4, max_iter=1000)


def reference_suite(X, item_ids, n_persons):
    """JMLE vs tam.jml, CMLE vs eRm::RM, MML-EM vs tam.mml, WLE vs tam.wle. Returns a list of check rows."""
    rows = []
    j = jmle(X, JMLESettings())
    rj = run_reference("JMLE", X, JML_REFERENCE, item_ids)
    rows += [dict(method="JMLE vs TAM::tam.jml", **r) for r in compare("JMLE", dict(b=j.b, se_b=j.se_b, theta=j.theta), rj)]
    try:
        c = cmle(X)
        rc = run_reference("CMLE", X, {}, item_ids)
        rows += [dict(method="CMLE vs eRm::RM", **r)
                 for r in compare("CMLE", dict(b=c.b, se_b=c.se_b, cond_loglik=c.cond_loglik), rc, n_persons)]
    except ValueError as e:
        rows.append(dict(method="CMLE vs eRm::RM", check="Python CMLE", status=f"Not run: {e}"))
    m = mml_em(X, MMLSettings())
    rm = run_reference("MML", X, MML_REFERENCE, item_ids)
    rows += [dict(method="MML-EM vs TAM::tam.mml", **r)
             for r in compare("MML", dict(b=m.b, se_b=m.se_b_complete, se_label=" (complete-data)", variance=m.variance,
                                          loglik=m.loglik_tam), rm, n_persons)]
    if rm.get("status") == "Executed":
        w, wse = score_persons(X, rm["b"], "WLE")
        for label, a, b2 in (("WLE abilities", w, rm["wle"]), ("WLE standard errors", wse, rm["wle_se"])):
            diff = max_abs(a, b2)
            rows.append(dict(method="WLE vs TAM::tam.wle", check=label, max_abs_difference=diff, tolerance=1e-3,
                             status="Pass" if diff <= 1e-3 else "Fail"))
    return rows
