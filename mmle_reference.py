"""irtQ endpoint adapter and honest, configuration-aware validation reports."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from mmle_engine import (jsonable, estep, probabilities, moments, invariant_report,
                         signature, fit, recovery, stationarity_step)

# Endpoint tolerances, checked against an executed baseline (2026-11-01): Python vs installed irtQ 1.1.0
# on all 16 design x model x density configurations (N=500 per group, 12 items, default settings).
# Largest observed free-coefficient differences: a 0.023, b 0.005, c 0.001; response curves 0.002.
DEFAULT_TOLERANCES = dict(a_atol=.02,a_rtol=.01,b_atol=.02,b_rtol=.01,c_atol=.005,c_rtol=.01,
                          curve=.01,moment=.02,density_tv=.02,loglik_per_person=1e-4)
BASELINE_NOTE = ("Tolerances checked against an executed baseline: Python vs installed irtQ 1.1.0 on all 16 "
                 "design/model/density configurations (N=500 per group, 12 items). Largest coefficient differences "
                 "there: a 0.023, b 0.005, c 0.001; curves 0.002. Pass means every check is within these "
                 "tolerances and the irtQ endpoint had stopped moving. It is endpoint agreement with one reference "
                 "package, not proof of correctness, global optimality or instrument validity.")


def r_environment():
    """Environment for Rscript: put the project's private R library (where irtQ is installed) first."""
    env=os.environ.copy()
    library=env.get("MMLE_R_LIBRARY") or str(Path.home()/".cache/ai_psychometrics2/R-library")
    if Path(library).is_dir():
        env["R_LIBS"]=library+(os.pathsep+env["R_LIBS"] if env.get("R_LIBS") else "")
    return env


def check_irtq():
    executable=shutil.which("Rscript")
    if not executable:
        return dict(available=False,status="Unavailable",reason="Rscript not found")
    code='cat(jsonlite::toJSON(list(R=R.version.string,available=requireNamespace("irtQ",quietly=TRUE),version=if(requireNamespace("irtQ",quietly=TRUE))as.character(packageVersion("irtQ"))else NA_character_),auto_unbox=TRUE))'
    try:
        p=subprocess.run([executable,"-e",code],capture_output=True,text=True,timeout=20,env=r_environment())
        if p.returncode:
            return dict(available=False,status="Unavailable",reason=p.stderr.strip() or "R/jsonlite probe failed")
        info=json.loads(p.stdout)
        info.update(status="Not run" if info["available"] else "Unavailable")
        if info["available"]:
            info["reason"]=f"Installed irtQ {info['version']} available ({info['R']})"
            if version_tuple(info["version"])<(1,3,0):
                info["reason"]+=(". This is older than the reviewed irtQ 1.4.0: releases before 1.3.0 test the signed "
                                 "maximum change, so they can stop while a coefficient is still decreasing. Each irtQ "
                                 "endpoint is therefore checked for stationarity before its agreement is interpreted.")
        else:
            info["reason"]=f"irtQ is not installed ({info['R']}); irtQ 1.1.0 is the last release for R 4.4, later releases require R >= 4.5"
        return info
    except (OSError,ValueError,subprocess.TimeoutExpired) as e:
        return dict(available=False,status="Unavailable",reason=str(e))


def version_tuple(version):
    try:
        return tuple(int(part) for part in str(version).split(".")[:3])
    except ValueError:
        return (0,)


def endpoint_stationarity(data,result,params,weights):
    """Largest item / density change produced by one EM cycle started at an endpoint."""
    step=stationarity_step(data,result.config,params,weights,result.nodes)
    return step.item_change,step.weight_change


# An endpoint whose next EM cycle moves a free coefficient by more than this many times
# the item tolerance was still moving when the reference stopped.
STATIONARITY_FACTOR = 10.
# Checks whose values depend on the population masses (judged only once the density has settled).
DENSITY_DEPENDENT = ("Group mean / SD","Population total variation","Common-grid log likelihood / person")


def configuration_differences(data,result):
    c=result.config
    differences=[]
    if c.positive_a and not c.priors.a:
        differences.append("Python positive-a bound differs from unbounded irtQ a domain")
    if c.strict_density_stop:
        differences.append("Python additionally requires density stability")
    if data.generation.get("D",c.D)!=c.D:
        differences.append("Generating D differs from fitting D; truth recovery needs separate alignment")
    differences.append("SciPy L-BFGS-B versus R nlminb numerical M-step; tighter R optimizer controls recorded")
    return differences


def fit_irtq(data,result,timeout=600):
    info=check_irtq()
    if not info["available"]:
        return dict(status="Unavailable",reason=info["reason"],environment=info)
    cfg=result.config
    with tempfile.TemporaryDirectory(prefix="mmle_irtq_") as directory:
        folder=Path(directory)
        pd.DataFrame(data.responses,columns=data.item_ids).to_csv(folder/"responses.csv",index=False)
        pd.DataFrame(data.administered.astype(int),columns=data.item_ids).to_csv(folder/"admin.csv",index=False)
        # 1/0, not True/False: R's read.csv keeps "True"/"False" as text, which silently unfixed every anchor.
        pd.DataFrame(dict(id=data.item_ids,fixed=data.fixed.astype(int))).to_csv(folder/"items.csv",index=False)
        pd.DataFrame(result.initial_params,columns=["a","b","c"]).to_csv(folder/"starts.csv",index=False)
        pd.DataFrame(dict(theta=result.nodes,weight=result.initial_weights[0])).to_csv(folder/"initial_weights.csv",index=False)
        request=dict(config=asdict(cfg),group=data.group,group_names=data.group_names,anchored=cfg.anchored)
        (folder/"request.json").write_text(json.dumps(jsonable(request)))
        try:
            p=subprocess.run([shutil.which("Rscript"),str(Path(__file__).with_name("mmle_irtq.R")),str(folder/"request.json")],
                             capture_output=True,text=True,timeout=timeout,env=r_environment())
            if p.returncode:
                return dict(status="Review",reason="irtQ execution failed",stderr=p.stderr[-10000:],stdout=p.stdout[-2000:])
            reference=pd.read_csv(folder/"reference_items.csv")
            params=reference[["a","b","c"]].to_numpy()
            if reference.id.tolist()!=data.item_ids or not np.isfinite(params).all():
                raise ValueError("Reference endpoint IDs or coefficients are invalid")
            weights=[]
            for g in range(len(data.group_names)):
                frame=pd.read_csv(folder/f"reference_weights_{g+1}.csv")
                if not np.allclose(frame.iloc[:,0],result.nodes,atol=1e-12):
                    raise ValueError("Reference quadrature nodes differ; no matched-grid comparison")
                weights.append(frame.iloc[:,1].to_numpy())
            return dict(status="Executed",params=params,weights=np.array(weights),
                        reference=json.loads((folder/"reference_report.json").read_text()),stdout=p.stdout[-2000:],stderr=p.stderr[-2000:])
        except subprocess.TimeoutExpired:
            return dict(status="Review",reason=f"irtQ exceeded {timeout} seconds; comparison not completed")
        except (OSError,ValueError,KeyError) as e:
            return dict(status="Review",reason=str(e))


def compare_endpoints(data,result,ref,tolerances=None):
    tol=dict(DEFAULT_TOLERANCES,**(tolerances or {}))
    report=dict(status=ref.get("status","Not run"),run_id=result.run_id,data_id=result.data_id,
                settings=jsonable(result.config),versions=result.versions,seed=data.generation.get("seed"),
                tolerances=tol,tolerance_rationale=BASELINE_NOTE,differences=configuration_differences(data,result),
                executed=[],pending=["Grid sensitivity","Start sensitivity","Repeated-simulation recovery"],reference=jsonable(ref))
    if ref.get("status")!="Executed":
        report["reason"]=ref.get("reason","Reference check not run")
        return report
    rp,rw=np.asarray(ref["params"]),np.asarray(ref["weights"])
    rows=[]
    free=~data.fixed
    for k,name in enumerate(("a","b","c")):
        if k==2 and result.config.model=="2PL": continue
        delta=result.params[:,k]-rp[:,k]
        allowed=tol[name+"_atol"]+tol[name+"_rtol"]*np.abs(rp[:,k])
        for j,item in enumerate(data.item_ids):
            rows.append(dict(item=item,parameter=name,python=float(result.params[j,k]),irtQ=float(rp[j,k]),
                             difference=float(delta[j]),allowance=float(allowed[j]),fixed=bool(data.fixed[j]),
                             status="Pass" if abs(delta[j])<=allowed[j] else "Fail"))
    curve=float(np.max(np.abs(probabilities(result.nodes,result.params,result.config.D)-probabilities(result.nodes,rp,result.config.D))))
    mo=float(np.max(np.abs(moments(result.nodes,result.weights)-moments(result.nodes,rw))))
    tv=float(np.max(.5*np.sum(np.abs(result.weights-rw),axis=1)))
    pyll=float(estep(data,result.params,result.weights,result.nodes,result.config.D)[2].sum())
    rll=float(estep(data,rp,rw,result.nodes,result.config.D)[2].sum())
    llerror=abs(pyll-rll)/len(data.responses)
    checks=[dict(check="Parameter endpoints",value=max([abs(r['difference']) for r in rows],default=0),criterion="Per-row absolute + relative allowance",status="Pass" if all(r['status']=="Pass" for r in rows) else "Fail"),
            dict(check="Response curves (max absolute)",value=curve,criterion=f"≤ {tol["curve"]:g}",status="Pass" if curve<=tol["curve"] else "Fail"),
            dict(check="Group mean / SD (max absolute)",value=mo,criterion=f"≤ {tol["moment"]:g}",status="Pass" if mo<=tol["moment"] else "Fail"),
            dict(check="Population total variation (max group)",value=tv,criterion=f"≤ {tol["density_tv"]:g}",status="Pass" if tv<=tol["density_tv"] else "Fail"),
            dict(check="Common-grid log likelihood / person",value=llerror,criterion=f"≤ {tol["loglik_per_person"]:g}",status="Pass" if llerror<=tol["loglik_per_person"] else "Fail")]
    oem=result.config.anchored and result.config.schedule=="OEM"
    still_moving=density_unsettled=False
    if free.any() and not oem:
        # Guard against early stopping in the reference (irtQ < 1.3.0 tests the signed maximum change).
        limit=STATIONARITY_FACTOR*result.config.tolerance
        r_move,r_density=endpoint_stationarity(data,result,rp,rw)
        py_move,_=endpoint_stationarity(data,result,result.params,result.weights)
        still_moving=r_move>limit
        checks.append(dict(check="irtQ endpoint stationarity (one EM cycle from irtQ endpoint)",value=r_move,
                           criterion=f"≤ {limit:g}",status="Review" if still_moving else "Pass"))
        py_density=endpoint_stationarity(data,result,result.params,result.weights)[1]
        report.update(stationarity=dict(irtq_item_change=r_move,irtq_density_change=r_density,
                                        python_item_change=py_move,python_density_change=py_density,criterion=limit))
        # The usual (irtQ) stopping rule checks item changes only. Histogram masses converge far more slowly,
        # so two correct programs can stop at different points while the density is still moving. A density
        # disagreement between two unconverged densities cannot be judged: mark it Review, keep coefficients strict.
        # Judged on Python's own endpoint only (self-consistent), so it can never hide a Python error.
        density_limit=result.config.density_tolerance
        if result.config.density=="Histogram" and py_density>density_limit:
            density_unsettled=True
            note=(f"Density not converged (one more EM cycle moves a node mass by {py_density:.1e} from the "
                  f"Python endpoint and {r_density:.1e} from the irtQ endpoint; limit {density_limit:.0e})")
            for c in checks:
                if c["status"]=="Fail" and c["check"].split(" (")[0] in DENSITY_DEPENDENT:
                    c["status"]="Review"; c["note"]=note
            report["density_note"]=note
    report.update(endpoint_rows=rows,executed=checks,python_loglik=pyll,irtq_loglik=rll,
                  free_summary=[dict(parameter=n,bias=float(np.mean(result.params[free,k]-rp[free,k])),
                                     RMSE=float(np.sqrt(np.mean((result.params[free,k]-rp[free,k])**2))))
                                for k,n in enumerate(("a","b","c")) if free.any() and (k<2 or result.config.model=="3PL")])
    # Agreement alone cannot certify a preset before the requested baseline exercise.
    statuses=[c["status"] for c in checks]
    report["status"]="Fail" if "Fail" in statuses else "Pass" if all(x=="Pass" for x in statuses) else "Review"
    report["reason"]="Endpoint differences exceed the tolerances" if report["status"]=="Fail" else BASELINE_NOTE
    if (result.config.positive_a and not result.config.priors.a) or result.stop_reason in ("Cycle limit reached","Paused after recorded cycle"):
        report["status"]="Review"
        report["reason"]="Feasible-domain mismatch or incomplete convergence; inspect endpoint evidence"
    if density_unsettled and report["status"]!="Fail":
        report["status"]="Review"
        report["reason"]=(report["density_note"]+". Item coefficients are compared strictly; population-dependent checks "
                          "are Review until the density settles. Refit with 'Also require density stability' or a smaller "
                          "item tolerance, then validate again.")
    if still_moving and report["status"]=="Fail":
        # Never downgrade a Fail: the guard uses Python's EM, so it cannot excuse a disagreement.
        report["reason"]+=(f". Note: one more EM cycle from the irtQ endpoint moves a free coefficient by "
                           f"{report['stationarity']['irtq_item_change']:.2g}, so irtQ may also have stopped early.")
    elif still_moving:
        report["status"]="Review"
        report["reason"]=("The irtQ endpoint was still moving when irtQ stopped (one more EM cycle changes a free "
                          f"coefficient by {report['stationarity']['irtq_item_change']:.2g}). Endpoint differences may "
                          "reflect the reference stopping early, not a Python error.")
    return report


def sensitivity_checks(data,result):
    cfg=result.config
    # Same saved data; two actual additional fits, no generation.
    start_cfg=replace(cfg,start_a=cfg.start_a*.85,start_b=cfg.start_b+.15)
    grid_cfg=replace(cfg,nodes=2*cfg.nodes-1,initial_weights=None)
    if cfg.initial_weights is not None:
        newx=np.linspace(-cfg.extent,cfg.extent,grid_cfg.nodes)
        w=np.interp(newx,result.nodes,cfg.initial_weights)
        grid_cfg=replace(grid_cfg,initial_weights=(w/w.sum()).tolist())
    output=[]
    for label,c in (("Changed start",start_cfg),("Refined grid",grid_cfg)):
        alt=fit(data,c)
        output.append(dict(check=label,status="Review",max_parameter_change=float(np.max(np.abs(alt.params-result.params))),
                           mean_SD_change=float(np.max(np.abs(moments(alt.nodes,alt.weights)-moments(result.nodes,result.weights)))),
                           stop_reason=alt.stop_reason,run_id=alt.run_id))
    return output
