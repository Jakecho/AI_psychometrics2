"""MMLE–EM Learning Lab — run with: python3 -m streamlit run MMLE_EM.py.

Reviewed design: MMLE EM educational app design.md; layout: mmle-em-layout.svg.
Python calibration is independent of R; irtQ validates endpoints when available.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, lognorm, beta as beta_dist
from mmle_engine import (DESIGNS, VERSION, Dataset, FitConfig, Priors, simulate, fit, check_design,
                         signature, jsonable, probabilities, normal_weights, moments, recovery, invariant_report)
from mmle_animation import animation_payload, playback_html
from mmle_reference import (check_irtq, fit_irtq, compare_endpoints, DEFAULT_TOLERANCES, BASELINE_NOTE, sensitivity_checks)
from mmle_data import load_csv_dataset,export_experiment as export_zip
from mmle_lessons import LESSONS, SCORES, NOTATION

st.set_page_config(page_title="MMLE–EM Learning Lab",page_icon="📐",layout="wide")
st.markdown("""<style>.block-container{padding-top:2rem;max-width:1550px}h1,h2,h3{color:#17273f}div[data-testid='stMetric']{border:1px solid #dce3ef;border-radius:8px;padding:12px}div[data-testid='stSidebar']{background:#edf1f8}.stTabs [data-baseweb='tab-list']{gap:24px}</style>""",unsafe_allow_html=True)


def line_chart(x,series,title="",xlabel="Ability θ",ylabel="Probability / mass"):
    fig=go.Figure()
    for name,values in series.items():
        fig.add_trace(go.Scatter(x=x,y=values,name=name,mode="lines"))
    fig.update_layout(title=title,height=300,margin=dict(l=15,r=15,t=45,b=20),template="plotly_white",
                      xaxis_title=xlabel,yaxis_title=ylabel,legend=dict(orientation="h",y=1.12),font=dict(size=12))
    return fig


def item_frame(data,params):
    f=pd.DataFrame(params,columns=["a","b","c"])
    f.insert(0,"Item",data.item_ids)
    f["Bank fixed"]=data.fixed
    return f


def history_frame(result):
    return pd.DataFrame([dict(cycle=c.index,raw_loglik=c.loglik,penalized_objective=c.penalized,
                              item_change=c.item_change,weight_change=c.weight_change,moment_change=c.moment_change,
                              optimizer_OK=all(s["success"] for s in c.optimizer)) for c in result.cycles])


def matching_run(data,cfg,result):
    return result is not None and result.data_id==data.identity and result.constraints_id==signature(data.bank) and signature(result.config)==signature(cfg)


def execute_fit(data,cfg,one=False,resume=False):
    previous=st.session_state.get("run")
    if not (resume and matching_run(data,cfg,previous)):
        previous=None
    status=st.empty()
    def progress(c):
        if c.index==1 or c.index%10==0:
            status.info(f"Calibration cycle {c.index} · item change {c.item_change:.3g} · density change {c.weight_change:.3g}")
    try:
        with st.spinner("Fitting saved responses; recording real E/M states…"):
            result=fit(data,cfg,previous=previous,cycles=1 if one else None,callback=progress)
        st.session_state.run=result
        st.session_state.run_data=deepcopy(data)
        st.session_state.validation=None
        st.session_state.sensitivity=None
        st.session_state.invariants=None
        st.session_state.replicated=None
        status.success(f"{len(result.cycles)} recorded cycles · {result.stop_reason}")
    except (ValueError,FloatingPointError) as e:
        status.error(str(e))


def validation_record(result):
    primary=st.session_state.get("validation")
    record=deepcopy(primary) if primary and primary.get("run_id")==result.run_id else dict(status="Not run",run_id=result.run_id,data_id=result.data_id,settings=jsonable(result.config),versions=result.versions,executed=[],pending=["irtQ endpoint agreement"])
    pending=list(record.get("pending",[]))
    for key,title in [("invariants","Constraints / arithmetic / animation"),("sensitivity","Grid / start sensitivity"),("replicated","Repeated-simulation recovery")]:
        saved=st.session_state.get(key)
        if saved and saved[0]==result.run_id:
            record[key]=saved[1]
            pending=[p for p in pending if p not in (["Grid sensitivity","Start sensitivity"] if key=="sensitivity" else ["Repeated-simulation recovery"] if key=="replicated" else [])]
        else:
            if title not in pending: pending.append(title)
    record["pending"]=pending
    return record


@st.cache_data(ttl=60,show_spinner=False)
def reference_environment():
    return check_irtq()


if "dataset" not in st.session_state:
    st.session_state.dataset=simulate()
    st.session_state.experiments=[]
    st.session_state.run=None
    st.session_state.validation=None

st.title("MMLE–EM Learning Lab")
st.caption("Understand how responses, population weights and calibration constraints work together.")
with st.sidebar:
    st.header("Experiment settings")
    level=st.radio("Learning view",["Guided","Explore"],horizontal=True)
    design=st.selectbox("Calibration design",DESIGNS)
    model=st.selectbox("Fitted IRT model",["2PL","3PL"])
    density=st.selectbox("Population ability distribution",["Normal","Histogram"])
    D=st.number_input("Fitted logistic scale D",min_value=.1,max_value=3.,value=1.,step=.1)
    anchored=design in DESIGNS[2:]
    schedule=st.selectbox("Anchored schedule",["MEM","OEM"],disabled=not anchored,
                          help="MEM: anchors in first E-step, all items later. OEM: one item-fitting cycle, then return updated density.")
    with st.expander("Numerical settings",expanded=level=="Explore"):
        Q=st.number_input("Quadrature nodes",5,151,49,2)
        extent=st.number_input("Grid half-width",2.,12.,6.,.5)
        fixed_mean=st.number_input("Initial/reference mean",value=0.,step=.1)
        fixed_sd=st.number_input("Initial/reference SD",min_value=.1,value=1.,step=.1)
        maxcycles=st.number_input("Maximum cycles",1,2000,500,25)
        tolerance=st.number_input("Item change tolerance",min_value=1e-8,value=1e-4,format="%.6f")
        positive=st.checkbox("Constrain fitted a > 0 (app restriction)")
        strict=st.checkbox("Also require density stability (app rule)")
        dtol=st.number_input("Density change tolerance",min_value=1e-8,value=1e-5,format="%.6f",disabled=not strict)
        sa=st.number_input("Starting a",min_value=.01,value=1.,step=.1)
        sb=st.number_input("Starting b",value=0.,step=.1)
        sc=st.number_input("Starting c",min_value=.0001,max_value=.9999,value=.15,step=.01,disabled=model=="2PL")
        custom_initial=st.checkbox("Custom initial node masses")
        initial_text=st.text_area("Initial masses JSON (same grid, all groups)",value="[]",disabled=not custom_initial)
    with st.expander("Item parameter priors",expanded=False):
        st.caption("Scope: all free items. Bank anchors stay locked. Population masses and generating distributions are separate.")
        preset=st.selectbox("Prior preset",["Pure MMLE — all off","irtQ defaults","Custom"],key="prior_preset")
        def mark_prior_custom():
            st.session_state.prior_preset="Custom"
            st.session_state.last_prior_preset="Custom"
        pure_reset=st.button("Pure MMLE · reset all item priors off")
        irtq_reset=st.button("Reset to irtQ prior defaults")
        # Callback resets widget values before the coefficient widgets are constructed.
        if st.session_state.get("last_prior_preset")!=preset or pure_reset or irtq_reset:
            for key,value in dict(prior_a=False,prior_b=False,prior_c=(irtq_reset or (preset=="irtQ defaults" and not pure_reset)) and model=="3PL",amu=0.,asd=.5,bmu=0.,bsd=1.,calpha=5.,cbeta=16.).items():
                st.session_state[key]=value
            st.session_state.last_prior_preset=preset
        if model=="2PL": st.session_state.prior_c=False
        if st.session_state.get("last_prior_model")!=model and model=="3PL" and preset=="irtQ defaults": st.session_state.prior_c=True
        st.session_state.last_prior_model=model
        pa=st.checkbox("Enable a prior",key="prior_a",on_change=mark_prior_custom)
        st.selectbox("a family / scale",["Lognormal on D × a"])
        amu=st.number_input("a log mean",key="amu",on_change=mark_prior_custom)
        asd=st.number_input("a log SD",min_value=.01,key="asd",on_change=mark_prior_custom)
        pb=st.checkbox("Enable b prior",key="prior_b",on_change=mark_prior_custom)
        st.selectbox("b family",["Normal on b"])
        bmu=st.number_input("b prior mean",key="bmu",on_change=mark_prior_custom)
        bsd=st.number_input("b prior SD",min_value=.01,key="bsd",on_change=mark_prior_custom)
        pc=st.checkbox("Enable c prior (3PL)",key="prior_c",disabled=model=="2PL",on_change=mark_prior_custom) and model=="3PL"
        st.selectbox("c family",["Beta on c"],disabled=model=="2PL")
        calpha=st.number_input("c shape α",min_value=.05,key="calpha",disabled=model=="2PL",on_change=mark_prior_custom)
        cbeta=st.number_input("c shape β",min_value=.05,key="cbeta",disabled=model=="2PL",on_change=mark_prior_custom)
        prior_dist=[("D × a",lognorm(s=asd,scale=np.exp(amu)),pa,D),("b",norm(bmu,bsd),pb,1.),("c",beta_dist(calpha,cbeta),pc,1.)]
        for name,dist,on,scale in prior_dist:
            if on:
                qs=dist.ppf([.025,.5,.975]); xx=np.linspace(max(1e-8,qs[0]) if name=="c" else qs[0],min(1-1e-8,qs[-1]) if name=="c" else qs[-1],150)
                st.plotly_chart(line_chart(xx,{name:dist.pdf(xx)},f"{name} prior","Prior scale","Density"),width="stretch",key=f"prior_plot_{name}")
                st.caption(f"{name}: mean {dist.mean():.3g}; SD {dist.std():.3g}; 2.5/50/97.5% {np.round(qs,3)}")
                if name=="D × a": st.caption(f"Actual a: mean {dist.mean()/scale:.3g}; SD {dist.std()/scale:.3g}; quantiles {np.round(qs/scale,3)}")
        st.caption("Default bounds: a/b unbounded; free 3PL c ∈ [1e−8,1−1e−8]; 2PL c=0. Lognormal support restricts a to positive values.")
    with st.expander("Response generation",expanded=False):
        st.caption("Changes below take effect only when Generate / Regenerate is pressed.")
        source=st.radio("Data source",["Simulation","Upload"])
        generation_model=st.selectbox("Generating model",["2PL","3PL"])
        G=st.number_input("Generating groups/forms",1,5,1 if design in (DESIGNS[0],DESIGNS[2]) else 2)
        n=st.number_input("People per group",30,10000,1000,50)
        Jform=st.number_input("Items per form",4,100,20)
        common=st.number_input("Common items across forms",0,int(Jform),min(8,int(Jform)))
        bank_count=st.number_input("Bank anchors (first global items)",0,int(Jform),min(5,int(Jform)) if anchored else 0)
        generation_D=st.number_input("Generating D",min_value=.1,max_value=3.,value=1.,step=.1)
        seed=st.number_input("Generation seed",0,2**31-1,2026)
        missing=st.slider("Response missingness",0.,.5,0.,.01)
        a_lo=st.number_input("True a minimum",min_value=.05,value=.5)
        a_hi=st.number_input("True a maximum",min_value=.05,value=2.)
        b_lo=st.number_input("True b minimum",value=-2.)
        b_hi=st.number_input("True b maximum",value=2.)
        c_lo=st.number_input("True c minimum",0.,.95,.1,disabled=generation_model=="2PL")
        c_hi=st.number_input("True c maximum",0.,.95,.25,disabled=generation_model=="2PL")
        drift=st.number_input("Anchor drift lesson: supplied first-bank b offset",value=0.,step=.1)
    generate_sidebar=st.button("Generate / Regenerate responses",width="stretch")
    run_sidebar=st.button("Run calibration",type="primary",width="stretch")
    st.caption("Generate changes the dataset. Fitting and playback reuse saved responses.")

priors=Priors(pa,pb,pc,amu,asd,bmu,bsd,calpha,cbeta)
try:
    initial=json.loads(initial_text) if custom_initial else None
    cfg=FitConfig(design,model,density,D,int(Q),extent,fixed_mean,fixed_sd,int(maxcycles),tolerance,dtol,strict,schedule,
                  positive,sa,sb,sc,priors,initial)
    cfg.validate()
except (ValueError,TypeError,json.JSONDecodeError) as e:
    st.error(f"Invalid pending fitting settings: {e}")
    st.stop()

st.caption(f"Current saved dataset: {st.session_state.dataset.identity} · {len(st.session_state.dataset.responses):,} people · {len(st.session_state.dataset.item_ids)} global item IDs · {len(st.session_state.dataset.group_names)} groups")
if not st.session_state.get("validation"):
    st.info("Not validated · Run the Python calibration and execute the primary irtQ check in Validation. Both models remain experimental until reference checks are completed.")
design_tab,walk_tab,results_tab,validation_tab,compare_tab,notes_tab=st.tabs(["Design","EM walkthrough","Results","Validation","Compare","Method notes"])
with design_tab:
    st.subheader("1 · Define and inspect the experiment")
    st.write("Which items are estimated? What is shared across groups? What fixes the ability scale?")
    with st.expander("Generation design · group distributions, forms and item truth",expanded=source=="Simulation"):
        Jglobal=int(Jform) if G==1 else int(common+G*(Jform-common))
        spec_id=signature([G,Jform,common,generation_model,seed,a_lo,a_hi,b_lo,b_hi,c_lo,c_hi,bank_count,drift])
        default_groups=pd.DataFrame([dict(name=f"G{g+1}",mean=0. if g==0 else .5,sd=1.,shape="Normal",nodes="[-2, -1, 0, 1, 2]",weights="[0.1, 0.2, 0.4, 0.2, 0.1]") for g in range(int(G))])
        groups_editor=st.data_editor(default_groups,hide_index=True,key="gen_groups_"+str(G),width="stretch",
                                     column_config={"shape":st.column_config.SelectboxColumn(options=["Normal","Skewed","Bimodal","Custom"]),"sd":st.column_config.NumberColumn(min_value=.01)})
        st.caption("Skewed and bimodal presets are standardized to the declared population mean/SD. Custom nodes are literal; their normalized masses determine the exact population moments.")
        rng=np.random.default_rng(int(seed))
        truth=np.column_stack([rng.uniform(a_lo,a_hi,Jglobal) if a_hi>=a_lo else np.ones(Jglobal),
                               rng.uniform(b_lo,b_hi,Jglobal) if b_hi>=b_lo else np.zeros(Jglobal),
                               rng.uniform(c_lo,c_hi,Jglobal) if generation_model=="3PL" and c_hi>=c_lo else np.zeros(Jglobal)])
        template=pd.DataFrame(truth,columns=["true a","true b","true c"])
        template.insert(0,"Item",[f"I{j+1:02}" for j in range(Jglobal)])
        template["Bank fixed"]=[j<bank_count for j in range(Jglobal)]
        template["supplied a"]=truth[:,0];template["supplied b"]=truth[:,1];template["supplied c"]=truth[:,2]
        if bank_count: template.loc[0,"supplied b"]+=drift
        edited_truth=st.data_editor(template,key="truth_"+spec_id,hide_index=True,width="stretch",disabled=["Item"])
        st.caption("True item values generate the responses. Supplied bank values constrain fitting only when Bank fixed is checked. Edits remain pending until Generate.")
        amap=np.zeros((int(G),Jglobal),bool)
        for g in range(int(G)):
            if G==1: amap[g]=True
            else:
                amap[g,:int(common)]=True
                amap[g,int(common)+g*(int(Jform)-int(common)):int(common)+(g+1)*(int(Jform)-int(common))]=True
        form_editor=st.data_editor(pd.DataFrame(amap,index=[f"G{g+1}" for g in range(int(G))],columns=template.Item),key="forms_"+signature([G,Jform,common]),width="stretch")
        st.caption("Each row defines one group's form. Same global item ID shares one parameter vector. Arbitrary per-person administration can be supplied in upload mode.")
        generate_local=st.button("Generate responses from these tables",disabled=source!="Simulation")
    with st.expander("Upload data · response, administration and bank CSVs",expanded=source=="Upload"):
        st.write("Responses CSV: a `group` column plus global item-ID columns containing 0, 1 or blank. Administration CSV: the same item columns and row order, with 1=administered and 0=absent. A blank administered response is missing. Bank CSV: `id,fixed,a,b,c`; free rows may be omitted.")
        up_y=st.file_uploader("Responses CSV",type="csv",key="responses_upload")
        up_a=st.file_uploader("Administration CSV (recommended for multiple forms)",type="csv",key="admin_upload")
        up_b=st.file_uploader("Optional bank/item metadata CSV",type="csv",key="bank_upload")
        st.caption("Without an administration CSV all items are treated as administered, including missing responses. Optional metadata model/cats must match the selected homogeneous 2PL/3PL model and two categories.")
        load_upload=st.button("Load uploaded responses",disabled=source!="Upload" or up_y is None)
    if (generate_sidebar or generate_local) and source=="Simulation":
        try:
            specs=[]
            for row in groups_editor.to_dict("records"):
                if row["shape"]=="Custom": row.update(nodes=json.loads(row["nodes"]),weights=json.loads(row["weights"]))
                else: row.pop("nodes",None);row.pop("weights",None)
                specs.append(row)
            bank=np.full((Jglobal,3),np.nan)
            fixed=edited_truth["Bank fixed"].to_numpy(bool)
            bank[fixed]=edited_truth.loc[fixed,["supplied a","supplied b","supplied c"]]
            new=simulate(int(n),int(Jform),specs,generation_model,generation_D,int(seed),int(common),int(bank_count),missing,
                         (a_lo,a_hi),(b_lo,b_hi),(c_lo,c_hi),edited_truth[["true a","true b","true c"]].to_numpy(),form_editor.to_numpy(),bank,drift)
            st.session_state.dataset=new;st.session_state.run=None;st.session_state.validation=None
            st.success(f"Saved generated responses · {new.identity}. Fitting changes now reuse this dataset.")
        except (ValueError,TypeError,KeyError) as e: st.error(f"Generation not applied: {e}")
    if load_upload or (generate_sidebar and source=="Upload"):
        try:
            if up_y is None: raise ValueError("Choose a responses CSV first.")
            new=load_csv_dataset(up_y.getvalue(),up_a.getvalue() if up_a else None,up_b.getvalue() if up_b else None,model)
            st.session_state.dataset=new;st.session_state.run=None;st.session_state.validation=None
            st.success(f"Loaded {len(new.responses)} rows; preserved administration and missingness separately.")
        except (ValueError,KeyError,TypeError) as e: st.error(f"Upload not applied: {e}")
    data=st.session_state.dataset
    st.markdown("**Actual saved administration and item constraints**")
    present=np.array([data.administered[data.group==g].any(axis=0) for g in range(len(data.group_names))])
    labels=np.where(~present,"Not administered",np.where(data.fixed[None,:],"Bank/fixed",np.where(present.sum(axis=0)[None,:]>1,"Common/shared","Unique/free")))
    st.dataframe(pd.DataFrame(labels,index=data.group_names,columns=data.item_ids),width="stretch")
    constraint=pd.DataFrame(dict(id=data.item_ids,model=[cfg.model]*len(data.item_ids),categories=[2]*len(data.item_ids),
                                 participating_groups=[", ".join(data.group_names[g] for g in range(len(data.group_names)) if present[g,j]) for j in range(len(data.item_ids))],
                                 fixed=data.fixed,a=data.bank[:,0],b=data.bank[:,1],c=data.bank[:,2]))
    bank_editor=st.data_editor(constraint,hide_index=True,width="stretch",key="bank_constraints_"+signature([data.identity,data.bank,cfg.model]),disabled=["id","model","categories","participating_groups"])
    if st.button("Apply supplied bank constraints"):
        try:
            dd=deepcopy(data);dd.bank[:]=np.nan
            lock=bank_editor.fixed.to_numpy(bool)
            dd.bank[lock]=bank_editor.loc[lock,["a","b","c"]].to_numpy(float)
            dd.validate();st.session_state.dataset=dd;data=dd
            st.success("Bank constraints applied. Responses and truth preserved; rerun calibration to use them.")
        except ValueError as e: st.error(str(e))
    group_constraints=pd.DataFrame([dict(group=name,people=int(np.sum(data.group==g)),
                                         initial_distribution="Custom node masses" if cfg.initial_weights is not None else "Normal",
                                         fitted_distribution=cfg.density,
                                         initial_mean=fixed_mean,initial_SD=fixed_sd,
                                         distribution_rule=("Update shape; rescale reference moments" if not anchored and g==0 else "Update weights freely") if cfg.density=="Histogram" else ("Retain initial masses" if not anchored and g==0 else "Update normal moments"),
                                         moments="Free on bank scale" if anchored else ("Reference fixed" if g==0 else "Free on shared scale"),
                                         scale_source="Supplied bank" if anchored else "Reference group "+data.group_names[0]) for g,name in enumerate(data.group_names)])
    if cfg.initial_weights is not None:
        actual_initial=moments(np.linspace(-cfg.extent,cfg.extent,cfg.nodes),np.asarray(cfg.initial_weights)/np.sum(cfg.initial_weights))
        group_constraints["initial_mean"]=actual_initial[0];group_constraints["initial_SD"]=actual_initial[1]
    st.dataframe(group_constraints,hide_index=True,width="stretch")
    try: st.info(check_design(data,cfg))
    except ValueError as e: st.warning(f"Design needs attention before Run: {e}")
    constant_items=[data.item_ids[j] for j in range(len(data.item_ids)) if not data.fixed[j] and len(np.unique(data.responses[data.observed[:,j],j]))<2]
    if constant_items: st.warning("Free items with constant observed responses may have unstable or unbounded likelihood estimates: "+", ".join(constant_items))
    if data.generation.get("D",D)!=D: st.warning("Generating and fitted D differ. Known-truth recovery is unavailable for this mismatch; the saved responses are unchanged.")
    lcol,rcol=st.columns(2)
    with lcol:
        item=st.selectbox("Preview item",data.item_ids)
        if data.truth is not None:
            xx=np.linspace(-6,6,151);j=data.item_ids.index(item)
            series={"True curve":probabilities(xx,data.truth[j],data.generation["D"])[0]}
            if data.fixed[j]: series["Supplied bank curve"]=probabilities(xx,data.bank[j],D)[0]
            st.plotly_chart(line_chart(xx,series,"Known truth and supplied bank"),width="stretch")
        st.dataframe(pd.DataFrame(dict(item=data.item_ids,observed=data.observed.sum(axis=0),administered=data.administered.sum(axis=0),proportion_correct=np.nanmean(data.responses,axis=0))),hide_index=True,width="stretch")
    with rcol:
        if data.theta is not None:
            fig=go.Figure()
            for g,name in enumerate(data.group_names): fig.add_trace(go.Histogram(x=data.theta[data.group==g],name=name,histnorm="probability",opacity=.6))
            fig.update_layout(title="Realized true abilities (simulation)",barmode="overlay",height=300,template="plotly_white")
            st.plotly_chart(fig,width="stretch")
        codes=np.where(~data.administered,-2,np.where(~data.observed,-1,data.responses))
        fig=go.Figure(go.Heatmap(z=codes[:100],x=data.item_ids,zmin=-2,zmax=1,colorscale=[[0,"#e2e8f0"],[.333,"#e3ad57"],[.667,"#fafcff"],[1,"#2d5eb2"]],colorbar=dict(tickvals=[-2,-1,0,1],ticktext=["Absent","Missing","0","1"])))
        fig.update_layout(title="Responses / administration · first 100 people",height=300,template="plotly_white")
        st.plotly_chart(fig,width="stretch")
    st.download_button("Export saved dataset, maps, truth and seed",export_zip(data),"mmle_dataset.zip","application/zip")

# Use the current saved data, including any just-applied bank constraints.
data=st.session_state.dataset
if run_sidebar: execute_fit(data,cfg)
run=st.session_state.get("run")
if run and not matching_run(data,cfg,run):
    st.warning("Pending fitting settings or constraints differ from the saved run. Charts retain the saved run until Run is pressed.")
run_data=st.session_state.get("run_data",data)

with walk_tab:
    st.subheader("2 · Watch one complete EM cycle")
    st.caption("Input state ψ(t) → posterior τ(t) → updated state ψ(t+1). Playback reuses recorded states.")
    one_col,to_col=st.columns(2)
    one=one_col.button("Run one cycle")
    to_rule=to_col.button("Run to stopping rule")
    if one or to_rule:
        execute_fit(data,cfg,one=one,resume=True)
        run=st.session_state.get("run");run_data=st.session_state.get("run_data",data)
    if run and run.cycles:
        gc,jc,pc=st.columns(3)
        selected_group=gc.selectbox("Playback group",range(len(run_data.group_names)),format_func=lambda g:run_data.group_names[g])
        selected_item=jc.selectbox("Playback item",range(len(run_data.item_ids)),format_func=lambda j:run_data.item_ids[j]+(" · 🔒 bank" if run_data.fixed[j] else ""))
        people=np.flatnonzero(run_data.group==selected_group)
        person=pc.selectbox("Person / response pattern",people,format_func=lambda i:f"Person {i+1} · "+"".join("—" if not run_data.administered[i,j] else "?" if not run_data.observed[i,j] else str(int(run_data.responses[i,j])) for j in range(len(run_data.item_ids))))
        payload=animation_payload(run_data,run,selected_item,selected_group,int(person))
        st.iframe(playback_html(payload),height=950,tab_index=0)
        with st.expander("Expected item likelihood and prior contributions"):
            decomposition=pd.DataFrame([dict(cycle=f["cycle"],expected_loglik_before=f["expected_before"],expected_loglik_after=f["expected_after"],log_prior_before=f["prior_before"],log_prior_after=f["prior_after"],combined_before=f["q_before"],combined_after=f["q_after"]) for f in payload["frames"]])
            decomposition["relative_expected_likelihood_before"]=np.exp(decomposition.expected_loglik_before-np.maximum(decomposition.expected_loglik_before,decomposition.expected_loglik_after))
            decomposition["relative_expected_likelihood_after"]=np.exp(decomposition.expected_loglik_after-np.maximum(decomposition.expected_loglik_before,decomposition.expected_loglik_after))
            st.dataframe(decomposition,hide_index=True,width="stretch")
            st.caption("Relative expected likelihood is exp(Q − max(Q_before,Q_after)) within that cycle's fixed counts. Log-prior and combined-target columns use log units. These expected-count targets differ from the raw marginal likelihood.")
        st.download_button("Export this playback's actual numerical states",json.dumps(payload,indent=2),"mmle_playback.json","application/json")
    else:
        st.info("The bundled responses are ready. Run one cycle to inspect likelihood × prior → posterior → expected counts.")
    phase=st.selectbox("Equation phase",range(len(LESSONS)),format_func=lambda k:LESSONS[k][0])
    title,equation,explanation,notation,derivation=LESSONS[phase]
    eqtab,extab,notab=st.tabs(["Equation","Explanation","Notation"])
    with eqtab: st.latex(equation)
    with extab: st.write(explanation)
    with notab: st.latex(notation)
    if level=="Guided": st.write(explanation)
    with st.expander("Derivation and interpretation"):
        st.latex(derivation)
        if phase==3:
            for eq in SCORES[:5]: st.latex(eq)
            st.write("All free interior scores are zero at an optimum. At a boundary use constrained optimality conditions. Fixing c=0 removes its score and reduces a/b to 2PL. Priors add only to free-item targets; the original direct-coefficient objective has no optimization-transform Jacobian.")
    st.info("What changed: a recorded E/M state. Why: fractional counts fit the item curves and update the permitted population distributions. What to watch: missing exposure, locked anchors, scale constraints and separate density stability.")

with results_tab:
    st.subheader("3 · Inspect estimates, stability and recovery")
    if not run or not run.cycles: st.info("Run calibration to view recorded results.")
    else:
        last=run.cycles[-1]
        a,b,c=st.columns(3)
        a.metric("Item change",f"{last.item_change:.3g}")
        b.metric("Density weight change",f"{last.weight_change:.3g}")
        c.metric("Population moment change",f"{last.moment_change:.3g}")
        st.write(f"**Stop reason:** {run.stop_reason} · {len(run.cycles)} cycles · {run.elapsed:.2f}s")
        st.caption(f"Saved target: {'Bayes modal' if any([run.config.priors.a,run.config.priors.b,run.config.priors.c]) else 'Pure MMLE'} · {run.config.design} · {run.config.model} · {run.config.density} · D={run.config.D}")
        if not all(s["success"] for s in last.optimizer): st.warning("Some item optimizers need review. Inspect the diagnostic table.")
        st.dataframe(item_frame(run_data,run.params),hide_index=True,width="stretch")
        hf=history_frame(run)
        columns=st.columns(2)
        columns[0].plotly_chart(line_chart(hf.cycle,{"Raw marginal log likelihood":hf.raw_loglik,"Penalized objective":hf.penalized_objective},"Full updated-state objectives","Cycle","Objective"),width="stretch")
        columns[1].plotly_chart(line_chart(hf.cycle,{"Item Δ":hf.item_change,"Density Δ":hf.weight_change,"Moment Δ":hf.moment_change},"Separate stability diagnostics","Cycle","Maximum change"),width="stretch")
        ritem=st.selectbox("Result curve / trajectory item",range(len(run_data.item_ids)),format_func=lambda j:run_data.item_ids[j])
        cols=st.columns(2)
        xx=run.nodes
        cols[0].plotly_chart(line_chart(xx,{"Starting curve":probabilities(xx,run.initial_params[ritem],run.config.D)[0],"Final curve":probabilities(xx,run.params[ritem],run.config.D)[0]},"Item response curve"),width="stretch")
        path=np.array([run.initial_params[ritem]]+[cc.params[ritem] for cc in run.cycles])
        cols[1].plotly_chart(line_chart(np.arange(len(path)),{k:path[:,i] for i,k in enumerate(["a","b","c"]) if i<2 or run.config.model=="3PL"},"Recorded coefficient trajectory","State index","Coefficient"),width="stretch")
        st.plotly_chart(line_chart(xx,{name:run.weights[g] for g,name in enumerate(run_data.group_names)},"Fitted group population masses",ylabel="Probability mass"),width="stretch")
        mm=moments(xx,run.weights);initial_mm=moments(xx,run.initial_weights)
        st.dataframe(pd.DataFrame(dict(group=run_data.group_names,initial_mean=initial_mm[:,0],initial_SD=initial_mm[:,1],final_mean=mm[:,0],final_SD=mm[:,1])),hide_index=True)
        with st.expander("Numerical optimizer diagnostics and cycle table"):
            st.dataframe(pd.DataFrame(last.optimizer),hide_index=True)
            st.dataframe(hf,hide_index=True)
        if run_data.truth is not None:
            if run_data.generation["D"]!=run.config.D: st.info("Known-truth recovery unavailable: generating and fitted D differ.")
            else:
                rec=recovery(run_data,run)
                st.markdown("**Known-truth recovery · free items, each shared ID counted once**")
                st.caption(rec["alignment"])
                st.dataframe(pd.DataFrame(rec["metrics"]),hide_index=True)
                truth_table=item_frame(run_data,rec["raw_truth"]).rename(columns={k:"raw truth "+k for k in ["a","b","c"]})
                for i,k in enumerate(["a","b","c"]): truth_table["aligned truth "+k]=rec["aligned_truth"][:,i]
                st.dataframe(truth_table,hide_index=True,width="stretch")
                if rec["truth_grid_weights"] is not None:
                    st.plotly_chart(line_chart(run.nodes,{label:values for g,name in enumerate(run_data.group_names) for label,values in [(name+" fitted",run.weights[g]),(name+" true population on grid",rec["truth_grid_weights"][g])]},"Population shape recovery on the fitted scale",ylabel="Mass"),width="stretch")
                    st.caption("Continuous truth densities are renormalized on this finite grid; custom true masses are projected linearly to neighboring nodes. Total variation describes grid-mass recovery, not an exact continuous-density distance.")
                st.dataframe(pd.DataFrame(dict(group=run_data.group_names,true_mean_on_fit=rec["group_truth"][:,0],true_SD_on_fit=rec["group_truth"][:,1],estimated_mean=rec["group_estimates"][:,0],estimated_SD=rec["group_estimates"][:,1],density_TV=rec["density_TV"])),hide_index=True)
        st.caption("Approximate numerical M-steps, histogram rescaling, OEM timing and penalized targets need their own objective interpretation. A nonmonotone raw-likelihood trace is not automatically an algorithm failure.")
        st.download_button("Export complete run, cycle histories and validation record",export_zip(run_data,run,validation_record(run)),"mmle_experiment.zip","application/zip")

with validation_tab:
    st.subheader("4 · Validate the saved result")
    env=reference_environment()
    st.write(f"**Primary criterion: R irtQ endpoint agreement · {st.session_state.validation['status'] if st.session_state.get('validation') else env['status']}**")
    st.caption(env["reason"])
    st.write("Internal arithmetic checks and recovery cannot turn an unavailable or unexecuted irtQ comparison into a validation Pass.")
    with st.expander("Endpoint tolerances and matching record"):
        st.caption(BASELINE_NOTE)
        tol={}
        for name,value in DEFAULT_TOLERANCES.items(): tol[name]=st.number_input(name,min_value=0.,value=value,format="%.6f",key="tol_"+name)
        st.caption("Density distance = half the sum of absolute node-mass differences (total variation). Likelihoods are independently recomputed from both endpoints on the same saved grid. R starts, weights, D, priors, bank locks, IDs and schedules match the saved run; iteration paths need not match.")
    if run and run.cycles:
        valid_button=st.button("Validate against irtQ",type="primary")
        invariant_button=st.button("Run constraints, arithmetic and animation checks")
        sensitivity_button=st.button("Run refined-grid and changed-start sensitivity")
        if valid_button:
            with st.spinner("Running installed R irtQ on the same responses…"):
                st.session_state.validation=compare_endpoints(run_data,run,fit_irtq(run_data,run),tol)
            st.rerun()  # the banner and status header above were drawn before this result existed
        if invariant_button: st.session_state.invariants=(run.run_id,invariant_report(run_data,run))
        if sensitivity_button:
            with st.spinner("Fitting the saved data with a refined grid and changed start…"):
                st.session_state.sensitivity=(run.run_id,sensitivity_checks(run_data,run))
        report=st.session_state.get("validation")
        if report:
            st.write(f"**Primary reference status: {report['status']}**")
            st.caption(report.get("reason",""))
            if report.get("executed"):
                executed=pd.DataFrame(report["executed"])
                executed["value"]=[f"{v:.3g}" for v in executed["value"]]  # magnitudes span 1e-16 to 1e-2
                st.dataframe(executed,hide_index=True,width="stretch")
            if report.get("endpoint_rows"):
                st.dataframe(pd.DataFrame(report["endpoint_rows"]),hide_index=True,width="stretch")
            with st.expander("Full reference report, settings, versions and pending checks"): st.json(jsonable(report))
            st.download_button("Download reproducible validation record",json.dumps(jsonable(validation_record(run)),indent=2),"mmle_validation.json","application/json")
        inv=st.session_state.get("invariants")
        if inv and inv[0]==run.run_id: st.dataframe(pd.DataFrame(inv[1]),hide_index=True,width="stretch")
        else: st.caption("Constraints / arithmetic / animation checks: Not run for this saved run.")
        sen=st.session_state.get("sensitivity")
        if sen and sen[0]==run.run_id: st.dataframe(pd.DataFrame(sen[1]),hide_index=True)
        else: st.caption("Grid and start sensitivity: Not run.")
        if run_data.truth is not None:
            replications=st.number_input("Recovery replications (additional generated datasets)",2,20,3)
            if st.button("Run repeated-simulation recovery",disabled=run_data.generation["D"]!=run.config.D):
                try:
                    summaries=[]
                    population_summaries=[]
                    gen=deepcopy(run_data.generation)
                    with st.spinner("Running separately seeded recovery replications…"):
                        for k in range(int(replications)):
                            args={name:gen[name] for name in ["n_per_group","items","groups","model","D","common","anchors","missing","a_range","b_range","c_range","true_params","form_map","bank_params","drift"]}
                            dd=simulate(seed=int(gen["seed"])+k+1,**args);rr=fit(dd,run.config);rc=recovery(dd,rr)
                            for row in rc["metrics"]: summaries.append(dict(replication=k+1,seed=int(gen["seed"])+k+1,stop_reason=rr.stop_reason,**row))
                            for g,name in enumerate(dd.group_names):
                                population_summaries.append(dict(replication=k+1,group=name,mean_error=float(rc["group_estimates"][g,0]-rc["group_truth"][g,0]),SD_error=float(rc["group_estimates"][g,1]-rc["group_truth"][g,1]),density_TV=float(rc["density_TV"][g])))
                    st.session_state.replicated=(run.run_id,dict(items=summaries,populations=population_summaries))
                except (ValueError,KeyError) as e: st.error(str(e))
            rep=st.session_state.get("replicated")
            if rep and rep[0]==run.run_id:
                repframe=pd.DataFrame(rep[1]["items"]);st.dataframe(repframe,hide_index=True)
                if not repframe.empty: st.dataframe(repframe.groupby("parameter")[["bias","RMSE"]].agg(["mean","std"]))
                popframe=pd.DataFrame(rep[1]["populations"]);st.dataframe(popframe,hide_index=True)
                if not popframe.empty: st.dataframe(popframe.groupby("group")[["mean_error","SD_error","density_TV"]].agg(["mean","std"]))
                st.caption("Finite-sample recovery, not exact-truth acceptance. No standard-error or uncertainty-coverage claim is made.")
    else: st.info("Fit the Python model first, then execute the primary and supplementary checks.")

with compare_tab:
    st.subheader("5 · Compare immutable experiments on saved responses")
    if run and st.button("Save current experiment for comparison"):
        st.session_state.experiments.append(dict(result=deepcopy(run),data=deepcopy(run_data),validation=deepcopy(st.session_state.get("validation"))))
        st.success("Saved an immutable snapshot of the run and data.")
    saved=st.session_state.experiments
    if saved:
        summary=pd.DataFrame([dict(experiment=i+1,run=e["result"].run_id,data=e["result"].data_id,design=e["result"].config.design,
                                   model=e["result"].config.model,density=e["result"].config.density,schedule=e["result"].config.schedule,
                                   objective=e["result"].cycles[-1].loglik,validation=(e["validation"] or {}).get("status","Not run")) for i,e in enumerate(saved)])
        st.dataframe(summary,hide_index=True,width="stretch")
        selected=st.multiselect("Experiments to compare",range(len(saved)),format_func=lambda i:f"{i+1} · {saved[i]['result'].config.design} / {saved[i]['result'].config.density}",default=list(range(max(0,len(saved)-2),len(saved))))
        if len(selected)>=2:
            exps=[saved[i] for i in selected];base=exps[0]["result"]
            def scale_key(e):
                r=e["result"]
                dd=e["data"]
                target=moments(r.nodes,r.initial_weights[0]) if r.config.initial_weights is not None else [r.config.mean,r.config.sd]
                if dd.truth is not None:
                    mt,st_true=dd.generation["groups"][0]["mean"],dd.generation["groups"][0]["sd"]
                    slope,shift=(1.,0.) if r.config.anchored else (target[1]/st_true,target[0]-target[1]/st_true*mt)
                    return signature([r.config.D,"simulation coordinates",round(slope,8),round(shift,8)])
                return signature([r.config.D,"bank",dd.bank]) if r.config.anchored else signature([r.config.D,"reference",dd.group_names[0],target])
            compatible=len({e["result"].data_id for e in exps})==1 and len({scale_key(e) for e in exps})==1
            if not compatible: st.warning("Selected experiments have different responses or incompatible scale constraints. Direct coefficient / likelihood ranking is unavailable.")
            else:
                st.caption("Same responses and compatible scale. Each row below lists the saved fitting settings; changed assumptions are explicit.")
                st.dataframe(pd.DataFrame([dict(experiment=i+1,**{k:str(v) for k,v in asdict(saved[i]["result"].config).items()}) for i in selected]).astype(str).T,width="stretch")
                j=st.selectbox("Comparison item",range(len(exps[0]["data"].item_ids)),format_func=lambda j:exps[0]["data"].item_ids[j])
                xx=np.linspace(-6,6,151)
                st.plotly_chart(line_chart(xx,{f"Experiment {i+1}":probabilities(xx,saved[i]["result"].params[j],saved[i]["result"].config.D)[0] for i in selected},"Saved endpoint curves"),width="stretch")
                st.dataframe(pd.DataFrame([dict(experiment=i+1,a=saved[i]["result"].params[j,0],b=saved[i]["result"].params[j,1],c=saved[i]["result"].params[j,2],raw_loglik=saved[i]["result"].cycles[-1].loglik) for i in selected]),hide_index=True)
    else: st.info("Save at least two completed fits to compare normal versus histogram, OEM versus MEM, or item-prior assumptions on the same responses.")

with notes_tab:
    st.subheader("6 · Equations, constraints and method notes")
    for title,equation,explanation,notation,derivation in LESSONS:
        with st.expander(title,expanded=level=="Explore"):
            st.latex(equation);st.write(explanation);st.latex(notation);st.latex(derivation)
    with st.expander("Full 3PL a/b/c scores, priors and recovery alignment",expanded=True):
        for eq in SCORES: st.latex(eq)
        st.write("The expected counts remain fixed within each item M-step. Free interior scores vanish at an optimum; constrained boundary optima require boundary conditions. 2PL fixes c=0. The lognormal prior applies to D×a; its induced density on a has a constant factor D. No parameter-dependent transform Jacobian is added. Curated prior scope is all relevant free items; per-item prior families are a future extension.")
        st.write("Recovery uses one shared reference transformation for all items and groups. Bank fits retain the supplied bank scale and intentional drift; fixed anchors are excluded from free-item RMSE. Reference histograms use interpolation, so achieved grid moments may differ slightly from their targets.")
    st.dataframe(pd.DataFrame(dict(symbol=list(NOTATION),meaning=list(NOTATION.values()))),hide_index=True,width="stretch")
    st.write("**Schedules:** OEM uses fixed-item posteriors under initial population masses to fit free items once, then returns a density update without feedback. MEM uses fixed items in cycle one and all observed administered items thereafter. When every item is fixed, all anchor responses are used and no item optimization occurs.")
    st.write("**Reference strategy:** irtQ is the primary endpoint reference. Python is independently implemented with SciPy; R is used only for validation. R console iteration objectives use different timing from the app's full updated-state objective. Endpoint agreement does not establish instrument validity, global optimality or finite-sample recovery.")
    st.markdown("Reference: [irtQ single-group estimation](https://hwangq.github.io/irtQ/reference/est_irt.html), [multiple-group / MG-FIPC](https://hwangq.github.io/irtQ/reference/est_mg.html), [reviewed source commit](https://github.com/hwangQ/irtQ/tree/5ad30cd2aa3a7e6a74964df3162e70b0352e7ebc).")
    st.caption(f"Engine {VERSION}. Reviewed design: MMLE EM educational app design.md. Native reference comparison uses an installed package, never a substitute implementation. Experimental software; standard errors, coverage, polytomous models and arbitrary coefficient-specific anchors are outside this release.")
