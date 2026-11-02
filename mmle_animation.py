"""Self-contained, accessible playback of recorded Python estimation states."""
import json
import numpy as np
from scipy.special import logsumexp
from mmle_engine import log_probabilities, probabilities, moments, jsonable, log_prior


def animation_payload(data,result,item=0,group=0,person=0):
    """Reconstruct one person's exact E-step from each saved INPUT state."""
    if data.group[person] != group:
        raise ValueError("Selected person must belong to the selected group")
    x=result.nodes
    frames=[]
    for c in result.cycles:
        mask=data.observed[person] & c.likelihood_items
        y=np.where(mask,data.responses[person],0.)
        lp,l1p=log_probabilities(x,c.input_params,result.config.D)
        ll=y@lp+(mask-y)@l1p
        with np.errstate(divide="ignore"):
            joint=ll+np.log(c.input_weights[group])
        post=np.exp(joint-logsumexp(joint))
        relative=np.exp(ll-np.max(ll))
        weighted=relative*c.input_weights[group]
        observed=bool(data.observed[person,item])
        pooled_N=c.expected_N[:,item].sum(axis=0)
        pooled_R=c.expected_R[:,item].sum(axis=0)
        def components_at(params):
            ilp,il1=log_probabilities(x,params,result.config.D)
            expected=float(np.sum(pooled_R*ilp[0]+(pooled_N-pooled_R)*il1[0]))
            prior=0. if data.fixed[item] else log_prior(params,result.config.priors,result.config.D,result.config.model)
            return expected,prior
        expected_before,prior_before=components_at(c.input_params[item])
        expected_after,prior_after=components_at(c.params[item])
        frames.append(dict(cycle=c.index,input=c.input_params[item],output=c.params[item],
                           expected_before=expected_before,prior_before=prior_before,
                           expected_after=expected_after,prior_after=prior_after,
                           likelihood=relative,log_likelihood=ll,prior=c.input_weights[group],
                           weighted=weighted,posterior=post,person_N=post*observed,
                           person_R=post*observed*(data.responses[person,item] if observed else 0),
                           N=c.expected_N[group,item],R=c.expected_R[group,item],
                           pooled_N=c.expected_N[:,item].sum(axis=0),pooled_R=c.expected_R[:,item].sum(axis=0),
                           raw=c.raw_weights[group],next=c.weights[group],
                           curve_old=probabilities(x,c.input_params[item],result.config.D)[0],
                           curve_new=probabilities(x,c.params[item],result.config.D)[0],
                           moments=moments(x,c.weights[group]),
                           loglik=c.loglik,objective=c.penalized,dp=c.item_change,dw=c.weight_change,dm=c.moment_change,
                           q_before=c.q_before[item],q_after=c.q_after[item],
                           optimizer=c.optimizer[item],anchor_only=not c.likelihood_items.all(),
                           item_observed=observed))
    return jsonable(dict(x=x,frames=frames,item=data.item_ids[item],group=data.group_names[group],
                         person=int(person+1),locked=bool(data.fixed[item]),model=result.config.model,
                         run_id=result.run_id,stop=result.stop_reason,bayes=any([result.config.priors.a,result.config.priors.b,result.config.priors.c])))


def playback_html(payload):
    body = r'''<!doctype html><html><head><meta charset="utf-8"><style>
:root{font-family:system-ui,sans-serif;color:#17273f;font-size:14px}body{margin:0;background:#f6f8fc}.wrap{padding:16px;border:1px solid #d7deea;border-radius:12px;background:white}button,select,input{font:inherit}button,select{border:1px solid #c2cce0;border-radius:6px;background:#fff;padding:7px 10px;color:#234c9e;cursor:pointer}button:focus,input:focus,select:focus{outline:3px solid #88b5f9}button.active{background:#e7efff}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid #dae1ed;border-radius:8px;padding:10px;min-width:0}.title{font-weight:650;font-size:14px}.muted{color:#607086;font-size:12px}.status{padding:12px;border-radius:6px;background:#edf3ff;line-height:1.55;margin:12px 0}svg{width:100%;height:190px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:5px;text-align:right;border-bottom:1px solid #e1e6ee}td:first-child,th:first-child{text-align:left}.scroll{max-height:230px;overflow:auto}summary{cursor:pointer;padding:8px}h3{margin:6px 0}.tag{font-size:12px;border-radius:4px;padding:3px 6px;background:#e8f4ee}.small{font-size:12px}#cycle{min-width:150px;flex:1}label{font-size:12px}.phase{font-size:16px;font-weight:650}.range{width:54px}@media(max-width:750px){.cards{grid-template-columns:1fr}svg{height:170px}}</style></head><body><div class="wrap" tabindex="0" id="root" aria-label="Recorded EM playback">
<h3 id="heading"></h3><div class="muted" id="identity"></div>
<div class="controls"><button id="emode" class="active">E-step playback</button><button id="cmode">Convergence playback</button><label><input type="checkbox" id="reduced"> Reduced motion / manual stepping</label></div>
<div class="controls"><button id="prev" aria-label="Previous recorded step">← Previous</button><button id="play">▶ Play</button><button id="next">Next →</button><button id="restart">Restart</button><label>Speed <select id="speed"><option value="1500">Slow</option><option value="800" selected>Normal</option><option value="300">Fast</option></select></label><label>Range <input class="range" id="lo" type="number" min="1" value="1"> to <input class="range" id="hi" type="number" min="1"></label></div>
<div class="controls"><label for="cycle">Recorded cycle</label><input id="cycle" type="range" min="1" step="1"><span id="counter"></span><label>Phase <select id="phase"><option value="0">1 · Likelihood</option><option value="1">2 · Multiply by prior</option><option value="2">3 · Normalize posterior</option><option value="3">4 · Fractional counts</option><option value="4">5 · Item M-step</option><option value="5">6 · Population / cycle check</option></select></label></div>
<div class="status" aria-live="polite"><div id="phaseTitle" class="phase"></div><div id="explanation"></div><div id="diagnostic" class="small"></div></div>
<div class="cards"><div class="card"><div id="t1" class="title"></div><svg id="chart1" role="img"></svg></div><div class="card"><div id="t2" class="title"></div><svg id="chart2" role="img"></svg></div><div class="card"><div id="t3" class="title"></div><svg id="chart3" role="img"></svg></div></div>
<div class="cards" style="margin-top:12px"><div class="card"><div class="title">Item response curve · old → updated</div><svg id="curve" role="img"></svg></div><div class="card"><div class="title">Group population · input / raw / next</div><svg id="density" role="img"></svg></div><div class="card"><div class="title">Coefficient values · one shared item</div><table id="params"></table><div id="q" class="muted"></div></div></div>
<details><summary>Accessible node table · actual recorded/reconstructed values</summary><div class="scroll"><table id="nodes"></table></div></details>
<div class="muted">Blue: input / a · Green: update / b · Orange: raw / c. Likelihood is scaled by its maximum for display only; raw log likelihood is in the node table. No interpolated estimates. Use arrow keys to step; Space toggles playback.</div>
</div><script>
const d=PAYLOAD;let ix=0,phase=0,mode='e',timer=null;const $=id=>document.getElementById(id);const n=d.frames.length;
$('heading').textContent=`${d.item} ${d.locked?'🔒 Bank anchor':''} · ${d.group} · Person ${d.person}`;
$('identity').textContent=`Saved run ${d.run_id} · ${n} recorded cycles · ${d.stop}`;
$('hi').value=n;$('lo').max=n;$('hi').max=n;$('cycle').max=n;$('cycle').value=1;
$('reduced').checked=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function fmt(v){return v===null?'—':Number(v).toPrecision(5)}
function bounds(){let l=Math.max(0,Math.min(n-1,Number($('lo').value)-1));let h=Math.max(l,Math.min(n-1,Number($('hi').value)-1));return [l,h]}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='▶ Play'}
function play(){if(timer){stop();return}if($('reduced').checked)return;let [l,h]=bounds();if(ix>=h && (mode==='c'||phase===5)){ix=l;phase=0}timer=setInterval(()=>{advance(1,true)},Number($('speed').value));$('play').textContent='❚❚ Pause';render()}
function advance(step,automatic=false){let [l,h]=bounds();if(mode==='e'){phase+=step;if(phase>5){phase=0;ix++}if(phase<0){phase=5;ix--}}else{ix+=step;phase=5}if(ix>h){ix=h;phase=5;stop()}if(ix<l){ix=l;phase=0}if(!automatic)stop();render()}
function chart(id,series,x=d.x,ylabel='Value',startZero=true){const el=$(id),W=330,H=190,L=44,R=10,T=18,B=38;let all=series.flatMap(s=>s.y).filter(Number.isFinite);let ymin=startZero?Math.min(0,...all):Math.min(...all),ymax=Math.max(...all);if(ymax===ymin)ymax=ymin+1;let xmin=Math.min(...x),xmax=Math.max(...x);if(xmax===xmin)xmax=xmin+1;const sx=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),sy=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);let svg=`<line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="#9cacc4"/><line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#9cacc4"/>`;for(let i=0;i<3;i++){let v=ymin+(ymax-ymin)*i/2;svg+=`<text x="${L-4}" y="${sy(v)+4}" text-anchor="end" font-size="9" fill="#607086">${v.toPrecision(3)}</text>`}for(let v of [xmin,(xmin+xmax)/2,xmax])svg+=`<text x="${sx(v)}" y="${H-B+15}" text-anchor="middle" font-size="10" fill="#607086">${v.toPrecision(3)}</text>`;series.forEach((s,i)=>{const points=s.y.map((v,k)=>`${sx(x[k])},${sy(v)}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${s.color||['#2d5eb2','#27856f','#c58a41'][i]}" stroke-width="2.2"/><text x="${L+i*90}" y="12" fill="${s.color||['#2d5eb2','#27856f','#c58a41'][i]}" font-size="9">${s.name}</text>`});svg+=`<text x="${W/2}" y="${H-3}" text-anchor="middle" font-size="10" fill="#607086">${mode==='c'&&['chart1','chart2','chart3'].includes(id)?'Recorded cycle':'Ability θ'} · ${ylabel}</text>`;el.setAttribute('viewBox',`0 0 ${W} ${H}`);el.innerHTML=svg;el.setAttribute('aria-label',`${id}: ${series.map(s=>s.name).join(', ')}; values available in table`)}
const phases=['Likelihood over ability','Likelihood × population prior','Normalize the person posterior','Fractional expected counts','Coupled numerical item M-step','Population update and cycle check'];
function render(){const f=d.frames[ix];$('cycle').value=ix+1;$('counter').textContent=`${ix+1} / ${n}`;$('phase').value=phase;$('phase').disabled=mode==='c';$('emode').className=mode==='e'?'active':'';$('cmode').className=mode==='c'?'active':'';let sum=f.posterior.reduce((a,b)=>a+b,0);$('phaseTitle').textContent=mode==='e'?phases[phase]:`Convergence · output state ψ^(${f.cycle})`;
$('explanation').textContent=`E-step input ψ^(${f.cycle-1}) → posterior τ^(${f.cycle-1}) → M-step ψ^(${f.cycle}). ${f.anchor_only?'Fixed bank responses inform this E-step.':'All observed administered responses inform this E-step.'} Selected-person posterior sum = ${sum.toFixed(12)}.`;
$('diagnostic').textContent=`Raw log likelihood ${fmt(f.loglik)} · ${d.bayes?'Penalized objective '+fmt(f.objective)+' · ':''}item Δ ${fmt(f.dp)} · density Δ ${fmt(f.dw)} · moment Δ ${fmt(f.dm)} · ${f.optimizer.success?'Optimizer OK':'Optimizer needs review'}`;
if(mode==='e'){
 let series=phase===0?[[{name:'Relative likelihood',y:f.likelihood}],[{name:'Input population',y:f.prior}],[{name:'Posterior (next phase)',y:f.posterior}]]:phase===1?[[{name:'Likelihood',y:f.likelihood}],[{name:'Likelihood × prior',y:f.weighted}],[{name:'Posterior',y:f.posterior}]]:phase===2?[[{name:'Weighted likelihood',y:f.weighted}],[{name:'Posterior',y:f.posterior}],[{name:'Input population',y:f.prior}]]:phase===3?[[{name:'Person N',y:f.person_N},{name:'Person R',y:f.person_R}],[{name:'Group N',y:f.N},{name:'Group R',y:f.R}],[{name:'Pooled N',y:f.pooled_N},{name:'Pooled R',y:f.pooled_R}]]:phase===4?[[{name:'Old curve',y:f.curve_old},{name:'Updated curve',y:f.curve_new}],[{name:'Pooled N',y:f.pooled_N},{name:'Pooled R',y:f.pooled_R}],[{name:'Posterior',y:f.posterior}]]:[[{name:'Input',y:f.prior},{name:'Raw posterior avg',y:f.raw},{name:'Next',y:f.next}],[{name:'Person posterior',y:f.posterior}],[{name:'Old curve',y:f.curve_old},{name:'Updated curve',y:f.curve_new}]];
 for(let k=0;k<3;k++){ $('t'+(k+1)).textContent=series[k].map(s=>s.name).join(' / ');chart('chart'+(k+1),series[k])}
 if(phase===3)$('explanation').textContent+=f.item_observed?' This person contributes posterior fractions to the selected item.':' Selected item is missing or absent for this person: both count contributions are zero.';
}else{
 const hist=d.frames.slice(0,ix+1),cx=hist.map(f=>f.cycle);$('t1').textContent='Actual item coefficient trajectories';chart('chart1',['a','b','c'].slice(0,d.model==='2PL'?2:3).map((name,k)=>({name,y:hist.map(f=>f.output[k])})),cx,'coefficient',false);
 $('t2').textContent=d.bayes?'Raw likelihood / penalized objective':'Full-state raw marginal log likelihood';chart('chart2',[{name:'Raw log likelihood',y:hist.map(f=>f.loglik)},...(d.bayes?[{name:'Penalized',y:hist.map(f=>f.objective)}]:[])],cx,'objective',false);
 $('t3').textContent='Group mean / SD over cycles';chart('chart3',[{name:'Mean',y:hist.map(f=>f.moments[0])},{name:'SD',y:hist.map(f=>f.moments[1])}],cx,'moments',false);
}
chart('curve',[{name:'Input',y:f.curve_old},{name:'Updated',y:f.curve_new}],d.x,'P(correct)');chart('density',[{name:'Input',y:f.prior},{name:'Raw average',y:f.raw},{name:'Next state',y:f.next}],d.x,'mass');
$('params').innerHTML='<tr><th>Coefficient</th><th>Input</th><th>Updated</th></tr>'+['a','b','c'].map((k,i)=>`<tr><td>${k}${d.locked?' 🔒':''}</td><td>${fmt(f.input[i])}</td><td>${fmt(f.output[i])}</td></tr>`).join('');$('q').textContent=`Expected log likelihood ${fmt(f.expected_before)} → ${fmt(f.expected_after)}; log prior ${fmt(f.prior_before)} → ${fmt(f.prior_after)}; combined target ${fmt(f.q_before)} → ${fmt(f.q_after)}. ${d.locked?'No M-step: bank coefficients remain fixed.':'Counts remain fixed during the joint numerical M-step.'}`;
const fields=[['θ',d.x],['log L',f.log_likelihood],['Prior',f.prior],['τ',f.posterior],['Person N',f.person_N],['Person R',f.person_R],['Group N',f.N],['Group R',f.R],['Raw group',f.raw],['Next group',f.next]];
$('nodes').innerHTML='<tr>'+fields.map(v=>'<th>'+v[0]+'</th>').join('')+'</tr>'+d.x.map((_,q)=>'<tr>'+fields.map(v=>'<td>'+fmt(v[1][q])+'</td>').join('')+'</tr>').join('');}
$('play').onclick=play;$('prev').onclick=()=>advance(-1);$('next').onclick=()=>advance(1);$('restart').onclick=()=>{stop();ix=bounds()[0];phase=0;render()};$('cycle').oninput=()=>{stop();ix=Number($('cycle').value)-1;render()};$('phase').onchange=()=>{stop();phase=Number($('phase').value);render()};$('emode').onclick=()=>{stop();mode='e';phase=0;render()};$('cmode').onclick=()=>{stop();mode='c';phase=5;render()};$('speed').onchange=()=>{if(timer){stop();play()}};$('reduced').onchange=stop;['lo','hi'].forEach(id=>$(id).onchange=()=>{stop();ix=Math.max(bounds()[0],Math.min(bounds()[1],ix));render()});$('root').onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();play()}if(e.code==='ArrowRight'){e.preventDefault();advance(1)}if(e.code==='ArrowLeft'){e.preventDefault();advance(-1)}};render();
</script></body></html>'''
    return body.replace('PAYLOAD',json.dumps(payload,ensure_ascii=True).replace('</','<\\/'))
