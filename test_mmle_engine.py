"""Numerical and state-contract tests for the independent MMLE engine.
Run: python3 -m unittest test_mmle_engine test_mmle_app -v
"""
import unittest
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from mmle_engine import *
from mmle_animation import animation_payload,playback_html
from unittest import mock
import os
import pandas as pd
import mmle_reference
from mmle_reference import check_irtq,fit_irtq,compare_endpoints,r_environment,STATIONARITY_FACTOR


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.data=simulate(120,8,seed=53,missing=.1)

    def test_gradient_likelihood_and_priors(self):
        rng=np.random.default_rng(9)
        for model in ['2PL','3PL']:
            for D in [1.,1.702]:
                for priors in [Priors(),Priors(True,True,model=='3PL')]:
                    cfg=FitConfig(model=model,D=D,priors=priors)
                    for _ in range(8):
                        x=np.linspace(-6,6,49);N=rng.uniform(10,40,49);R=N*rng.uniform(.02,.98,49)
                        p=np.array([rng.uniform(.4,2),rng.uniform(-2,2),rng.uniform(.05,.35) if model=='3PL' else 0])
                        val,grad=item_objective(p,x,N,R,cfg)
                        k=2 if model=='2PL' else 3
                        fun=lambda v:item_objective(np.r_[v,0.] if k==2 else v,x,N,R,cfg)[0]
                        num=approx_derivative(fun,p[:k],method='3-point').ravel()
                        np.testing.assert_allclose(grad[:k],num,rtol=1e-5,atol=1e-5)

    def test_logspace_extremes(self):
        p=np.array([[8.,-8.,.1],[-8.,8.,0.]])
        lp,l1=log_probabilities(np.array([-100.,100.]),p)
        self.assertTrue(np.isfinite(lp).all());self.assertTrue(np.isfinite(l1).all())
        np.testing.assert_allclose(np.exp(lp)+np.exp(l1),1.,atol=1e-15)

    def test_posterior_and_exposure_masks(self):
        cfg=FitConfig(max_cycles=3);r=initialize(self.data,cfg)
        self.data.administered[0,0]=False;self.data.responses[0,0]=np.nan
        self.data.responses[1,0]=np.nan
        tau,_,_=estep(self.data,r.params,r.weights,r.nodes,1.)
        N,R=expected_counts(self.data,tau)
        np.testing.assert_allclose(tau.sum(axis=1),1.,atol=1e-12)
        np.testing.assert_allclose(N.sum(axis=-1)[0],self.data.observed.sum(axis=0))
        np.testing.assert_allclose(R.sum(axis=-1)[0],np.nansum(self.data.responses,axis=0))

    def test_all_missing_person_gets_current_prior(self):
        d=deepcopy(self.data);d.responses[0]=np.nan
        r=initialize(d,FitConfig())
        tau,_,m=estep(d,r.params,r.weights,r.nodes,1.)
        np.testing.assert_allclose(tau[0],r.weights[0]);self.assertAlmostEqual(m[0],0.)

    def test_all_supported_design_model_density_combinations(self):
        for design in DESIGNS:
            for model in ['2PL','3PL']:
                for density in ['Normal','Histogram']:
                    with self.subTest(design=design,model=model,density=density):
                        mg=design in (DESIGNS[1],DESIGNS[3]);anchored=design in DESIGNS[2:]
                        groups=[dict(name=f'G{g}',mean=.3*g,sd=1.,shape='Normal') for g in range(2 if mg else 1)]
                        d=simulate(120,10,groups,model,common=6,anchors=4 if anchored else 0,missing=.12)
                        r=fit(d,FitConfig(design=design,model=model,density=density,max_cycles=8,priors=Priors(c=model=='3PL')))
                        self.assertTrue(all(row['status']=='Pass' for row in invariant_report(d,r)))
                        for c in r.cycles: np.testing.assert_array_equal(c.params[d.fixed],d.bank[d.fixed])

    def test_shared_counts_pool_once(self):
        d=simulate(100,8,[dict(name='G1',mean=0,sd=1),dict(name='G2',mean=.5,sd=1)],common=4)
        r=fit(d,FitConfig(design=DESIGNS[1],max_cycles=1))
        c=r.cycles[0];j=0
        # A genuinely joint optimizer on the pooled group counts equals the saved update.
        cfg=r.config
        val,_=item_objective(c.params[j],r.nodes,c.expected_N[:,j].sum(axis=0),c.expected_R[:,j].sum(axis=0),cfg)
        self.assertAlmostEqual(-val,c.q_after[j],places=7)
        self.assertEqual(r.params.shape,(12,3))

    def test_oem_does_not_feed_back_returned_density(self):
        d=simulate(160,10,anchors=5)
        normal=fit(d,FitConfig(design=DESIGNS[2],schedule='OEM'))
        hist=fit(d,FitConfig(design=DESIGNS[2],schedule='OEM',density='Histogram'))
        self.assertEqual(len(normal.cycles),1);self.assertIn('One cycle completed',normal.stop_reason)
        np.testing.assert_allclose(normal.params,hist.params,atol=1e-12)
        self.assertFalse(np.allclose(normal.weights,hist.weights))
        self.assertTrue(np.array_equal(normal.cycles[0].likelihood_items,d.fixed))

    def test_oem_across_anchored_designs_models_and_densities(self):
        for design in DESIGNS[2:]:
            for model in ['2PL','3PL']:
                groups=[dict(name=f'G{g}',mean=.3*g,sd=1.) for g in range(2 if design==DESIGNS[3] else 1)]
                d=simulate(120,10,groups,model,anchors=4,common=6,missing=.1)
                fits=[]
                for density in ['Normal','Histogram']:
                    cfg=FitConfig(design=design,model=model,density=density,schedule='OEM',priors=Priors(c=model=='3PL'))
                    r=fit(d,cfg);fits.append(r)
                    self.assertEqual(len(r.cycles),1)
                    self.assertTrue(all(row['status']=='Pass' for row in invariant_report(d,r)))
                np.testing.assert_allclose(fits[0].params,fits[1].params,atol=1e-12)

    def test_mem_likelihood_timing_and_resume(self):
        d=simulate(150,10,anchors=4)
        cfg=FitConfig(design=DESIGNS[2],max_cycles=3)
        one=fit(d,cfg,cycles=1);resumed=fit(d,cfg,previous=one,cycles=2);full=fit(d,cfg)
        np.testing.assert_allclose(resumed.params,full.params)
        self.assertEqual(len(one.cycles),1)
        self.assertTrue(np.array_equal(full.cycles[0].likelihood_items,d.fixed))
        self.assertTrue(full.cycles[1].likelihood_items.all())
        np.testing.assert_allclose(full.cycles[1].input_weights,full.cycles[0].weights)
        np.testing.assert_allclose(full.cycles[1].input_params,full.cycles[0].params)

    def test_changed_bank_blocks_resume(self):
        d=simulate(120,8,anchors=3);cfg=FitConfig(design=DESIGNS[2],max_cycles=2)
        r=fit(d,cfg,cycles=1);d.bank[0,1]+=.1
        with self.assertRaisesRegex(ValueError,'Resume'):fit(d,cfg,previous=r)

    def test_fixed_normal_and_histogram_reference(self):
        r=fit(self.data,FitConfig(max_cycles=8))
        for c in r.cycles: np.testing.assert_array_equal(c.weights,r.initial_weights)
        h=fit(self.data,FitConfig(density='Histogram',max_cycles=8))
        mm=moments(h.nodes,h.weights[0]);self.assertLess(abs(mm[0]),.03);self.assertLess(abs(mm[1]-1),.03)
        self.assertFalse(np.allclose(h.weights,h.initial_weights))

    def test_free_group_update_matches_posterior_moments(self):
        d=simulate(120,8,anchors=4);r=fit(d,FitConfig(design=DESIGNS[2],max_cycles=2))
        for c in r.cycles:
            mu,sd=moments(r.nodes,c.raw_weights[0]);np.testing.assert_allclose(c.weights[0],normal_weights(r.nodes,mu,sd))

    def test_all_fixed_items(self):
        d=simulate(100,8,anchors=8);r=fit(d,FitConfig(design=DESIGNS[2],max_cycles=5))
        np.testing.assert_array_equal(r.params,d.bank)
        self.assertTrue(all(s['locked'] for c in r.cycles for s in c.optimizer))

    def test_truth_alignment_preserves_curves(self):
        p=np.array([[1.2,-.3,.2],[.7,2.,0.]])
        theta=np.linspace(-4,4,40);mt,st,mf,sf=1.7,1.4,-.5,.8
        aligned=align_truth(p,mt,st,mf,sf)
        np.testing.assert_allclose(probabilities(theta,p,1.702),probabilities(mf+sf/st*(theta-mt),aligned,1.702),atol=1e-14)

    def test_one_shared_truth_alignment_for_concurrent(self):
        d=simulate(100,8,[dict(name='G1',mean=1.,sd=1.5),dict(name='G2',mean=2.,sd=.8)],common=4)
        r=fit(d,FitConfig(design=DESIGNS[1],mean=-.5,sd=.8,max_cycles=2))
        rec=recovery(d,r)
        np.testing.assert_allclose(rec['aligned_truth'],align_truth(d.truth,1.,1.5,-.5,.8))
        np.testing.assert_allclose(rec['group_truth'][1],[-.5+.8/1.5, .8*.8/1.5])

    def test_bank_drift_not_transformed_away(self):
        d=simulate(100,8,anchors=4,drift=.5);r=fit(d,FitConfig(design=DESIGNS[2],max_cycles=2));rc=recovery(d,r)
        np.testing.assert_array_equal(rc['aligned_truth'],d.truth)
        self.assertAlmostEqual(r.params[0,1]-d.truth[0,1],.5)

    def test_reproducible_generation(self):
        d=simulate(120,8,seed=53,missing=.1)
        self.assertEqual(d.identity,self.data.identity)
        before=self.data.identity;fit(self.data,FitConfig(density='Histogram',max_cycles=2));self.assertEqual(self.data.identity,before)

    def test_custom_generator_uses_literal_nodes_and_moments(self):
        spec=dict(name='G1',mean=100.,sd=10.,shape='Custom',nodes=[-1.,2.],weights=[1.,3.])
        d=simulate(100,8,[spec]);self.assertTrue(np.isin(d.theta,[-1.,2.]).all())
        self.assertAlmostEqual(d.generation['groups'][0]['mean'],1.25)

    def test_custom_initial_masses_preserved_for_fixed_normal(self):
        x=np.linspace(-6,6,49);w=normal_weights(x,1.,.8)
        r=fit(self.data,FitConfig(max_cycles=2,initial_weights=w.tolist()))
        np.testing.assert_allclose(r.weights[0],w)

    def test_disconnected_design_rejected(self):
        d=simulate(100,6,[dict(name='G1',mean=0,sd=1),dict(name='G2',mean=0,sd=1)],common=0)
        with self.assertRaisesRegex(ValueError,'No observed item path'):check_design(d,FitConfig(design=DESIGNS[1]))

    def test_invalid_response_and_partial_bank_rejected(self):
        d=deepcopy(self.data);d.responses[0,0]=2
        with self.assertRaises(ValueError):d.validate()
        d=deepcopy(self.data);d.bank[0,0]=1
        with self.assertRaises(ValueError):d.validate()

    def test_prior_scaled_slope_and_boundaries(self):
        p=np.array([1.1,.2,.25]);pr=Priors(True,True,True)
        value,score=log_prior(p,pr,1.702,'3PL',True)
        num=approx_derivative(lambda v:log_prior(v,pr,1.702,'3PL'),p).ravel()
        np.testing.assert_allclose(score,num,atol=1e-6)
        self.assertTrue(np.isneginf(log_prior([-1.,0.,.2],pr)))
        self.assertTrue(np.isneginf(log_prior([1.,0.,0.],pr)))

    def test_full_state_objective_and_q_ascent(self):
        r=fit(self.data,FitConfig(max_cycles=10))
        for c in r.cycles:
            self.assertAlmostEqual(c.loglik,estep(self.data,c.params,c.weights,r.nodes,r.config.D)[2].sum())
            self.assertTrue(np.all(c.q_after>=c.q_before-1e-6))
        self.assertTrue(np.all(np.diff([c.loglik for c in r.cycles])>=-1e-6))

    def test_animation_matches_history_and_excludes_missing_person(self):
        d=deepcopy(self.data);d.responses[0,0]=np.nan
        r=fit(d,FitConfig(max_cycles=3));payload=animation_payload(d,r,0,0,0)
        for k,f in enumerate(payload['frames']):
            tau,_,_=replay_estep(d,r,k)
            np.testing.assert_allclose(f['posterior'],tau[0]);np.testing.assert_array_equal(f['person_N'],np.zeros(49))
            np.testing.assert_array_equal(f['output'],r.cycles[k].params[0]);self.assertEqual(f['cycle'],k+1)
        self.assertIn('Convergence playback',playback_html(payload));self.assertIn('setInterval',playback_html(payload))

    def test_unavailable_reference_never_passes(self):
        r=fit(self.data,FitConfig(max_cycles=2));report=compare_endpoints(self.data,r,dict(status='Unavailable',reason='test'))
        self.assertEqual(report['status'],'Unavailable');self.assertFalse(report['executed'])

    def test_positive_slope_domain_mismatch_is_review(self):
        r=fit(self.data,FitConfig(max_cycles=2,positive_a=True))
        ref=dict(status='Executed',params=r.params.copy(),weights=r.weights.copy(),reference={})
        report=compare_endpoints(self.data,r,ref);self.assertEqual(report['status'],'Review')
        self.assertTrue(any('domain' in s for s in report['differences']))

    def test_em_cycle_reproduces_recorded_cycle_and_stationarity(self):
        cfg=FitConfig(max_cycles=200);r=fit(self.data,cfg)
        # One cycle from a recorded input state reproduces that recorded output exactly.
        c=r.cycles[4];step=stationarity_step(self.data,cfg,c.input_params,c.input_weights,r.nodes)
        np.testing.assert_array_equal(step.params,c.params);np.testing.assert_array_equal(step.weights,c.weights)
        # A converged endpoint barely moves; the random-start state moves a lot.
        self.assertIn('stopping rule satisfied',r.stop_reason)
        self.assertLessEqual(stationarity_step(self.data,cfg,r.params,r.weights,r.nodes).item_change,STATIONARITY_FACTOR*cfg.tolerance)
        self.assertGreater(stationarity_step(self.data,cfg,r.initial_params,r.initial_weights,r.nodes).item_change,.01)

    def test_reference_policy_pass_fail_and_early_stop_review(self):
        d=simulate(400,10,seed=7);cfg=FitConfig();r=fit(d,cfg)
        same=dict(status='Executed',params=r.params.copy(),weights=r.weights.copy(),reference={})
        report=compare_endpoints(d,r,same)
        self.assertEqual(report['status'],'Pass',report['executed'])
        self.assertTrue(any('stationarity' in c['check'] for c in report['executed']))
        shifted=deepcopy(same);shifted['params'][~d.fixed,1]+=.1
        self.assertEqual(compare_endpoints(d,r,shifted)['status'],'Fail')
        # A reference that agrees within tolerance but was still moving cannot Pass.
        nudged=deepcopy(same);nudged['params'][~d.fixed,1]+=.015
        report=compare_endpoints(d,r,nudged)
        self.assertNotIn('Fail',[c['status'] for c in report['executed'] if 'stationarity' not in c['check']])
        self.assertEqual(report['status'],'Review');self.assertIn('still moving',report['reason'])
        # A real disagreement stays Fail even though that reference also looks unconverged.
        early=dict(status='Executed',params=r.cycles[2].params.copy(),weights=r.cycles[2].weights.copy(),reference={})
        self.assertEqual(compare_endpoints(d,r,early)['status'],'Fail')

    def test_unsettled_histogram_density_is_review_not_pass(self):
        d=simulate(500,12,[dict(name='G0',mean=0.,sd=1.)],common=8);r=fit(d,FitConfig(density='Histogram'))
        self.assertEqual(r.stop_reason,'Item stopping rule satisfied')  # converged items, unsettled density
        ref=dict(status='Executed',params=r.params.copy(),weights=r.weights.copy(),reference={})
        report=compare_endpoints(d,r,ref)
        self.assertGreater(report['stationarity']['python_density_change'],r.config.density_tolerance)
        self.assertEqual(report['status'],'Review');self.assertIn('Density not converged',report['reason'])
        # Coefficients stay strict even while the density is unsettled.
        bad=deepcopy(ref);bad['params'][~d.fixed,0]+=.2
        self.assertEqual(compare_endpoints(d,r,bad)['status'],'Fail')

    def test_reference_files_flag_anchors_as_integers_and_use_project_library(self):
        d=simulate(200,8,anchors=3);r=fit(d,FitConfig(design=DESIGNS[2],max_cycles=2))
        seen={}
        def fake_run(command,**kwargs):
            folder=Path(command[-1]).parent
            seen['items']=pd.read_csv(folder/'items.csv');seen['env']=kwargs.get('env',{})
            return subprocess.CompletedProcess(command,1,'','stopped by test')
        with mock.patch.object(mmle_reference,'check_irtq',return_value=dict(available=True,reason='test')), \
             mock.patch.object(mmle_reference.subprocess,'run',side_effect=fake_run):
            fit_irtq(d,r)
        # True/False text is read by R as character and silently unfixes every anchor.
        self.assertEqual(seen['items'].fixed.tolist(),d.fixed.astype(int).tolist())
        with tempfile.TemporaryDirectory() as library, mock.patch.dict(os.environ,{'MMLE_R_LIBRARY':library}):
            self.assertTrue(r_environment()['R_LIBS'].startswith(library))

    def test_truth_population_grid_weights_for_all_shapes(self):
        for shape in ['Normal','Skewed','Bimodal','Custom']:
            spec=dict(name='G1',mean=1.,sd=1.5,shape=shape,nodes=[-2.,0.,3.],weights=[.2,.5,.3])
            d=simulate(80,8,[spec]);r=fit(d,FitConfig(mean=.3,sd=.8,max_cycles=2));rec=recovery(d,r)
            tw=rec['truth_grid_weights']
            self.assertTrue(np.isfinite(tw).all());self.assertTrue((tw>=0).all())
            np.testing.assert_allclose(tw.sum(axis=1),1.)
            self.assertTrue(np.all((rec['density_TV']>=0)&(rec['density_TV']<=1)))

    def test_invalid_initial_masses_and_constant_anchors(self):
        with self.assertRaisesRegex(ValueError,'Custom initial masses'):
            FitConfig(initial_weights=[1.,2.]).validate()
        d=simulate(80,8,anchors=3);d.responses[:,0:3]=1.
        with self.assertRaisesRegex(ValueError,'constant'):
            check_design(d,FitConfig(design=DESIGNS[2]))

    @unittest.skipUnless(shutil.which('Rscript'),'R unavailable')
    def test_reference_source_histogram_rule(self):
        # Supplementary comparison to the reviewed source function, NOT a package
        # endpoint test and never promoted to the primary validation status.
        source=Path('/Users/jakecho/Documents/Codex/2026-11-01/realtime-voice-chat-2/work/irtQ-source/R')
        if not source.is_dir():self.skipTest('Reviewed source clone not present')
        rng=np.random.default_rng(10);x=np.linspace(-6,6,49)
        for _ in range(3):
            w=rng.uniform(.01,1,49)*normal_weights(x,rng.normal(),rng.uniform(.7,1.5));w/=w.sum()
            with tempfile.TemporaryDirectory() as folder:
                np.savetxt(Path(folder)/'w.csv',np.c_[x,w],delimiter=',')
                code=f'source({json.dumps(str(source/"util.R"))});source({json.dumps(str(source/"scale_prior.R"))});z<-read.csv({json.dumps(folder+"/w.csv")},header=FALSE);write.table(scale_prior(pmax(z[,2],1e-20),z[,2],z[,1],c(.3,1.2^2)),file={json.dumps(folder+"/out.csv")},row.names=FALSE,col.names=FALSE)'
                subprocess.run(['Rscript','-e',code],check=True,capture_output=True)
                reference=np.loadtxt(Path(folder)/'out.csv')
                np.testing.assert_allclose(rescale_histogram(x,w,.3,1.2),reference,rtol=1e-12,atol=1e-14)

    @unittest.skipUnless(check_irtq()['available'],'Installed irtQ unavailable; primary endpoint baselines not run')
    def test_installed_irtq_endpoint_baselines(self):
        for design in DESIGNS:
            for model in ['2PL','3PL']:
                for density in ['Normal','Histogram']:
                    mg=design in (DESIGNS[1],DESIGNS[3]);anchor=design in DESIGNS[2:]
                    d=simulate(500,12,[dict(name=f'G{g}',mean=g*.4,sd=1.) for g in range(2 if mg else 1)],model=model,common=8,anchors=5 if anchor else 0)
                    cfg=FitConfig(design=design,model=model,density=density,priors=Priors(c=model=='3PL'))
                    with self.subTest(design=design,model=model,density=density):
                        r=fit(d,cfg);ref=fit_irtq(d,r)
                        self.assertEqual(ref['status'],'Executed',ref)
                        report=compare_endpoints(d,r,ref)
                        # Coefficients, curves and the reference-stationarity guard never fail.
                        self.assertNotIn('Fail',[row['status'] for row in report['executed']],report['executed'])
                        if density=='Normal':
                            self.assertEqual(report['status'],'Pass',report['executed'])
                        else:
                            # Item-only stopping leaves histogram masses moving in both programs.
                            self.assertIn(report['status'],['Pass','Review'])
        # OEM: one item-fitting cycle in both programs, then endpoint agreement.
        d=simulate(500,12,anchors=5);r=fit(d,FitConfig(design=DESIGNS[2],schedule='OEM'))
        ref=fit_irtq(d,r);self.assertEqual(ref['status'],'Executed',ref)
        self.assertNotIn('Fail',[row['status'] for row in compare_endpoints(d,r,ref)['executed']])

if __name__=='__main__':unittest.main()
