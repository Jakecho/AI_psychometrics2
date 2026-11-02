"""Streamlit interaction, CSV/ZIP and playback JavaScript state tests."""
import unittest
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import numpy as np
from streamlit.testing.v1 import AppTest
from mmle_engine import simulate,fit,FitConfig,DESIGNS,signature
from mmle_data import export_experiment,load_csv_dataset
from mmle_animation import animation_payload,playback_html
from mmle_reference import check_irtq

APP=Path(__file__).with_name('MMLE_EM.py')


def choose(items,label):
    return next(item for item in items if item.label==label)


class AppTests(unittest.TestCase):
    def create_app(self):
        return AppTest.from_file(str(APP),default_timeout=45).run()

    def assert_no_errors(self,app):
        self.assertEqual([e.message for e in app.exception],[])

    def test_initial_six_screens_and_not_validated(self):
        at=self.create_app();self.assert_no_errors(at)
        self.assertTrue(set(['Design','EM walkthrough','Results','Validation','Compare','Method notes'])<={tab.label for tab in at.tabs})
        self.assertIn('Not validated',at.info[0].value)
        self.assertIsNone(at.session_state['run'])

    def test_one_cycle_resume_and_no_regeneration(self):
        at=self.create_app();data_id=at.session_state['dataset'].identity
        choose(at.button,'Run one cycle').click().run();self.assert_no_errors(at)
        self.assertEqual(len(at.session_state['run'].cycles),1)
        first=at.session_state['run'].cycles[0].params.copy()
        choose(at.button,'Run one cycle').click().run();self.assert_no_errors(at)
        self.assertEqual(len(at.session_state['run'].cycles),2)
        np.testing.assert_array_equal(at.session_state['run'].cycles[1].input_params,first)
        self.assertEqual(at.session_state['dataset'].identity,data_id)

    def test_display_and_pending_fit_changes_do_not_regenerate(self):
        at=self.create_app();choose(at.button,'Run one cycle').click().run()
        data_id=at.session_state['dataset'].identity;run_id=at.session_state['run'].run_id
        choose(at.radio,'Learning view').set_value('Explore').run();self.assert_no_errors(at)
        self.assertEqual(at.session_state['run'].run_id,run_id)
        choose(at.selectbox,'Population ability distribution').set_value('Histogram').run();self.assert_no_errors(at)
        self.assertEqual(at.session_state['dataset'].identity,data_id);self.assertEqual(at.session_state['run'].run_id,run_id)
        self.assertTrue(any('Pending fitting settings' in w.value for w in at.warning))

    def test_generate_is_explicit_and_seed_is_reproducible(self):
        at=self.create_app();old=at.session_state['dataset'].identity
        choose(at.number_input,'Generation seed').set_value(300).run();self.assert_no_errors(at)
        self.assertEqual(at.session_state['dataset'].identity,old)
        choose(at.button,'Generate / Regenerate responses').click().run();self.assert_no_errors(at)
        new=at.session_state['dataset'].identity;self.assertNotEqual(old,new)
        choose(at.button,'Generate / Regenerate responses').click().run();self.assert_no_errors(at)
        self.assertEqual(at.session_state['dataset'].identity,new)

    def test_default_run_internal_checks_and_reference_status(self):
        at=self.create_app();choose(at.button,'Run calibration').click().run();self.assert_no_errors(at)
        self.assertIn('stopping rule satisfied',at.session_state['run'].stop_reason)
        choose(at.button,'Run constraints, arithmetic and animation checks').click().run();self.assert_no_errors(at)
        self.assertTrue(all(r['status']=='Pass' for r in at.session_state['invariants'][1]))
        choose(at.button,'Validate against irtQ').click().run();self.assert_no_errors(at)
        status=at.session_state['validation']['status']
        # Default run: 2PL, normal population. With irtQ installed it must agree; without it, Unavailable.
        self.assertEqual(status,'Pass' if check_irtq()['available'] else 'Unavailable',at.session_state['validation'].get('executed'))

    def test_immutable_compare_and_prior_change(self):
        at=self.create_app();choose(at.button,'Run calibration').click().run()
        choose(at.button,'Save current experiment for comparison').click().run()
        saved=at.session_state['experiments'][0]['result'].params.copy()
        choose(at.selectbox,'Prior preset').set_value('Custom').run()
        choose(at.checkbox,'Enable a prior').check().run()
        choose(at.button,'Run calibration').click().run();self.assert_no_errors(at)
        np.testing.assert_array_equal(at.session_state['experiments'][0]['result'].params,saved)
        self.assertFalse(np.array_equal(at.session_state['run'].params,saved))
        choose(at.button,'Save current experiment for comparison').click().run();self.assert_no_errors(at)
        self.assertEqual(len(at.session_state['experiments']),2)

    def test_multigroup_bank_and_3pl_ui(self):
        at=self.create_app()
        choose(at.selectbox,'Calibration design').set_value(DESIGNS[3]).run()
        choose(at.selectbox,'Fitted IRT model').set_value('3PL').run()
        choose(at.selectbox,'Generating model').set_value('3PL').run()
        choose(at.number_input,'People per group').set_value(100).run()
        choose(at.number_input,'Maximum cycles').set_value(5).run()
        choose(at.selectbox,'Prior preset').set_value('irtQ defaults').run()
        choose(at.button,'Generate / Regenerate responses').click().run();self.assert_no_errors(at)
        d=at.session_state['dataset'];self.assertEqual(len(d.group_names),2);self.assertTrue(d.fixed.any())
        choose(at.button,'Run calibration').click().run();self.assert_no_errors(at)
        self.assertTrue(at.session_state['run'].config.priors.c)
        np.testing.assert_array_equal(at.session_state['run'].params[d.fixed],d.bank[d.fixed])

    def test_replications_and_sensitivity_keep_original_data(self):
        at=self.create_app()
        choose(at.number_input,'People per group').set_value(100).run()
        choose(at.number_input,'Maximum cycles').set_value(10).run()
        choose(at.button,'Generate / Regenerate responses').click().run()
        choose(at.button,'Run calibration').click().run()
        identity=at.session_state['dataset'].identity
        choose(at.number_input,'Recovery replications (additional generated datasets)').set_value(2).run()
        choose(at.button,'Run repeated-simulation recovery').click().run();self.assert_no_errors(at)
        self.assertEqual(len(at.session_state['replicated'][1]['populations']),2)
        choose(at.button,'Run refined-grid and changed-start sensitivity').click().run();self.assert_no_errors(at)
        self.assertEqual(len(at.session_state['sensitivity'][1]),2)
        self.assertEqual(at.session_state['dataset'].identity,identity)


class DataAndPlaybackTests(unittest.TestCase):
    def test_export_roundtrip_preserves_missing_and_absence(self):
        d=simulate(100,8,[dict(name='G1',mean=0,sd=1),dict(name='G2',mean=.5,sd=1)],common=4,anchors=3,missing=.2)
        r=fit(d,FitConfig(design=DESIGNS[3],max_cycles=2))
        blob=export_experiment(d,r,dict(status='Unavailable',executed=[],pending=['irtQ']))
        with zipfile.ZipFile(BytesIO(blob)) as z:
            loaded=load_csv_dataset(z.read('responses.csv'),z.read('administration.csv'),z.read('bank.csv'))
            self.assertEqual(loaded.identity,d.identity)
            np.testing.assert_allclose(loaded.bank,d.bank,equal_nan=True)
            h=json.loads(z.read('history.json'));self.assertEqual(len(h['cycles']),2)
            self.assertEqual(json.loads(z.read('validation.json'))['status'],'Unavailable')
            self.assertIn('versions',json.loads(z.read('settings.json')))

    def test_upload_conflicting_banks_rejected(self):
        y=b'group,I1,I2\nG1,1,0\nG1,0,1\n'
        bank=b'id,a,b,c\nI1,1,0,0\nI1,1,1,0\n'
        with self.assertRaisesRegex(ValueError,'Conflicting'):load_csv_dataset(y,bank_bytes=bank)

    def test_upload_absent_response_and_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError,'unadministered'):
            load_csv_dataset(b'I1,I2\n1,0\n0,1\n',b'I1,I2\n0,1\n1,1\n')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            load_csv_dataset(b'I1,I1\n1,0\n0,1\n')

    @unittest.skipUnless(shutil.which('node'),'Node unavailable')
    def test_javascript_play_pause_restart_scrub_modes_and_ranges(self):
        d=simulate(80,8);r=fit(d,FitConfig(max_cycles=4));html=playback_html(animation_payload(d,r))
        # Execute the actual shipped JS with a small DOM test harness. This tests
        # event/state logic without pretending that browser layout was inspected.
        script=html.split('<script>')[1].split('</script>')[0]
        harness=r'''
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{id,value:'',checked:false,innerHTML:'',textContent:'',className:'',disabled:false,setAttribute(k,v){this[k]=v}});return elements.get(id)}
let tick=null;const context={document:{getElementById:element},window:{matchMedia:()=>({matches:false})},setInterval:(fn,ms)=>{tick=fn;return 1},clearInterval:()=>{tick=null},console};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
assert.equal(element('counter').textContent,'1 / 4');assert.match(element('phaseTitle').textContent,/Likelihood/);
element('next').onclick();assert.equal(element('phase').value,1);
element('play').onclick();assert.equal(typeof tick,'function');tick();assert.equal(element('phase').value,2);element('play').onclick();assert.equal(tick,null);
element('cmode').onclick();assert.match(element('phaseTitle').textContent,/Convergence/);element('next').onclick();assert.equal(element('counter').textContent,'2 / 4');
element('cycle').value=3;element('cycle').oninput();assert.equal(element('counter').textContent,'3 / 4');
element('lo').value=2;element('hi').value=3;element('restart').onclick();assert.equal(element('counter').textContent,'2 / 4');
element('emode').onclick();assert.equal(element('phase').value,0);element('reduced').checked=true;element('play').onclick();assert.equal(tick,null);
assert.match(element('nodes').innerHTML,/Person N/);assert.match(element('params').innerHTML,/Coefficient/);
console.log('Playback JS event/state checks passed');
'''
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,'player.js').write_text(script);Path(folder,'harness.js').write_text(harness)
            proc=subprocess.run(['node',str(Path(folder,'harness.js')),str(Path(folder,'player.js'))],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)

if __name__=='__main__':unittest.main()
