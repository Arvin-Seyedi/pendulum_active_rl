"""Source/graph, mocked rejection logging and dispatch checks for v2."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest,tempfile
import numpy as np
from src import cstr_expert_pilot_v2 as p
from src import cstr_verification as v
from src.cstr_pilot_preflight import zero_operations

ROOT=Path(__file__).resolve().parents[1]

class CSTRPilotV2Tests(unittest.TestCase):
    def test_only_config_difference(self):
        a=p.read(ROOT/'configs/cstr_expert_pilot_v1.json');b=p.read(ROOT/p.CONFIG)
        self.assertEqual(b.pop('controller_extra_ipopt_options'),{'bound_relax_factor':0.})
        b['version']=a['version'];self.assertEqual(a,b)

    def test_setup_changes_only_one_option(self):
        with zero_operations():
            m,mpc,_,_=p.native_templates(p.read(ROOT/p.CONFIG));ctrl=mpc(m(),silence_solver=True)
            self.assertEqual(ctrl.settings.nlpsol_opts,{'ipopt.print_level':0,'ipopt.sb':'yes','print_time':0,'ipopt.bound_relax_factor':0.})

    def test_invalid_returned_action_logged_before_raw_rejection(self):
        command=np.array([np.nextafter(5.,-np.inf),-.1]);ctrl=SimpleNamespace(make_step=lambda _:command)
        with tempfile.TemporaryDirectory() as t,patch.object(p.base,'memory',return_value={'previous_u':command}):
            observed=p.query(ctrl,[.8,.5,134.,130.],t,0,None)
            with self.assertRaises(AssertionError):v.require_input(observed)
            folder=Path(t)/'attempts/0000'
            self.assertTrue((folder/'postquery_memory.npz').exists());self.assertTrue((folder/'native_timing.json').exists())
            self.assertEqual(np.load(folder/'returned_command.npz')['command'].tobytes(),command.tobytes())

    def test_failed_native_query_keeps_available_memory_without_inventing_command(self):
        def fail(_):raise RuntimeError('injected failure')
        ctrl=SimpleNamespace(make_step=fail)
        with tempfile.TemporaryDirectory() as t,patch.object(p.base,'memory',return_value={'previous_u':np.array([5.,0.])}):
            with self.assertRaisesRegex(RuntimeError,'injected'):p.query(ctrl,[.8,.5,134.,130.],t,0,None)
            f=Path(t)/'attempts/0000'
            self.assertTrue((f/'post_attempt_memory.npz').exists());self.assertFalse((f/'returned_command.npz').exists())
            self.assertFalse(p.read(f/'native_timing.json')['native_query_returned'])

    def test_nonfinite_return_packet_is_preserved(self):
        command=np.array([np.nan,np.inf]);ctrl=SimpleNamespace(make_step=lambda _:command)
        with tempfile.TemporaryDirectory() as t,patch.object(p.base,'memory',return_value={'previous_u':command}):
            observed=p.query(ctrl,[.8,.5,134.,130.],t,0,None)
            with self.assertRaises(ValueError):v.require_input(observed)
            saved=np.load(Path(t)/'attempts/0000/returned_command.npz')['command']
            self.assertTrue(np.isnan(saved[0]));self.assertTrue(np.isinf(saved[1]))

    def test_effective_option_failure_blocks_backend(self):
        backend=SimpleNamespace();called=[]
        class Backend:
            def __call__(self,**kwargs):called.append(True)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'attempts/0000';f.mkdir(parents=True)
            owner=SimpleNamespace(folder=Path(t),index=1)
            observer=p.ObservedBackend(Backend(),owner)
            with patch.object(p,'inspect_options',return_value={**p.expected_options(),'bound_relax_factor':1e-8}):
                with self.assertRaisesRegex(AssertionError,'Effective'):observer()
            self.assertFalse(called);self.assertFalse(p.read(f/'effective_ipopt_options.json')['passed'])

    def test_both_sided_margins_recorded_without_clipping(self):
        with tempfile.TemporaryDirectory() as t:
            x=np.array([[.8,.5,140.+1e-8,130.]]);u=np.array([[5.-1e-8,0.]])
            p.record_margins(t,x,u);a=np.load(Path(t)/'raw_margins.npz')
            self.assertLess(a['state_upper'][0,2],0);self.assertLess(a['input_lower'][0,0],0)

    def test_same_frozen_job_order_and_budget(self):
        a=p.base.jobs(p.read(ROOT/'configs/cstr_expert_pilot_v1.json'));b=p.base.jobs(p.read(ROOT/p.CONFIG))
        self.assertEqual(a,b);self.assertEqual(len(b),40)

if __name__=='__main__':unittest.main()
