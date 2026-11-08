"""Static MPC80 preflight tests; no environment trajectories or real solves."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.quadrotor_mpc80_feasibility_v1 import (
    CONFIG, validate_cohort, require_distinct_evaluation, job_plan, reserve_job,
    finish_job, charge_overhead, durable_json, zero_operations)
from src.quadrotor_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]


class MPC80PreflightTests(unittest.TestCase):
    def config(self):return read(ROOT/CONFIG)

    def ledger(self,tmp):
        path=Path(tmp)/'ledger.json'
        durable_json(path,{'preflight_passed':True,'seal_sha256':'a'*64,
            'preflight_report_sha256':'b'*64,'halted':False,'jobs':job_plan(),
            'caps':self.config()['budget'],'reservations':[],
            'charged':{'solver_calls':0,'physics_steps':0,'cpu_seconds':0.,'wall_seconds':0.}})
        approval={'execution_approved':True,'seal_sha256':'a'*64,'preflight_report_sha256':'b'*64,
                  'review_reference':'synthetic unit fixture only, not an actual approval'}
        return path,approval

    def test_original_cohort_bytes_and_all_coordinates(self):
        a,r=validate_cohort(ROOT,self.config())
        self.assertTrue(r['passed']);self.assertEqual(a.shape,(16,6))
        self.assertEqual(r['coordinate_count'],96);self.assertFalse(r['authoritative_files_rewritten'])
        self.assertFalse(r['independent_learning_evaluation_defined'])

    def test_tampered_manifest_rejected(self):
        c=copy.deepcopy(self.config())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mp=root/c['cohort']['manifest'];ap=root/c['cohort']['array']
            mp.parent.mkdir(parents=True)
            mp.write_bytes((ROOT/c['cohort']['manifest']).read_bytes())
            ap.write_bytes((ROOT/c['cohort']['array']).read_bytes())
            m=read(mp);m['states'][0]['coordinates'][0]+=1e-6
            mp.write_text(json.dumps(m));c['cohort']['manifest_sha256']=sha(mp)
            with self.assertRaises(AssertionError):validate_cohort(root,c)

    def test_manifest_identity_required_before_loading(self):
        c=copy.deepcopy(self.config());c['cohort']['manifest_sha256']='0'*64
        with self.assertRaises(AssertionError):validate_cohort(ROOT,c)

    def test_fixed_protocol_denominators_and_failed_gates(self):
        c=self.config();self.assertFalse(c['execution_authorized'])
        self.assertEqual(c['original_overall_qualification'],'FAILED')
        self.assertEqual(c['original_model_agreement'],{'threshold':.001,'status':'FAILED'})
        self.assertEqual(c['horizon'],80);self.assertEqual(c['acceptance']['complete_states'],16)
        self.assertEqual(c['acceptance']['complete_executions'],32)
        self.assertEqual(c['acceptance']['negative_margin_tolerance'],0)

    def test_exact_plan_order_and_resource_totals(self):
        p=job_plan();self.assertEqual(len(p),96)
        self.assertEqual([j['kind'] for j in p[:3]],['episode','native_replay','motor_replay'])
        self.assertEqual(p[3]['phase'],'repeat');self.assertEqual(p[6]['state_id'],1)
        self.assertEqual(sum(j['solver_calls'] for j in p),9600)
        self.assertEqual(sum(j['physics_steps'] for j in p),576000)
        self.assertEqual(len({j['id'] for j in p}),96)

    def test_no_dispatch_reservation_without_separate_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);before=p.read_bytes()
            with self.assertRaises(PermissionError):reserve_job(p,job_plan()[0]['id'],None)
            self.assertEqual(p.read_bytes(),before)

    def test_approval_must_bind_seal_and_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);a['seal_sha256']='wrong'
            with self.assertRaises(PermissionError):reserve_job(p,job_plan()[0]['id'],a)
            self.assertEqual(read(p)['charged']['solver_calls'],0)

    def test_receipt_is_persisted_before_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);r=reserve_job(p,job_plan()[0]['id'],a)
            self.assertEqual(read(p)['reservations'][-1],r)
            self.assertEqual(read(p)['charged']['solver_calls'],300)
            self.assertEqual(read(p)['charged']['physics_steps'],6000)

    def test_failed_work_consumes_reservation_and_halts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);key=job_plan()[0]['id'];reserve_job(p,key,a);finish_job(p,key,False)
            s=read(p);self.assertTrue(s['halted']);self.assertEqual(s['charged']['solver_calls'],300)
            with self.assertRaises(RuntimeError):reserve_job(p,key,a)
            with self.assertRaises(RuntimeError):reserve_job(p,job_plan()[1]['id'],a)

    def test_out_of_order_or_interrupted_dispatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp)
            with self.assertRaises(RuntimeError):reserve_job(p,job_plan()[1]['id'],a)
            reserve_job(p,job_plan()[0]['id'],a)
            with self.assertRaises(RuntimeError):reserve_job(p,job_plan()[1]['id'],a)

    def test_completed_replay_needs_own_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);reserve_job(p,job_plan()[0]['id'],a);finish_job(p,job_plan()[0]['id'],True)
            reserve_job(p,job_plan()[1]['id'],a)
            self.assertEqual(read(p)['charged']['solver_calls'],300)
            self.assertEqual(read(p)['charged']['physics_steps'],12000)

    def test_failed_persistence_yields_no_dispatch_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);before=p.read_bytes()
            with patch('src.quadrotor_mpc80_feasibility_v1.durable_json',side_effect=OSError('synthetic disk failure')):
                with self.assertRaises(OSError):reserve_job(p,job_plan()[0]['id'],a)
            self.assertEqual(before,p.read_bytes())

    def test_overhead_consumes_cpu_and_wall_budget_without_solves(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);charge_overhead(p,10,20)
            self.assertEqual(read(p)['charged']['cpu_seconds'],10)
            self.assertEqual(read(p)['charged']['solver_calls'],0)
            with self.assertRaises(RuntimeError):charge_overhead(p,1800,0)
            self.assertTrue(read(p)['halted'])

    def test_invalid_resource_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp)
            for cpu,wall in [(np.nan,1),(1,np.inf),(-1,0)]:
                with self.assertRaises(ValueError):charge_overhead(p,cpu,wall)

    def test_wall_overhead_cap_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp)
            with self.assertRaises(RuntimeError):charge_overhead(p,0,3601)
            self.assertTrue(read(p)['halted'])

    def test_job_cannot_clear_prior_resource_halt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);reserve_job(p,job_plan()[0]['id'],a)
            with self.assertRaises(RuntimeError):charge_overhead(p,1801,0)
            finish_job(p,job_plan()[0]['id'],True)
            self.assertTrue(read(p)['halted'])

    def test_solver_envelope_checked_before_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);s=read(p);s['caps']['solver_calls']=0;durable_json(p,s)
            with self.assertRaises(RuntimeError):reserve_job(p,job_plan()[0]['id'],a)
            self.assertEqual(read(p)['charged']['solver_calls'],0)

    def test_preflight_pass_required_even_with_approval_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,a=self.ledger(tmp);s=read(p);s['preflight_passed']=False;durable_json(p,s)
            with self.assertRaises(PermissionError):reserve_job(p,job_plan()[0]['id'],a)

    def test_qualification_coordinates_are_not_independent_policy_evaluation(self):
        q=np.arange(96,dtype=float).reshape(16,6)
        with self.assertRaises(ValueError):require_distinct_evaluation(q[:1],q)
        require_distinct_evaluation(q[:1]+.01,q)

    def test_runtime_guard_prevents_backend_and_physics_calls(self):
        import casadi as c
        import pybullet as b
        with zero_operations(controller_apis=False) as counts:
            with self.assertRaises(RuntimeError):c.nlpsol('not_a_solver','ipopt',{})
            with self.assertRaises(RuntimeError):b.stepSimulation()
        self.assertEqual(counts,{'nlpsol':1,'stepSimulation':1})
