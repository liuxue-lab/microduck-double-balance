"""Run the actual pinned original arithmetic on CPU; GPU capture remains untested here."""
import ast
import dataclasses
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import bam_graph as bg
from model_limits import LimitStop
REAL_GRAPH_CLASS=bg.GraphedBam

torch.set_num_threads(1)
FIXTURES=Path(__file__).parent/'fixtures'


def original_classes():
    hashes={'bam-actuator.py':'6927d7b2341cbaca0e5d872fae4ddce6200056885310d9ee5acaba1f34e09c13',
        'bam-mjlab.py':'af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440',
        'friction-dr-bam.py':'75cb3e993cd8fdc9d90081129f793deb1eb22fc1ad921f75a97511fa98d38fb7'}
    namespace=dict(torch=torch,_FRICTION_DOF_CONSTRAINT=1)
    for name,classname,methods,base in (
        ('bam-actuator.py','VoltageControlledActuator',('compute_control','compute_torque'),'object'),
        ('bam-mjlab.py','BamActuator',('compute','_as_tensor','_dof_friction_force','_compute_friction_budget','_write_frictions'),'object'),
        ('friction-dr-bam.py','FrictionDRBamActuator',('compute','_compute_friction_budget'),'BamActuator')):
        data=(FIXTURES/name).read_bytes();assert hashlib.sha256(data).hexdigest()==hashes[name]
        node=next(n for n in ast.parse(data).body if isinstance(n,ast.ClassDef) and n.name==classname)
        node.bases=[ast.Name(id=base,ctx=ast.Load())];node.keywords=[];node.decorator_list=[]
        node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),name,'exec'),namespace)
    return namespace


CLASSES=original_classes()


@dataclasses.dataclass
class Cmd:
    pos:torch.Tensor
    vel:torch.Tensor
    position_target:torch.Tensor


def fixture(drop=True):
    g=CLASSES['FrictionDRBamActuator']();g.cfg=NS(vin_min=5.0)
    bam=NS(stribeck=True,load_dependent=True,directional=True,quadratic=True)
    vals=dict(kt=.36601349688984386,R=2.8113923539223227,friction_base=.04,
        friction_viscous=.02,friction_stribeck=.01,load_friction_external=.03,
        load_friction_motor=.04,load_friction_external_stribeck=.01,
        load_friction_motor_stribeck=.02,load_friction_external_quad=.003,
        load_friction_motor_quad=.004,dtheta_stribeck=.15,alpha=1.2)
    for n,v in vals.items():setattr(bam,n,NS(value=v))
    act=CLASSES['VoltageControlledActuator']();act.model=bam;act.backend=NS(clamp=torch.clamp)
    act.max_current=1.75;act.max_pwm=1.;act.error_gain=.00287737;act.kp=400.;act.vin=7.5
    bam.actuator=act;g._bam_model=bam;g._base_kp=400.;g._dt=.002;g._num_envs=1
    g._friction_fields_checked=False;g._device='cpu';g._dof_ids=torch.arange(2,16)
    g.vin_tensor=torch.full((1,1),7.5);g.vin_drop_gain=torch.full((1,1),.02) if drop else None
    g.kp_scale=torch.full((1,1),.96);g.kd_scale=torch.full((1,1),1.03)
    g.friction_scale=torch.full((1,1),.9);g._prev_motor_torque=torch.zeros(1,14)
    g.diagnostics_enabled=True;g.last_diagnostics=None
    g._mjwarp_model=NS(dof_frictionloss=torch.zeros(1,20),dof_damping=torch.zeros(1,20))
    g._data=NS(qfrc_bias=torch.zeros(1,20),qfrc_constraint=torch.zeros(1,20),
        qfrc_actuator=torch.zeros(1,20),nefc=torch.tensor([14]),
        efc=NS(type=torch.ones(1,18,dtype=torch.int32),id=torch.tensor([list(range(2,16))+[0]*4],dtype=torch.int32),
               force=torch.zeros(1,18)))
    return g,Cmd(torch.zeros(1,14),torch.zeros(1,14),torch.zeros(1,14))


class OriginalArithmeticTests(unittest.TestCase):
    def emulate_capture(self,g,cmd):
        """Exercise replay bookkeeping on CPU; this does not test CUDA capture."""
        runner=REAL_GRAPH_CLASS.__new__(REAL_GRAPH_CLASS)
        runner.group=g;runner.torch=torch;runner.config=bg.configuration(g)
        runner.fields=bg.live_fields(g)
        runner.signatures={k:bg.storage_signature(v) for k,v in runner.fields.items()}
        runner.command=dataclasses.replace(cmd,**{n:getattr(cmd,n).clone() for n in bg.COMMAND_FIELDS})
        runner.previous=g._prev_motor_torque.clone();runner.shadow=bg.Shadow(g);runner.reference=bg.Shadow(g)
        runner.calls=runner.checked_calls=runner.scalar_comparisons=0;runner.maximum_error=0.
        runner.damping=g._bam_model.friction_viscous.value
        def replay():runner.outputs=runner.shadow.run(runner.command,runner.previous)
        runner.graph=NS(replay=replay)
        return runner
    def test_replay_state_commit_and_diagnostic_freshness(self):
        g,c=fixture();reference,reference_cmd=fixture();runner=self.emulate_capture(g,c)
        last=None
        for i in range(160):
            c.position_target.fill_(.1 if i%2 else -.07);c.vel.fill_(.2 if i%3 else -.12)
            reference_cmd.position_target.copy_(c.position_target);reference_cmd.vel.copy_(c.vel)
            expected=reference.compute(reference_cmd);actual=runner(c)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(g._prev_motor_torque,reference._prev_motor_torque))
            self.assertTrue(torch.equal(g._mjwarp_model.dof_frictionloss,reference._mjwarp_model.dof_frictionloss))
            self.assertIsNot(last,g.last_diagnostics);last=g.last_diagnostics
            for key in last:self.assertTrue(torch.equal(last[key],reference.last_diagnostics[key]))
        self.assertEqual(runner.checked_calls,bg.SHADOW_CHECK_CALLS);self.assertEqual(runner.calls,160)
    def test_changed_storage_stops_before_live_output_write(self):
        g,c=fixture();runner=self.emulate_capture(g,c);g.kp_scale=g.kp_scale.clone()
        with self.assertRaisesRegex(RuntimeError,'backing storage'):runner(c)
        self.assertEqual(runner.calls,0);self.assertIsNone(g.last_diagnostics)
        self.assertFalse(g._mjwarp_model.dof_frictionloss.any())
    def test_shadow_has_no_live_writes_or_recurrent_state_mutation(self):
        g,c=fixture();shadow=bg.Shadow(g);prev=g._prev_motor_torque;act=g._bam_model.actuator
        c.position_target.fill_(.15);q=g._data.qfrc_bias.clone()
        result=shadow.run(c,prev)
        self.assertIs(g._prev_motor_torque,prev);self.assertIsNone(g.last_diagnostics)
        self.assertEqual(act.vin,7.5);self.assertEqual(act.kp,400.)
        self.assertFalse(g._mjwarp_model.dof_frictionloss.any())
        self.assertTrue(torch.equal(g._data.qfrc_bias,q))
        self.assertFalse(torch.equal(result['output_torque'],prev))
        self.assertIs(shadow.group._bam_model.actuator.model,shadow.group._bam_model)
    def test_original_math_and_isolated_shadow_match_across_dynamic_inputs(self):
        for drop in (False,True):
            g,c=fixture(drop);shadow=bg.Shadow(g);generator=torch.Generator().manual_seed(7801)
            for tick in range(160):
                c.pos.copy_(torch.rand((1,14),generator=generator)*.2-.1)
                c.vel.copy_(torch.rand((1,14),generator=generator)*.6-.3)
                c.position_target.copy_(c.pos+torch.rand((1,14),generator=generator)*.3-.15)
                for n in ('qfrc_bias','qfrc_constraint','qfrc_actuator'):
                    getattr(g._data,n).copy_(torch.rand((1,20),generator=generator)*.2-.1)
                g._data.nefc[0]=tick%15
                g._data.efc.force.copy_(torch.rand((1,18),generator=generator)*.04)
                candidate=shadow.run(c,g._prev_motor_torque)
                actual=type(g).compute(g,c)
                expected=dict(g.last_diagnostics,output_torque=actual,
                    frictionloss=g._mjwarp_model.dof_frictionloss[:,g._dof_ids],
                    actuator_vin=g._bam_model.actuator.vin,actuator_kp=g._bam_model.actuator.kp)
                worst,count=bg.check_outputs(torch,expected,candidate)
                self.assertEqual(worst,0.);self.assertGreater(count,100)
    def test_stale_constraint_rows_are_excluded_by_original_code(self):
        g,c=fixture();shadow=bg.Shadow(g);g._data.nefc.zero_()
        a=shadow.run(c,g._prev_motor_torque)['frictionloss'].clone()
        g._data.efc.force.fill_(1e6)
        b=shadow.run(c,g._prev_motor_torque)['frictionloss']
        self.assertTrue(torch.equal(a,b))
        g._data.nefc.fill_(14)
        self.assertFalse(torch.equal(a,shadow.run(c,g._prev_motor_torque)['frictionloss']))
    def test_same_input_gate_rejects_changed_or_nonfinite_output(self):
        base={'torque':torch.ones(1,14)}
        for value in (1.1,float('nan')):
            with self.assertRaises(RuntimeError):bg.check_outputs(torch,base,{'torque':torch.full((1,14),value)})
    def test_storage_and_configuration_changes_are_visible(self):
        g,c=fixture();a=bg.configuration(g);g._bam_model.kt.value*=1.1
        self.assertNotEqual(a,bg.configuration(g))
        old=bg.storage_signature(g.kp_scale);g.kp_scale=g.kp_scale.clone()
        self.assertNotEqual(old,bg.storage_signature(g.kp_scale))
    def test_bam_binding_restores_after_failure_and_is_idempotent(self):
        g,c=fixture()
        with patch.object(bg,'GraphedBam',side_effect=RuntimeError('capture failure')):
            graphs=bg.BamGraphs([g],torch)
            with self.assertRaises(RuntimeError):g.compute(c)
            graphs.close();graphs.close()
        self.assertNotIn('compute',vars(g));self.assertIsNone(g.last_diagnostics)


class BatchReadTests(unittest.TestCase):
    def setup_guard(self):
        data=NS(qpos=torch.tensor([[9.,.1,.2,8.]]),qvel=torch.tensor([[.3,9.,.4,8.]]),
            actuator_force=torch.tensor([[.5,.6,.7]]))
        rows=[];guard=NS(raw=NS(sim=NS(data=data)),qids=[2,1],vids=[2,0],ctrl_indices=[2,0],
            capture=lambda fn:fn(),audit=NS(after_integration=lambda *values:rows.append(values)))
        def original():
            guard.capture(lambda:guard.audit.after_integration(data.qpos[0,guard.qids].tolist(),
                data.qvel[0,guard.vids].tolist(),data.actuator_force[0,guard.ctrl_indices].tolist()))
        guard.after_integration=original
        return guard,rows,original
    def test_packed_reads_preserve_mapping_values_and_freshness(self):
        guard,rows,original=self.setup_guard();original();first=rows[-1]
        reads=bg.BatchedAuditReads(guard,torch)
        guard.after_integration();self.assertEqual(rows[-1],first)
        guard.raw.sim.data.qpos[0,2]=.9;guard.after_integration()
        self.assertAlmostEqual(rows[-1][0][0],.9,places=6)
        self.assertEqual(reads.calls,2);reads.close();self.assertIs(guard.after_integration,original)
    def test_audit_exception_propagates_and_binding_restores(self):
        guard,rows,original=self.setup_guard()
        def fail(*_):raise RuntimeError('model limit')
        guard.audit.after_integration=fail;reads=bg.BatchedAuditReads(guard,torch)
        with self.assertRaisesRegex(RuntimeError,'model limit'):guard.after_integration()
        self.assertEqual(reads.calls,0);reads.close();self.assertIs(guard.after_integration,original)


if __name__=='__main__':unittest.main()
