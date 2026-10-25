"""Compare actual native solves, including state precision and writable views."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from mjlab.sim.sim import SimulationCfg
from native_backend import NativeSimulation, NativeSensorContext

XML = '''<mujoco><option timestep=".005"/><worldbody>
<geom type="plane" size="2 2 .1"/><body pos="0 0 .8"><freejoint/>
<geom type="sphere" size=".1" mass="1"/><site name="s" pos=".1 0 0"/>
</body></worldbody><sensor><framepos objtype="site" objname="s"/></sensor></mujoco>'''


class NativeBackend(unittest.TestCase):
    def sim(self):
        cfg=SimulationCfg();cfg.mujoco.timestep=.005
        return NativeSimulation(1,cfg,mujoco.MjModel.from_xml_string(XML),'cpu')

    def test_float64_integration_matches_direct_mujoco_with_float32_views(self):
        sim=self.sim();model=deepcopy(sim.native_model);data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        # Merely reading qpos must never round the native solver state back to
        # float32. Exercise inference_mode, where tensor version counters vanish.
        with torch.inference_mode(), patch('mujoco_warp.step',side_effect=AssertionError('Warp solver used')), \
             patch('mujoco_warp.forward',side_effect=AssertionError('Warp forward used')):
            sensor_view=sim.data.sensordata[:, :3]
            for i in range(200):
                force=np.float32(.3*np.sin(i*.13))
                sim.data.xfrc_applied[0,1,0]=float(force)
                data.xfrc_applied[1,0]=float(force)
                _=sim.data.qpos,sim.data.qvel,sim.data.qacc,sim.data.site_xmat
                sim.step();mujoco.mj_step(model,data)
                np.testing.assert_array_equal(sim.mj_data.qpos,data.qpos)
                np.testing.assert_array_equal(sim.mj_data.qvel,data.qvel)
                if i%4==3:
                    sim.forward();mujoco.mj_forward(model,data)
                    np.testing.assert_allclose(sensor_view.numpy()[0],data.sensordata,atol=5e-8,rtol=0)
        self.assertEqual(sim.snapshot()['native_step_calls'],200)

    def test_reset_writes_and_randomization_recompute_preserve_nominal_defaults(self):
        sim=self.sim();original=sim.get_default_field('body_mass').clone()
        sim.expand_model_fields(('body_mass','body_inertia'))
        sim.model.body_mass[0,1]*=1.1
        sim.model.body_inertia[0,1]*=1.1
        sim.data.qpos[0,0]=.25
        sim.data.qvel[0,0]=.125
        sim.recompute_constants(NS(name='set_const'))
        sim.forward()
        self.assertEqual(sim.mj_data.qpos[0],.25)
        self.assertEqual(sim.mj_data.qvel[0],.125)
        self.assertTrue(torch.equal(original,sim.get_default_field('body_mass')))
        self.assertAlmostEqual(sim.native_model.body_mass[1],1.1,places=6)
        self.assertEqual(sim.mj_model.body_mass[1],1.)
        sim.reset(torch.tensor([0]));self.assertEqual(sim.data.qpos[0,0],0.)

    def test_nonfinite_input_preserved_instead_of_resetting(self):
        sim=self.sim();sim.data.qvel[0,0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'Nonfinite native input'):
            sim.step()
        self.assertEqual(sim.step_calls,0)
        self.assertTrue(torch.isnan(sim.data.qvel[0,0]))

    def test_native_terrain_rays_keep_original_sensor_reduction(self):
        from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
        sim=self.sim()
        cfg=TerrainHeightSensorCfg(name='foot_height_scan',frame=ObjRef(type='site',name='s'),
            pattern=RingPatternCfg.single_ring(radius=.04,num_samples=2),
            ray_alignment='world',max_distance=1.,include_geom_groups=(0,),exclude_parent_body=True)
        sensor=cfg.build();sensor.initialize(sim.mj_model,sim.model,sim.data,'cpu')
        ctx=NativeSensorContext(sim.mj_model,sim.model,sim.data,[],[sensor],'cpu')
        sim.set_sensor_context(ctx);sim.forward();sim.sense()
        np.testing.assert_allclose(sensor.data.heights.numpy(),[[.8]],atol=1e-7)
        self.assertEqual(sensor.num_rays,3)
        self.assertEqual(sim.sense_calls,1)


if __name__=='__main__':unittest.main()
