"""Only audit time conversion changes: 12,000 native steps must mean 60 seconds."""
import unittest
from support import old
import native_model_limits as native
import model_limits as original

class ModelClock(unittest.TestCase):
    def test_snapshot_and_fault_timestamps_use_5ms_without_changing_thresholds(self):
        j=dict(name='j',lower_rad=-1.,upper_rad=1.,soft_lower_rad=-.8,soft_upper_rad=.8)
        a=native.Audit([j],[],joint_violation_handling='record_and_continue_simulation')
        a.observe_joints([0.],[0.],initial=True)
        a.completed_physics_steps=11999
        a.after_integration([1.1],[0.],[])
        row=a.snapshot()
        self.assertEqual(row['simulated_seconds'],60.)
        self.assertEqual(row['first_limit_event']['simulated_seconds'],60.)
        self.assertEqual(row['joints'][0]['first_violation_seconds'],60.)
        self.assertEqual(row['joints'][0]['sampled_limit_violation_seconds'],.005)
        self.assertAlmostEqual(row['joints'][0]['first_violation_context']['motor_sample_time_s'],59.995)
        self.assertEqual(original.PHYSICS_DT,.002)
        for values in ((1.,),(.03,-2.,4.)):
            self.assertEqual(native.tolerance(*values),original.tolerance(*values))
