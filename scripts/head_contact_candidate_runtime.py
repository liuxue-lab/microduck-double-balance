"""Independent fixed-head candidate forward route; installed engine unchanged.

Only a guarded contact-frame write between native collision and constraint
construction differs from pinned native forward. No graph capture or warmup.
No task registration, external force, model-parameter edit or training.
"""
import numpy as np
import warp as wp
from mujoco_warp._src import forward as native
from mujoco_warp._src.forward import (smooth,collision_driver,constraint,island,sensor,solver,
 DisableBit,EnableBit,Model,Data,fwd_velocity,fwd_actuation,fwd_acceleration)

@wp.kernel
def _write_frame(frames:wp.array[wp.mat33],index:int,replacement:wp.array[wp.mat33]):
 frames[index]=replacement[0]


def replace_contact_frame(data,index,frame):
 replacement=wp.array(np.asarray(frame,np.float32).reshape(1,3,3),dtype=wp.mat33)
 wp.launch(_write_frame,dim=1,inputs=[data.contact.frame,index,replacement])
 wp.synchronize()

def fwd_position(m: Model, d: Data, factorize: bool = True, hook=None):
  """Position-dependent computations.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    factorize: Flag to factorize interia matrix.
  """
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    collision_driver.collision(m, d)
  if hook is not None:
    hook(m, d)
  constraint.make_constraint(m, d)
  # TODO(team): remove False after island features are more complete
  if False and not (m.opt.disableflags & DisableBit.ISLAND):
    island.island(m, d)
  smooth.transmission(m, d)

def forward(m: Model, d: Data, hook=None):
  """Forward dynamics."""
  energy = m.opt.enableflags & EnableBit.ENERGY

  fwd_position(m, d, factorize=False, hook=hook)
  d.sensordata.zero_()
  sensor.sensor_pos(m, d)
  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=True)

  solver.solve(m, d)
  sensor.sensor_acc(m, d)
