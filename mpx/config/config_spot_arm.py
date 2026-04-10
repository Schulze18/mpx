import os
from functools import partial

import jax
import jax.numpy as jnp

import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives


dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, "..")) + "/data/boston_dynamics_spot/scene_arm.xml"

# Contact frame names and body names for the Spot feet / lower legs.
contact_frame = ["FL", "FR", "HL", "HR"]
body_name = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
# contact_frame = ["FL", "FR", "RL", "RR"]
# body_name = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

# Time and stage parameters.
dt = 0.02
N = 25
mpc_frequency = 50

# Gait parameters.
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = 0.7
step_freq = 1.0
step_height = 0.08
initial_height = 0.46
robot_height = 0.35

# Initial base state and nominal joint posture.
p0 = jnp.array([0.0, 0.0, initial_height])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([0, -3.14, 3.06, 0, 0, 0, 0, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8, 0.0, 1.04, -1.8])
q0_init = q0

# Nominal foot positions in the body frame at the home posture.
p_legs0 = jnp.array([
    0.34, 0.175, 0.0,
    0.34, -0.175, 0.0,
    -0.34, 0.175, 0.0,
    -0.34, -0.175, 0.0,
])

# Dimensions.
n_joints = 12 + 7
n_contact = len(contact_frame)
n = 13 + 2 * n_joints + 6 * n_contact
m = n_joints
grf_as_state = True

# Reference controls.
u_ref = jnp.zeros(m)

# Cost weights.
Qp = jnp.diag(jnp.array([0.0, 0.0, 1e4]))
Qrot = jnp.diag(jnp.array([1000.0, 1000.0, 0.0])) * 10
Qq = jnp.diag(jnp.ones(n_joints)) * 1e0
Qdp = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e3
Qomega = jnp.diag(jnp.array([1.0, 1.0, 1.0])) * 1e2
Qdq = jnp.diag(jnp.ones(n_joints)) * 1e-1
Qtau = jnp.diag(jnp.ones(n_joints)) * 1e-2
Q_grf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-3
Qleg = jnp.diag(jnp.tile(jnp.array([1e4, 1e4, 1e5]), n_contact))
W = jax.scipy.linalg.block_diag(Qp, Qrot, Qq, Qdp, Qomega, Qdq, Qleg, Qtau, Q_grf)

use_terrain_estimation = True

cost = partial(mpc_objectives.quadruped_wb_obj, True)
hessian_approx = partial(mpc_objectives.quadruped_wb_hessian_gn, True)
dynamics = mpc_dyn_model.quadruped_arm_wb_dynamics

# Torque bounds used by the MPC cost / clipping.
max_torque = 500
min_torque = -500