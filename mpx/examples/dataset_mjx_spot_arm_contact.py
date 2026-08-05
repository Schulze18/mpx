import os
from timeit import default_timer as timer

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

import mpx.config.config_spot_arm as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils
import mpx.utils.rotation as rotation_utils
import mpx.utils.excitation_ref as exc_ref

import pickle

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


class Contact:
    def __init__(self, c):
        self.dist = c.dist
        self.geom1 = c.geom1
        self.geom2 = c.geom2
        self.pos = c.pos.copy()
        self.frame = c.frame.copy()

    @property
    def normal(self):
        return self.frame[0]
    
# Dataset Parameters
total_counter = 0
run_length_time = 10.0
n_runs = 50
dataset = {}
env_id = 0
run_id = -1
sim_frequency = 200.0
dataset_frequency = 100.0

gait_mode = 'walk' if jnp.sum(config.timer_t) > 1.0 else 'trot'

cmd_ref = False #'fourier'
base_ref_fixed = False
seed = 42 if cmd_ref == 'fourier' else 0
rand_step_freq = True

def _build_solve_fn(mpc):
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact):
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(jnp.asarray(qpos))
            .at[mpc.qvel_slice].set(jnp.asarray(qvel))
            .at[mpc.foot_slice].set(jnp.asarray(foot))
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc

def init_run_dataset():
    training_labels = [
        "labels", "t", 
        "q", "qd", "qdd",
        "tau_dyn", "tau_m", "tau_c", "tau_g",
        "tau_ext_total", "tau_act", "tau_contact",
        "tau_contact_feet", "contact_state", "contact_Jac",
        "contact_force", "contact_force_threshold",
        "command",
        "all_contact_state", "all_contact_force", "all_contact_Jac", "all_contact_Proj"
    ]
    dataset = {label: [] for label in training_labels}
    return dataset


def compute_constraint_torque_breakdown(mj_model, mj_data):
    mujoco.mj_forward(mj_model, mj_data)

    nefc = int(mj_data.nefc)
    nv = int(mj_model.nv)
    efc_force = np.asarray(mj_data.efc_force[:nefc], dtype=float)
    efc_type = np.asarray(mj_data.efc_type[:nefc], dtype=int)
    efc_id = np.asarray(mj_data.efc_id[:nefc], dtype=int)

    type_names = {
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS): "contact_frictionless",
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL): "contact_pyramidal",
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC): "contact_elliptic",
        int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT): "joint_limit",
        int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF): "friction_dof",
    }

    breakdown = []
    for idx in range(nefc):
        one_hot = np.zeros(nefc)
        one_hot[idx] = efc_force[idx]
        tau_i = np.zeros(nv)
        mujoco.mj_mulJacTVec(mj_model, mj_data, tau_i, one_hot)

        breakdown.append(
            {
                "index": idx,
                "type": int(efc_type[idx]),
                "type_name": type_names.get(int(efc_type[idx]), f"type_{int(efc_type[idx])}"),
                "id": int(efc_id[idx]),
                "force": float(efc_force[idx]),
                "tau": tau_i,
            }
        )

    tau_sum = np.sum([item["tau"] for item in breakdown], axis=0) if breakdown else np.zeros(nv)
    return breakdown, tau_sum



def get_feet_force(mj_model, mj_data, feet_geom_id, feet_body_id, sim_mj_data_contact):

    n_contacts = len(feet_geom_id)
    contact_state = [0] * n_contacts
    feet_contact_forces = [[] for _ in range(n_contacts)]

    full_wrench_contact = np.zeros(mj_model.nv)
    force_not_in_foot = np.zeros(3)
    contact_geom2_check = []
    jac_contact = []
    contact_pos = []

    for contact_id, contact in enumerate(sim_mj_data_contact):

        # Get body IDs from geom IDs
        geom1_id = contact.geom1
        geom2_id = contact.geom2

        if geom2_id in feet_geom_id:  # Check if contact occurs with the feet
            # if geom1_id != 0:
            #     print(f'self contact: g1 {geom1_id} g2 {geom2_id}')
            index = feet_geom_id.index(geom2_id)
            contact_state[index] += 1
            # Contact normal is R_c[:,0], that is the x-axis of the contact frame
            R_c = contact.frame.reshape(3, 3)
            force_c = np.zeros(6)  # 6D wrench vector
            mujoco.mj_contactForce(mj_model, mj_data, id=contact_id, result=force_c)
            # Transform the contact force to the world frame
            force_w = R_c.T @ force_c[:3]
            # print(f'force_c {force_c[3:]}')
            feet_contact_forces[index].append(force_w)

            jacp = np.zeros((3, mj_model.nv))  # linear part
            mujoco.mj_jac(mj_model, mj_data, jacp, None, contact.pos, feet_body_id[index])
            # print(f' pos {contact.pos}')
            full_wrench_contact += jacp.T @ force_w
            jac_contact.append(jacp)
            contact_pos.append(contact.pos)

        else:
            R_c = contact.frame.reshape(3, 3)
            force_c = np.zeros(6)  # 6D wrench vector
            mujoco.mj_contactForce(mj_model, mj_data, id=contact_id, result=force_c)
            force_not_in_foot += force_c[:3]
            contact_geom2_check.append([contact.geom1, contact.geom2])



    total_feet_contact_forces = [[]] * n_contacts
    list_forces = []
    for i in range(n_contacts):
        if contact_state[i]:
            total_feet_contact_forces[i] = np.sum(feet_contact_forces[i], axis=0)
            for f in feet_contact_forces[i]:
                list_forces.append(f)
        else:
            total_feet_contact_forces[i] = np.zeros(3)

    return np.array(contact_state), total_feet_contact_forces, list_forces, full_wrench_contact, jac_contact

def add_sim_data_to_dataset(dataset, time, sim_mj_data, sim_mj_model, feet_geom_id, feet_body_id, sim_mj_data_aux = None):
    # Joint
    q = np.copy(sim_mj_data.qpos[7:])

    # Base
    base_lin_pos = np.copy(sim_mj_data.qpos[:3])
    base_lin_vel = np.copy(sim_mj_data.qvel[:3])
    base_lin_acc = np.copy(sim_mj_data.qacc[:3])

    base_quat = np.roll((np.copy(sim_mj_data.qpos[3:7])),-1) # Get in xyzw order

    q_full = np.concatenate((base_lin_pos,
                             base_quat,
                             q,
                            ))
    qd_full = np.copy(sim_mj_data.qvel)
    qdd_full = np.copy(sim_mj_data.qacc)

    # Mujoco Simulation model
    M_full_mj_real = np.zeros((sim_mj_model.nv, sim_mj_model.nv))
    mujoco.mj_fullM(sim_mj_model, M_full_mj_real, sim_mj_data.qM)
    tau_m_mj_real = M_full_mj_real @ qdd_full
    tau_cg_mj_real = np.copy(sim_mj_data.qfrc_bias.reshape((sim_mj_model.nv,)))
    tau_mj_real = tau_m_mj_real + tau_cg_mj_real

    if sim_mj_data_aux is not None:
        sim_mj_data_aux.qpos = sim_mj_data.qpos.copy()
        sim_mj_data_aux.qvel[:] = 0
        mujoco.mj_forward(sim_mj_model, sim_mj_data_aux)
        tau_g_mj_real = np.copy(sim_mj_data_aux.qfrc_bias.reshape((sim_mj_model.nv,)))

        sim_mj_data_contact = [Contact(c) for c in sim_mj_data.contact]
    else:
        tau_g_mj_real = np.zeros(sim_mj_model.nv)

    tau_c_mj_real = tau_cg_mj_real - tau_g_mj_real
    tau_ctrl = np.copy(sim_mj_data.qfrc_actuator)


    contact_state, total_feet_contact_forces, feet_contact_forces, full_wrench_contact, jac_contact = get_feet_force(sim_mj_model, sim_mj_data, feet_geom_id, feet_body_id, sim_mj_data_contact)

    tau_contact_feet = np.zeros(sim_mj_model.nv)
    tau_contact_feet_radius = np.zeros(sim_mj_model.nv)
    for i, body_id in enumerate(feet_body_id):
        if contact_state[i] > 0:
            jacp_geom = np.zeros((3, sim_mj_model.nv))
            feet_pos = sim_mj_data_aux.geom_xpos[feet_geom_id[i]].copy()
            mujoco.mj_jac(sim_mj_model, sim_mj_data_aux, jacp_geom, None, feet_pos, body_id)
            tau_contact_feet += jacp_geom.T @ total_feet_contact_forces[i]

            mujoco.mj_jac(sim_mj_model, sim_mj_data_aux, jacp_geom, None, feet_pos-np.array([0,0.0,0.1]), body_id)
            tau_contact_feet_radius += jacp_geom.T @ total_feet_contact_forces[i]
            # print(f'feet {i} {feet_pos} force {total_feet_contact_forces[i]}')


    tau_contact_recompute = np.zeros(sim_mj_model.nv)
    for i in range(len(feet_contact_forces)):
        tau_contact_recompute += jac_contact[i].T @ feet_contact_forces[i]
    # print(feet_contact_forces)

    if len(jac_contact) > 0:
        Jac_all = np.concatenate(np.array(jac_contact), axis=0)
        Proj_i = np.eye(sim_mj_model.nv) - np.linalg.pinv(Jac_all) @ Jac_all
    else:
        Proj_i = np.eye(sim_mj_model.nv)


    # print(f'\nqfrc_constraint       : {sim_mj_data.qfrc_constraint}')

    # constraint_breakdown, tau_constraint_sum = compute_constraint_torque_breakdown(sim_mj_model, sim_mj_data)
    # print(f'tau_constraint_sum     : {tau_constraint_sum}')
    # for item in constraint_breakdown:
    #     tau_norm = np.linalg.norm(item["tau"])
    #     if abs(item["force"]) > 1e-8 or tau_norm > 1e-8:
    #         print(
    #             f"constraint {item['index']:3d} | {item['type_name']:<22s} | id {item['id']:3d} | force {item['force']:+.6f} | tau_norm {tau_norm:.6f}"
    #         )
    # print(f'full_wrench_contact    : {full_wrench_contact}')
    # print(f'tau_contact_feet       : {tau_contact_feet}')
    # print(f'tau_contact_feet_radius: {tau_contact_feet_radius}')
    # print(f'tau_contact_recompute: {tau_contact_recompute}')

    # print(f'\ndiff dyn ctrl               : {tau_mj_real - tau_ctrl}\n')
    # print(f'\ndiff check               : {tau_mj_real - sim_mj_data.qfrc_constraint - sim_mj_data.qfrc_passive - tau_ctrl}')
    # print(f'diff full wrench         : {tau_mj_real - full_wrench_contact - sim_mj_data.qfrc_passive - tau_ctrl}')
    # print(f'diff feet wrench         : {tau_mj_real - tau_contact_feet - sim_mj_data.qfrc_passive - tau_ctrl}')
    # print(f'diff feet wrench radius  : {tau_mj_real - tau_contact_feet_radius - sim_mj_data.qfrc_passive - tau_ctrl}')
    # print(f'diff feet recompute      : {tau_mj_real - tau_contact_recompute - sim_mj_data.qfrc_passive - tau_ctrl}')
    # print(f'diff constraint contact  : {sim_mj_data.qfrc_constraint - tau_contact_recompute}')
    tau_ext_total = tau_ctrl + tau_contact_recompute

    # Save to Dictionary
    dataset['t'].append(time)
    dataset['q'].append(q_full)
    dataset['qd'].append(qd_full)
    dataset['qdd'].append(qdd_full)

    dataset['tau_dyn'].append(tau_mj_real)
    dataset['tau_m'].append(tau_m_mj_real)
    dataset['tau_g'].append(tau_g_mj_real)
    dataset['tau_c'].append(tau_c_mj_real)

    dataset['tau_ext_total'].append(tau_ext_total)
    dataset['tau_act'].append(tau_ctrl)
    dataset['tau_contact'].append(tau_contact_recompute)
    dataset['tau_contact_feet'].append(tau_contact_feet)

    dataset['contact_state'].append(contact_state.astype(bool))
    dataset['contact_force'].append(np.array(total_feet_contact_forces).reshape(-1))

    dataset['all_contact_state'].append(contact_state)
    dataset['all_contact_force'].append(feet_contact_forces)
    dataset['all_contact_Jac'].append(jac_contact)
    dataset['all_contact_Proj'].append(Proj_i)

    return dataset, tau_mj_real

def sample_reset_references(key):
    key, key_ang, key_lin, key_amp, key_freq, key_init, key_z_offset, key_z_amp, key_z_freq, key_ang_offset, key_ang_amp, key_ang_freq, key_z_fourier = jax.random.split(key, 13)

    ref_base_ang_vel_lim = 0.4
    ref_base_ang_vel = jnp.array(
        [
            0.0,
            0.0,
            jax.random.uniform(
                key_ang,
                shape=(),
                minval=-ref_base_ang_vel_lim,
                maxval=ref_base_ang_vel_lim,
            ),
        ]
    )

    ref_base_lin_vel_lim = jnp.array([0.5, 0.2, 0.0])
    ref_base_lin_vel = jax.random.uniform(
        key_lin,
        shape=(3,),
        minval=-ref_base_lin_vel_lim,
        maxval=ref_base_lin_vel_lim,
    )

    amp_nom = config.extra_qref_data["amp"]
    freq_nom = config.extra_qref_data["freq"]
    joint_index = config.extra_qref_data["joint_index"]

    # arm_range = jnp.array([1.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5])
    arm_range = jnp.array([2.0, 1.0, 0.6, 2.0, 1.0, 1.2, 0])
    # arm_range = jnp.array([0*1.5, 0*1.0, 1.8, 0*2.0, 0*1.0, 0*1.2, 0])
    freq_range = jnp.array([0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    amp_ref = jax.random.uniform(
        key_amp,
        shape=amp_nom.shape,
        minval=-arm_range,
        maxval=arm_range,
    )
    freq_ref = jax.random.uniform(
        key_freq,
        shape=freq_nom.shape,
        minval=0.2 * jnp.ones_like(freq_range),
        maxval=1.0 * freq_range,
    )


    delta_z_offset = jax.random.uniform(key_z_offset, shape=(), minval=-0.04, maxval=0.04)
    z_offset = config.robot_height + delta_z_offset

    z_amp = jax.random.uniform(key_z_amp, shape=(), minval=0.02, maxval=0.15)
    z_freq = jax.random.uniform(key_z_freq, shape=(), minval=0.1, maxval=0.3)

    ang_offset_range = jnp.array([0.1, 0.1, 0.0])
    ang_amp_range = jnp.array([0.4, 0.8, 0.0])
    ang_freq_range = jnp.array([0.6, 1.0, 0.0])
    ang_offset = jax.random.uniform(key_ang_offset, shape=ang_offset_range.shape, minval=-ang_offset_range, maxval=ang_offset_range)
    ang_amp = jax.random.uniform(key_ang_amp, shape=ang_amp_range.shape, minval=-ang_amp_range, maxval=ang_amp_range)
    ang_freq = jax.random.uniform(key_ang_freq, shape=ang_freq_range.shape, minval=0.1 * jnp.ones_like(ang_freq_range), maxval=ang_freq_range)
    # No Yaw motion
    # ang_offset[2] = 0.0
    # ang_amp[2] = 0.0
    # ang_freq[2] = 0.0

    extra_qref_data = {
        "amp": amp_ref,
        "freq": freq_ref,
        "joint_index": joint_index,
        "z_offset": z_offset,
        "z_amp": z_amp,
        "z_freq": z_freq,
        "ang_offset": ang_offset,
        "ang_amp": ang_amp,
        "ang_freq": ang_freq
    }

    q_init = jnp.concatenate([config.p0, config.quat0, config.q0])
    q_arm_delta = jax.random.uniform(
        key_init,
        shape=joint_index.shape,
        minval=-0.3,
        maxval=0.3,
    )
    q_arm_delta = q_arm_delta.at[-1].set(0.0)
    q_init = q_init.at[7 + joint_index].add(q_arm_delta)

    # print(f'q_init {q_init}')

    command = jnp.array(
        [
            ref_base_lin_vel[0],
            ref_base_lin_vel[1],
            ref_base_lin_vel[2],
            ref_base_ang_vel[0],
            ref_base_ang_vel[1],
            ref_base_ang_vel[2],
            config.robot_height,
        ]
    )

    # Generate param fourier traj
    z_fourier_params = exc_ref.generate_random_param(order=5, njoints=1, param_range=jnp.array([0.5]), key=key_z_fourier)
    # append duration and initial offset as separate entries so params = [A, B, phi0, duration, q0]
    z_fourier_params.append(10.0)
    z_fourier_params.append(z_offset)
    # print(f'z_fourier_params length {len(z_fourier_params)}')
    extra_qref_data.update({"z_fourier_params": z_fourier_params})

    if hasattr(config, "reference_generator"):
        command = jnp.concatenate([command, jnp.array([1.0, 0.0, 0.0, 0.0])])

    return key, q_init, command, extra_qref_data


def sample_fourier_reset_references(key):
    # key, key_ang, key_lin, key_amp, key_freq, key_init, key_z_offset, key_z_amp, key_z_freq, key_ang_offset, key_ang_amp, key_ang_freq, key_z_fourier, key_xy_fourier, key_yaw_fourier = jax.random.split(key, 15)
    # key, key_ang, key_lin, key_amp, key_freq, key_init = jax.random.split(key, 15)
    key, key_ang, key_lin, key_init = jax.random.split(key, 4)


    ref_base_ang_vel_lim = 0.4
    ref_base_ang_vel = jnp.array(
        [
            0.0,
            0.0,
            jax.random.uniform(
                key_ang,
                shape=(),
                minval=-ref_base_ang_vel_lim,
                maxval=ref_base_ang_vel_lim,
            ),
        ]
    )

    if gait_mode == 'walk':
        ref_base_lin_vel_lim = jnp.array([0.4, 0.2, 0.0])
    else:
        ref_base_lin_vel_lim = jnp.array([0.5, 0.2, 0.0])
    ref_base_lin_vel = jax.random.uniform(
        key_lin,
        shape=(3,),
        minval=-ref_base_lin_vel_lim,
        maxval=ref_base_lin_vel_lim,
    )

    amp_nom = config.extra_qref_data["amp"]
    freq_nom = config.extra_qref_data["freq"]
    joint_index = config.extra_qref_data["joint_index"]

    # arm_range = jnp.array([1.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5])
    arm_range = jnp.array([2.0, 1.0, 0.6, 2.0, 1.0, 1.2, 0])
    # arm_range = jnp.array([0*1.5, 0*1.0, 1.8, 0*2.0, 0*1.0, 0*1.2, 0])
    freq_range = jnp.array([0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    # amp_ref = jax.random.uniform(
    #     key_amp,
    #     shape=amp_nom.shape,
    #     minval=-arm_range,
    #     maxval=arm_range,
    # )
    # freq_ref = jax.random.uniform(
    #     key_freq,
    #     shape=freq_nom.shape,
    #     minval=0.2 * jnp.ones_like(freq_range),
    #     maxval=1.0 * freq_range,
    # )

    key, key_z_old, key_z_old_offset, key_z_old_freq, key_ang_amp_old, key_ang_freq_old, key_ang_offset_old = jax.random.split(key, 7)

    delta_z_offset = jax.random.uniform(key_z_old_offset, shape=(), minval=-0.04, maxval=0.04)
    z_offset = config.robot_height + delta_z_offset

    z_amp = jax.random.uniform(key_z_old, shape=(), minval=0.02, maxval=0.15)
    z_freq = jax.random.uniform(key_z_old_freq, shape=(), minval=0.1, maxval=0.3)

    ang_offset_range = jnp.array([0.1, 0.1, 0.0])
    ang_amp_range = jnp.array([0.4, 0.8, 0.0])
    ang_freq_range = jnp.array([0.6, 1.0, 0.0])
    ang_offset = jax.random.uniform(key_ang_offset_old, shape=ang_offset_range.shape, minval=-ang_offset_range, maxval=ang_offset_range)
    ang_amp = jax.random.uniform(key_ang_amp_old, shape=ang_amp_range.shape, minval=-ang_amp_range, maxval=ang_amp_range)
    ang_freq = jax.random.uniform(key_ang_freq_old, shape=ang_freq_range.shape, minval=0.1 * jnp.ones_like(ang_freq_range), maxval=ang_freq_range)
    # No Yaw motion
    # ang_offset[2] = 0.0
    # ang_amp[2] = 0.0
    # ang_freq[2] = 0.0


    # q_init = jnp.concatenate([config.p0, config.quat0, config.q0])
    # q_arm_delta = jax.random.uniform(
    #     key_init,
    #     shape=joint_index.shape,
    #     minval=-0.3,
    #     maxval=0.3,
    # )
    # q_arm_delta = q_arm_delta.at[-1].set(0.0)
    # q_init = q_init.at[7 + joint_index].add(q_arm_delta)

    command = jnp.array(
        [
            ref_base_lin_vel[0],
            ref_base_lin_vel[1],
            ref_base_lin_vel[2],
            ref_base_ang_vel[0],
            ref_base_ang_vel[1],
            ref_base_ang_vel[2],
            config.robot_height,
        ]
    )
    q_init = jnp.concatenate([config.p0, config.quat0, config.q0])
    q_arm_delta = jax.random.uniform(
        key_init,
        shape=joint_index.shape,
        minval=jnp.array([-jnp.pi, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]),
        maxval=jnp.array([jnp.pi, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    )
    q_arm_delta = q_arm_delta.at[-1].set(0.0)
    q_init = q_init.at[7 + joint_index].add(q_arm_delta)
    print(f'q_init {q_init}')

    # Generate param fourier traj - orientation
    key, key_ang_freq, key_ang_fourier, key_ang_offset = jax.random.split(key, 4)

    ang_order = 3
    ang_duration_max = jnp.array([10.0, 10.0])
    ang_duration = jax.random.uniform(key_ang_freq, shape=ang_duration_max.shape, minval=0.5 * ang_duration_max, maxval=ang_duration_max)
    ang_fourier_params = exc_ref.generate_random_param(order=ang_order, njoints=2, param_range=jnp.array([0.6, 1.4]), key=key_ang_fourier)
    ang_offset = jax.random.uniform(key_ang_offset, shape=ang_offset_range.shape, minval=-ang_offset_range, maxval=ang_offset_range)
    ang_fourier_params.append(ang_duration)
    ang_fourier_params.append(ang_offset[:2]) # only roll and pitch
    # ang_fourier_params.append(ang_order)

    # Generate param fourier traj - z
    key, key_z_freq, key_z_fourier, key_z_offset = jax.random.split(key, 4)

    z_order = 3
    # z_duration = jax.random.uniform(key_ang_freq, minval=5.0, maxval=10.0)
    if cmd_ref == 'fourier':
        z_duration = jax.random.uniform(key_z_freq, minval=2.0, maxval=5.0) # for standing
    else:
        z_duration = jax.random.uniform(key_z_freq, minval=5.0, maxval=10.0)
    z_fourier_params = exc_ref.generate_random_param(order=z_order, njoints=1, param_range=jnp.array([0.3]), key=key_z_fourier)
    delta_z_offset = jax.random.uniform(key_z_offset, shape=(), minval=-0.04, maxval=0.04)
    z_offset = config.robot_height + delta_z_offset

    z_fourier_params.append(z_duration)
    z_fourier_params.append(z_offset)
    # z_fourier_params.append(z_order)
    # extra_qref_data.update({"z_fourier_params": z_fourier_params})

    # Generate param fourier traj - arm
    key, key_arm_freq, key_arm_fourier, key_arm_offset = jax.random.split(key, 4)
    arm_order = 3
    # arm_range = jnp.ones_like(jnp.array([2.0, 1.0, 0.6, 2.0, 1.0, 1.2, 0]))
    # arm_range = 5.0*jnp.array([2.0, 1.0, 0.6, 2.0, 1.0, 1.2, 0])
    # arm_range = 5.0*jnp.array([5.0, 1.0, 0.6, 2.0, 1.0, 1.2, 0])
    # arm_range = 5.0*jnp.array([2.0, 1.0, 0.8, 2.0, 1.0, 1.2, 0])
    arm_range = 2.5*jnp.array([2.0, 1.0, 0.8, 2.0, 1.0, 1.2, 0])
    arm_offset = jax.random.uniform(key_arm_offset, shape=arm_range.shape, minval=-2.0, maxval=2.0)
    # arm_offset = arm_offset.at[1:].set(0.0) # only first joint has offset
    # arm_offset = arm_offset + q_init[7 + joint_index]
    arm_offset = jnp.zeros_like(arm_range) # no offset for arm
    arm_offset = q_arm_delta
    arm_duration = jax.random.uniform(key_arm_freq, shape=arm_range.shape, minval=2.0, maxval=10.0)
    arm_fourier_params = exc_ref.generate_random_param(order=arm_order, njoints=arm_range.shape[-1], param_range=arm_range, key=key_arm_fourier)
    arm_fourier_params.append(arm_duration)
    arm_fourier_params.append(arm_offset)
    # arm_fourier_params.append(arm_order)

    print(f'\n\narm offset {arm_offset}\n q_init {q_init}\n')

    # Generate param fourier traj - xy
    key, key_xy_freq, key_xy_fourier, _ = jax.random.split(key, 4)
    xy_order = 3
    xy_duration = jax.random.uniform(key_xy_freq, minval=5.0, maxval=10.0)
    xy_range = 1.5*jnp.array([0.5, 0.3])
    xy_fourier_params = exc_ref.generate_random_param(order=xy_order, njoints=2, param_range=xy_range, key=key_xy_fourier)
    xy_fourier_params.append(xy_duration)
    xy_fourier_params.append(jnp.array([0.0, 0.0]))

    # Generate param fourier traj - yaw
    key, key_yaw_freq, key_yaw_fourier, key_yaw_offset = jax.random.split(key, 4)
    yaw_order = 3
    yaw_offset = jax.random.uniform(key_yaw_offset, minval=-0.1, maxval=0.1)
    yaw_duration = jax.random.uniform(key_yaw_freq, minval=5.0, maxval=10.0)
    yaw_fourier_params = exc_ref.generate_random_param(order=yaw_order, njoints=1, param_range=jnp.array([2*jnp.pi / 6]), key=key_yaw_fourier)
    yaw_fourier_params.append(yaw_duration)
    yaw_fourier_params.append(yaw_offset)

    # Step Frequency Sample
    key, key_step_freq = jax.random.split(key)
    if gait_mode == 'walk':
        step_freq = jax.random.uniform(key_step_freq, shape=(), minval=0.4, maxval=0.6)
    else:
        step_freq = jax.random.uniform(key_step_freq, shape=(), minval=0.9, maxval=1.5)

    extra_qref_data = {
        # "amp": amp_ref,
        # "freq": freq_ref,
        "joint_index": joint_index,
        "z_offset": z_offset,
        "z_amp": z_amp,
        "z_freq": z_freq,
        "ang_offset": ang_offset,
        "ang_amp": ang_amp,
        "ang_freq": ang_freq,
        "z_fourier_params": z_fourier_params,
        "ang_fourier_params": ang_fourier_params,
        "arm_fourier_params": arm_fourier_params,
        "xy_fourier_params": xy_fourier_params,
        "yaw_fourier_params": yaw_fourier_params,
        "step_freq": step_freq
    }

    if hasattr(config, "reference_generator"):
        command = jnp.concatenate([command, jnp.array([1.0, 0.0, 0.0, 0.0])])

    return key, q_init, command, extra_qref_data

@jax.jit
def update_z_command(command, counter, z_offset = 0.4, z_amp = 0.15, z_freq = 0.1):
    sin_time = counter / sim_frequency
    z_pos = z_offset + z_amp * jnp.sin(2 * jnp.pi * z_freq * sin_time)
    z_vel = 2 * jnp.pi * z_freq * z_amp * jnp.cos(2 * jnp.pi * z_freq * sin_time)
    command = command.at[6].set(z_pos)
    command = command.at[2].set(z_vel)
    return command

@jax.jit
def update_ang_command(command, counter, ang_offset = jnp.array([0.0, 0.0, 0.0]), ang_amp = jnp.array([0.0, 0.0, 0.0]), ang_freq = jnp.array([0.0, 0.0, 0.0])):
    sin_time = counter / sim_frequency
    euler_ref = ang_offset + ang_amp * jnp.sin(2 * jnp.pi * ang_freq * sin_time)
    quat_ref = rotation_utils.rpy_to_quat(euler_ref)
    command = command.at[-4:].set(quat_ref)
    return command

@jax.jit
def update_z_fourier(command, counter, z_fourier_params):
    sim_time = counter / sim_frequency
    # Generate time array from sin_time to t + N*Ts
    t_array = jnp.array([sim_time])
    z_pos, z_vel = exc_ref.generate_fourier_traj(t_array, params=z_fourier_params, return_vel = True)
    command = command.at[6].set(z_pos.reshape())
    command = command.at[2].set(z_vel.reshape())
    return command

@jax.jit
def update_ang_fourier(command, counter, ang_fourier_params):
    sim_time = counter / sim_frequency
    # Generate time array from sin_time to t + N*Ts
    t_array = jnp.array([sim_time])
    euler_ref = exc_ref.generate_fourier_traj(t_array, params=ang_fourier_params)
    # append 0 for yaw
    euler_ref = jnp.concatenate([euler_ref.reshape(-1), jnp.array([0.0])])
    quat_ref = rotation_utils.rpy_to_quat(euler_ref.reshape(-1))
    command = command.at[-4:].set(quat_ref)
    return command

@jax.jit
def update_xy_ang_fourier(command, counter, xy_fourier_params, ang_fourier_params, yaw_fourier_params):
    sim_time = counter / sim_frequency
    # Generate time array from sin_time to t + N*Ts
    t_array = jnp.array([sim_time])
    roll_pitch_ref = exc_ref.generate_fourier_traj(t_array, params=ang_fourier_params)
    yaw_ref = exc_ref.generate_fourier_traj(t_array, params=yaw_fourier_params)
    euler_ref = jnp.concatenate([roll_pitch_ref.reshape(-1), yaw_ref.reshape(-1)])
    # print(f'euler_ref {euler_ref.shape}')
    quat_ref = rotation_utils.rpy_to_quat(euler_ref.reshape(-1))
    command = command.at[-4:].set(quat_ref)
    # Zero ang vel in command
    command = command.at[3:6].set(0.0)
    
    _, xy_vel_ref = exc_ref.generate_fourier_traj(t_array, params=xy_fourier_params, return_vel = True)
    
    command = command.at[0].set(xy_vel_ref[0,0])
    command = command.at[1].set(xy_vel_ref[0,1])
    return command

def main():
    global total_counter, run_id

    model = mujoco.MjModel.from_xml_path(config.model_path)
    data = mujoco.MjData(model)
    data_aux = mujoco.MjData(model)
    model.opt.timestep = 1.0 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    feet_geom_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.contact_frame]
    feet_body_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in config.body_name]
    mpc = mpc_wrapper.MPCWrapper(config, limited_memory=True)
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = mpc.reset

    q_home = jnp.concatenate([config.p0, config.quat0, config.q0])
    data.qpos = q_home
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    rng_key = jax.random.PRNGKey(seed)
    period = int(sim_frequency / config.mpc_frequency)
    sample_period = int(sim_frequency / dataset_frequency)

    counter = 0
    tau = jnp.zeros(config.n_joints)
    command = jnp.zeros(7) if not hasattr(config, "reference_generator") else jnp.zeros(11)
    run_dataset = init_run_dataset()
    nan_flag = False
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if total_counter % int(run_length_time * sim_frequency) == 0 or total_counter == 0 or nan_flag:
                counter = 0
                if hasattr(config, "ref_type") and config.ref_type == "fourier":
                    rng_key, q_init, command, extra_qref_data = sample_fourier_reset_references(rng_key)
                    print(' SAMPLING FOURIER REFERENCES ')
                else:
                    rng_key, q_init, command, extra_qref_data = sample_reset_references(rng_key)
                print(f'Initial command {extra_qref_data}')

                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                data.qpos[:] = np.asarray(q_init)
                data.qvel[:] = np.zeros_like(data.qvel)
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)

                foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
                mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)
                mpc_data = mpc_data.replace(extra_qref_data=extra_qref_data)
                if rand_step_freq and ("step_freq" in extra_qref_data.keys()):
                    mpc_data = mpc_data.replace(step_freq=extra_qref_data["step_freq"])
                    print(f"New STEP FREQUENCY: {extra_qref_data['step_freq']}")

                if run_id >= 0:
                    for key in run_dataset.keys():
                        if key not in ["labels", "all_contact_Jac", "all_contact_force"]:
                            # run_dataset[key] = np.array(run_dataset[key])
                            try:
                                run_dataset[key] = np.array(run_dataset[key])
                            except Exception as e:
                                print(f"Error converting key: {key}")
                                raise

                if nan_flag is False:
                    env_id_array = env_id * np.ones(run_dataset["q"].shape[0]) if run_id >= 0 else np.ones(1)
                    if run_id > 0:
                        dataset["env_id"].append(env_id_array)
                        for key in run_dataset.keys():
                            dataset[key].append(run_dataset[key])
                    elif run_id == 0:
                        dataset["env_id"] = [env_id_array]
                        for key in run_dataset.keys():
                            dataset[key] = [run_dataset[key]]
                    run_id += 1
                else:
                    total_counter = run_id * int(run_length_time * sim_frequency)

                run_dataset = init_run_dataset()
                run_dataset["labels"] = f"env_{env_id}_run_{run_id}"
                nan_flag = False
                time_run_start = timer()

                print(f"Starting run: {run_dataset['labels']}")
                print(f"Sampled command: {command}")
                print(f"Sampled extra_qref_data: {extra_qref_data}")

            if total_counter == int(n_runs * run_length_time * sim_frequency):
                break

            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            current_time = counter / sim_frequency

            if counter % sample_period == 0:
                run_dataset, tau_mj_real = add_sim_data_to_dataset(run_dataset, current_time, data, model, feet_geom_id, feet_body_id, data_aux)
                run_dataset["command"].append(np.asarray(command))
                if np.any(np.isnan(tau_mj_real)) or np.any(np.isnan(qvel)):
                    nan_flag = True

                for i in range(data.nefc):
                    if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                        joint_name = mujoco.mj_id2name(
                                model,
                                mujoco.mjtObj.mjOBJ_JOINT,
                                data.efc_id[i]
                            )
                        if data.efc_id[i] != 7:
                            print(f"joint limit active id {data.efc_id[i]} name {joint_name} data.efc_force[i] {data.efc_force[i]} ")
                            nan_flag = True

            if counter % period == 0:
                foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
                contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
                # command = update_z_command(command, counter, z_offset=extra_qref_data["z_offset"], z_amp=extra_qref_data["z_amp"], z_freq=extra_qref_data["z_freq"])
                if not base_ref_fixed:
                    if hasattr(config, "ref_type") and config.ref_type == "fourier":
                        command = update_z_fourier(command, counter, extra_qref_data["z_fourier_params"])
                        command = update_ang_fourier(command, counter, extra_qref_data["ang_fourier_params"])
                    elif hasattr(config, "reference_generator"):
                        command = update_z_command(command, counter, z_offset=extra_qref_data["z_offset"], z_amp=extra_qref_data["z_amp"], z_freq=extra_qref_data["z_freq"])
                        command = update_ang_command(command, counter, ang_offset=extra_qref_data["ang_offset"], ang_amp=extra_qref_data["ang_amp"], ang_freq=extra_qref_data["ang_freq"])
                    else:
                        command = update_z_command(command, counter, z_offset=extra_qref_data["z_offset"], z_amp=extra_qref_data["z_amp"], z_freq=extra_qref_data["z_freq"])
                    
                    if cmd_ref == 'fourier' and config.ref_type == "fourier":
                        command = update_xy_ang_fourier(command, counter, extra_qref_data["xy_fourier_params"], extra_qref_data["ang_fourier_params"], extra_qref_data["yaw_fourier_params"])

                if base_ref_fixed:
                    command = jnp.concatenate(
                        [
                            jnp.zeros(6),
                            jnp.array([config.robot_height]),
                            jnp.array([1.0, 0.0, 0.0, 0.0]),
                        ]
                    )


                # print(command)
                # for i in range(data.nefc):
                #     if data.efc_type[i] != mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC:
                #         print(f"efc {i} type {data.efc_type[i]} id {data.efc_id[i]} force {data.efc_force[i]}")
                    # if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT:
                    #     joint_name = mujoco.mj_id2name(
                    #             model,
                    #             mujoco.mjtObj.mjOBJ_JOINT,
                    #             data.efc_id[i]
                    #         )
                    #     if data.efc_id[i] != 7:
                    #         print(f"joint limit active id {data.efc_id[i]} name {joint_name} data.efc_force[i] {data.efc_force[i]} bias {data.qfrc_bias[6+data.efc_id[i]]}")

                mpc_data, tau = solve_mpc(
                    mpc_data,
                    data.qpos.copy(),
                    data.qvel.copy(),
                    foot,
                    command,
                    contact * 0.0,
                )
                tau.block_until_ready()

            data.ctrl = np.asarray(tau)
            mujoco.mj_step(model, data)

            counter += 1
            total_counter += 1
            viewer.sync()

    print(f'qpos {data.qpos} qvel {data.qvel}')
    folder_name = "datasets/sim_spot_arm/"
    os.makedirs(folder_name, exist_ok=True)

    dataset["duty_factor"] = config.duty_factor
    dataset["step_freq"] = config.step_freq
    dataset["step_height"] = config.step_height

    # Plot z position and z reference (z_ref taken from saved command[:, 6])
    print('Plotting z position and reference from saved command...')
    if len(dataset["q"]) > 0 and "command" in dataset and len(dataset["command"]) > 0:
        all_times = []
        all_z_positions = []
        all_z_refs = []

        for run_idx, q_run in enumerate(dataset["q"]):
            q_run = np.asarray(q_run)
            t_run = np.asarray(dataset["t"][run_idx]) if run_idx < len(dataset["t"]) else np.arange(q_run.shape[0]) / dataset_frequency

            if run_idx < len(dataset["command"]):
                command_run = np.asarray(dataset["command"][run_idx])
                if command_run.ndim == 1:
                    z_ref_run = np.array([command_run[6]])
                else:
                    z_ref_run = command_run[:, 6]
            else:
                z_ref_run = np.zeros_like(t_run)

            if q_run.ndim == 1:
                z_pos = np.array([q_run[2]])
            else:
                z_pos = q_run[:, 2]

            L = min(len(t_run), len(z_pos), len(z_ref_run))
            times = t_run[:L] + run_idx * run_length_time
            all_times.extend(times.tolist())
            all_z_positions.extend(z_pos[:L].tolist())
            all_z_refs.extend(z_ref_run[:L].tolist())

        if len(all_times) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(all_times, all_z_positions, 'b-', label='z position', linewidth=2)
            plt.plot(all_times, all_z_refs, 'r--', label='z_ref', linewidth=1.5)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Z Position (m)', fontsize=12)
            plt.title('Base Z Position and Reference Over Time', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.tight_layout()

            plot_filename = folder_name + "z_position_vs_ref.png"
            plt.savefig(plot_filename, dpi=150)
            print(f"Plot saved to {plot_filename}")
            # try:
            #     plt.show()
            # except Exception:
            #     pass

    # Plot xy position and reference (xy_ref taken from saved command[:, 0:2])
    print('Plotting xy velocity and reference from saved command...')
    if len(dataset["q"]) > 0 and "command" in dataset and len(dataset["command"]) > 0:
        all_times = []
        all_x_velocities = []
        all_y_velocities = []
        all_x_refs = []
        all_y_refs = []

        for run_idx, qd_run in enumerate(dataset["qd"]):
            qd_run = np.asarray(qd_run)
            t_run = np.asarray(dataset["t"][run_idx]) if run_idx < len(dataset["t"]) else np.arange(qd_run.shape[0]) / dataset_frequency

            if run_idx < len(dataset["command"]):
                command_run = np.asarray(dataset["command"][run_idx])
                if command_run.ndim == 1:
                    x_ref_run = np.array([command_run[0]])
                    y_ref_run = np.array([command_run[1]])
                else:
                    x_ref_run = command_run[:, 0]
                    y_ref_run = command_run[:, 1]
            else:
                x_ref_run = np.zeros_like(t_run)
                y_ref_run = np.zeros_like(t_run)

            if qd_run.ndim == 1:
                x_pos = np.array([qd_run[0]])
                y_pos = np.array([qd_run[1]])
            else:
                x_pos = qd_run[:, 0]
                y_pos = qd_run[:, 1]

            L = min(len(t_run), len(x_pos), len(y_pos), len(x_ref_run), len(y_ref_run))
            times = t_run[:L] + run_idx * run_length_time
            all_times.extend(times.tolist())
            all_x_velocities.extend(x_pos[:L].tolist())
            all_y_velocities.extend(y_pos[:L].tolist())
            all_x_refs.extend(x_ref_run[:L].tolist())
            all_y_refs.extend(y_ref_run[:L].tolist())

        if len(all_times) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(all_times, all_x_velocities, 'b-', label='x velocity', linewidth=2)
            plt.plot(all_times, all_y_velocities, 'g-', label='y velocity', linewidth=2)
            plt.plot(all_times, all_x_refs, 'r--', label='x_ref', linewidth=1.5)
            plt.plot(all_times, all_y_refs, 'm--', label='y_ref', linewidth=1.5)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Velocity (m/s)', fontsize=12)
            plt.title('Base XY Velocity and Reference Over Time', fontsize=14)
            plt.grid(True, alpha=0.3)
    # Plot Euler angles and reference Euler angles (from saved command quaternion)
    print('Plotting Euler angles and reference from saved command...')
    if len(dataset["q"]) > 0 and "command" in dataset and len(dataset["command"]) > 0:
        all_times = []
        all_euler = []
        all_euler_ref = []

        for run_idx, q_run in enumerate(dataset["q"]):
            if run_idx >= len(dataset["command"]):
                continue

            q_run = np.asarray(q_run)
            command_run = np.asarray(dataset["command"][run_idx])

            if q_run.ndim == 1:
                q_run = q_run[None, :]
            if command_run.ndim == 1:
                command_run = command_run[None, :]

            # quaternion reference exists only when command includes orientation (len >= 11)
            if command_run.shape[1] < 11:
                continue

            t_run = np.asarray(dataset["t"][run_idx]) if run_idx < len(dataset["t"]) else np.arange(q_run.shape[0]) / dataset_frequency
            L = min(len(t_run), q_run.shape[0], command_run.shape[0])
            if L == 0:
                continue

            times = t_run[:L] + run_idx * run_length_time

            for i in range(L):
                # q_run stores base quaternion as xyzw in entries [3:7]
                quat_wxyz = q_run[i, 3:7]
                euler_cur = np.asarray(rotation_utils.quaternion_to_rpy(jnp.asarray(quat_wxyz)))

                # command stores quaternion reference as wxyz in last 4 entries
                quat_ref_wxyz = command_run[i, -4:]
                euler_ref = np.asarray(rotation_utils.quaternion_to_rpy(jnp.asarray(quat_ref_wxyz)))

                all_times.append(float(times[i]))
                all_euler.append(euler_cur)
                all_euler_ref.append(euler_ref)

        if len(all_times) > 0:
            all_times = np.asarray(all_times)
            all_euler = np.asarray(all_euler)
            all_euler_ref = np.asarray(all_euler_ref)

            fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            labels = ['roll', 'pitch', 'yaw']
            for k in range(3):
                axs[k].plot(all_times, all_euler[:, k], 'b-', linewidth=2, label=f'{labels[k]}')
                axs[k].plot(all_times, all_euler_ref[:, k], 'r--', linewidth=1.5, label=f'{labels[k]}_ref')
                axs[k].set_ylabel(f'{labels[k]} (rad)', fontsize=11)
                axs[k].grid(True, alpha=0.3)
                axs[k].legend(fontsize=10)

            axs[-1].set_xlabel('Time (s)', fontsize=12)
            fig.suptitle('Base Euler Angles and Quaternion Reference Over Time', fontsize=14)
            plt.tight_layout()

            plot_filename = folder_name + "euler_vs_ref.png"
            plt.savefig(plot_filename, dpi=150)
            print(f"Euler plot saved to {plot_filename}")
            # try:
            #     plt.show()
            # except Exception:
            #     pass

    # Plot arm joint positions
    print('Plotting arm joint positions...')
    if len(dataset["q"]) > 0:
        all_times = []
        all_arm_joints = []
        joint_index = np.arange(7, 14)  # assuming arm joints are at the end of qpos after base (7) and before any additional states

        for run_idx, q_run in enumerate(dataset["q"]):
            q_run = np.asarray(q_run)
            t_run = np.asarray(dataset["t"][run_idx]) if run_idx < len(dataset["t"]) else np.arange(q_run.shape[0]) / dataset_frequency

            if q_run.ndim == 1:
                # Single sample: extract arm joints (indices 7+)
                arm_joints = q_run[joint_index]
                arm_joints_traj = arm_joints[None, :]
            else:
                # Multiple samples: extract arm joints (indices 7+)
                arm_joints_traj = q_run[:, joint_index]
            L = min(len(t_run), arm_joints_traj.shape[0])
            if L == 0:
                continue

            times = t_run[:L] + run_idx * run_length_time

            for i in range(L):
                all_times.append(float(times[i]))
                all_arm_joints.append(arm_joints_traj[i, :])

        if len(all_times) > 0:
            all_times = np.asarray(all_times)
            all_arm_joints = np.asarray(all_arm_joints)
            n_joints = all_arm_joints.shape[1]

            fig, axs = plt.subplots(n_joints, 1, figsize=(12, 3 * n_joints), sharex=True)
            if n_joints == 1:
                axs = [axs]

            for j in range(n_joints):
                axs[j].plot(all_times, all_arm_joints[:, j], 'b-', linewidth=2, label=f'joint_{j}')
                axs[j].set_ylabel(f'joint_{j} (rad)', fontsize=11)
                axs[j].grid(True, alpha=0.3)
                axs[j].legend(fontsize=10)

            axs[-1].set_xlabel('Time (s)', fontsize=12)
            fig.suptitle('Arm Joint Positions Over Time', fontsize=14)
            plt.tight_layout()

            plot_filename = folder_name + "arm_joints.png"
            plt.savefig(plot_filename, dpi=150)
            print(f"Arm joints plot saved to {plot_filename}")
            try:
                plt.show()
            except Exception:
                pass

    nsamples = int(n_runs * run_length_time * dataset_frequency)
    filename = f"samples_{nsamples}_n_runs_{n_runs}_data_freq_{int(dataset_frequency)}"
    if gait_mode == 'walk':
        filename = "walk_" + filename
    filename += f"_sim_freq_{int(sim_frequency)}_total_time_{int(run_length_time)}_base_full_arm"
    # if reference with orientation:
    if hasattr(config, "reference_generator"):
        if config.ref_type == "fourier":
            filename += "_ang_ref_fourier_v2_fix"
        else:
            filename += "_ang_ref_v0_test"

    if config.step_freq <= 1e-3:
        filename += "_stand"
        print(dataset["labels"])
        modified_list = []
        for label in dataset["labels"]:
            # dataset["labels"][label] = "1" + dataset["labels"][label]  # add 1 to all labels to indicate standing
    
            # Split from the right exactly once at the last '_'
            prefix, run_number = label.rsplit("_", 1)

            # Convert to int, add 100, and format back into the string
            new_string = f"{prefix}_{int(run_number) + 100}"
            modified_list.append(new_string)
            print(f'modified label {new_string}')
        dataset["labels"] = modified_list

        print(f'new {dataset["labels"]}')

    if base_ref_fixed:
        filename += "_base_fixed"
    if rand_step_freq:
        filename += "_rand_step_freq"
    filename += f"_contact_Proj_seed{seed}.pkl"


    with open(folder_name + filename, "wb") as fp:
        pickle.dump(dataset, fp)
        print(f"Dictionary saved successfully to file {filename} | Nsamples {total_counter}")


if __name__ == "__main__":
    main()
