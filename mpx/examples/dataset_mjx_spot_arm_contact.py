import os
from timeit import default_timer as timer

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_spot_arm as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils

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
    print(f'diff constraint contact  : {sim_mj_data.qfrc_constraint - tau_contact_recompute}')
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
    key, key_ang, key_lin, key_amp, key_freq, key_init, key_z_offset, key_z_amp, key_z_freq = jax.random.split(key, 9)

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

    extra_qref_data = {
        "amp": amp_ref,
        "freq": freq_ref,
        "joint_index": joint_index,
        "z_offset": z_offset,
        "z_amp": z_amp,
        "z_freq": z_freq
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

    return key, q_init, command, extra_qref_data

def update_z_command(command, counter, z_offset = 0.4, z_amp = 0.15, z_freq = 0.1):
    sin_time = counter / sim_frequency
    z_pos = z_offset + z_amp * jnp.sin(2 * jnp.pi * z_freq * sin_time)
    z_vel = 2 * jnp.pi * z_freq * z_amp * jnp.cos(2 * jnp.pi * z_freq * sin_time)
    command = command.at[6].set(z_pos)
    command = command.at[2].set(z_vel)
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

    rng_key = jax.random.PRNGKey(0)
    period = int(sim_frequency / config.mpc_frequency)
    sample_period = int(sim_frequency / dataset_frequency)

    counter = 0
    tau = jnp.zeros(config.n_joints)
    command = jnp.zeros(7)
    run_dataset = init_run_dataset()
    nan_flag = False
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if total_counter % int(run_length_time * sim_frequency) == 0 or total_counter == 0 or nan_flag:
                counter = 0
                rng_key, q_init, command, extra_qref_data = sample_reset_references(rng_key)

                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)
                data.qpos[:] = np.asarray(q_init)
                data.qvel[:] = np.zeros_like(data.qvel)
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)

                foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
                mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)
                mpc_data = mpc_data.replace(extra_qref_data=extra_qref_data)

                if run_id >= 0:
                    for key in run_dataset.keys():
                        if key not in ["labels", "all_contact_Jac", "all_contact_force"]:
                            run_dataset[key] = np.array(run_dataset[key])

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
                            print(f"joint limit active id {data.efc_id[i]} name {joint_name} data.efc_force[i] {data.efc_force[i]} bias {data.qfrc_bias[6+data.efc_id[i]]}")
                            nan_flag = True

            if counter % period == 0:
                foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
                contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
                command = update_z_command(command, counter, z_offset=extra_qref_data["z_offset"], z_amp=extra_qref_data["z_amp"], z_freq=extra_qref_data["z_freq"])
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

    nsamples = int(n_runs * run_length_time * dataset_frequency)
    filename = f"samples_{nsamples}_n_runs_{n_runs}_data_freq_{int(dataset_frequency)}"
    filename += f"_sim_freq_{int(sim_frequency)}_total_time_{int(run_length_time)}_base_full_arm"
    filename += "_contact_Proj.pkl"

    with open(folder_name + filename, "wb") as fp:
        pickle.dump(dataset, fp)
        print(f"Dictionary saved successfully to file {filename} | Nsamples {total_counter}")


if __name__ == "__main__":
    main()
