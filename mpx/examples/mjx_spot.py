import argparse
import os
import sys
import time
from timeit import default_timer as timer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

import mpx.config.config_spot as config
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.utils.sim as sim_utils
import mpx.utils.rotation as rotation_utils

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact):
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc


def extend_comand(command, quat_ref):
    # Extend the command with the reference orientation for the base.
    return jnp.concatenate([command, quat_ref])

def main(headless=False, steps=500, scene="flat"):
    model = mujoco.MjModel.from_xml_path(
        dir_path + f"/../data/boston_dynamics_spot/scene.xml"
    )
    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    mpc = mpc_wrapper.MPCWrapper(config, limited_memory=True)
    command_handle = sim_utils.KeyboardVelocityCommand()
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    warm_command = jnp.asarray(command_handle.mpc_input(config.robot_height))
    if hasattr(config, "reference_generator"):
        quat_ref = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion as reference orientation
        warm_command = extend_comand(warm_command, quat_ref)
    warm_contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        warm_command,
        warm_contact,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    q_ref = config.q0.copy()
    euler_ref_traj = []
    euler_traj = []

    def step_controller():
        nonlocal counter, tau, q_ref, mpc_data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
           
            command = jnp.asarray(command_handle.mpc_input(config.robot_height))
            if hasattr(config, "reference_generator"):
                # quat_ref = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion as reference orientation
                roll_ref = 0.3 * jnp.sin(2 * jnp.pi * 1.0 *counter * model.opt.timestep)
                pitch_ref = 0.8 * jnp.sin(2 * jnp.pi * 1.0 *counter * model.opt.timestep)  # Oscillating pitch reference
                euler_ref = jnp.array([roll_ref, pitch_ref, 0.0])  # Roll and pitch oscillation, no yaw
                quat_ref = rotation_utils.rpy_to_quat(euler_ref)  # No rotation as reference
                command = extend_comand(command, quat_ref)

                euler_ref_traj.append(euler_ref)
                euler_traj.append(rotation_utils.quaternion_to_rpy(qpos[3:7]))

            print(f"Command: {command[6:10]}")
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
            print(f"Contact: {contact}")
            print(foot)
            print(f"Command: {command}")
            
            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                contact,
            )
            tau.block_until_ready()
            stop = timer()

            # tau = jnp.clip(tau, config.min_torque, config.max_torque)
            # The shifted warm start is the next joint target used by the PD stabilizer.
            q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau)
        mujoco.mj_step(model, data)
        counter += 1

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        viewer.sync()
        base_marker_ids = None
        base_pred_marker_ids = None
        pred_foot_marker_ids = None
        while viewer.is_running():
            overlay_text = command_handle.consume_overlay_text()
            tic = timer()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            step_controller()
            toc = timer()
            if toc - tic < model.opt.timestep:
                sleep_time = model.opt.timestep - (toc - tic)
                time.sleep(sleep_time)

            base_pos = data.qpos[:3].reshape(1, 3)
            base_marker_diameter = float(np.clip(0.03 * model.stat.extent, 0.01, 0.06))
            base_marker_ids = sim_utils.render_sphere_trajectory(
                viewer,
                base_pos,
                np.ones(base_pos.shape[0], dtype=np.float64),
                diameter=base_marker_diameter,
                color=np.array([1.0, 0.0, 0.0, 1.0]),
                geom_ids=base_marker_ids
            )

            base_pos_pred = mpc_data.X0[:, :3].reshape(-1, 3)
            base_pred_marker_ids= sim_utils.render_sphere_trajectory(
                viewer,
                base_pos_pred,
                np.ones(base_pos_pred.shape[0], dtype=np.float64),
                diameter=base_marker_diameter,
                # red color for predicted base position
                color=np.array([0.0, 1.0, 0.0, 1.0]),
                geom_ids=base_pred_marker_ids
            )

            pred_foot_pos = mpc_data.X0[:, mpc.foot_slice].reshape(-1, 3)
            pred_foot_marker_ids = sim_utils.render_sphere_trajectory(
                viewer,
                pred_foot_pos,
                np.ones(pred_foot_pos.shape[0], dtype=np.float64),
                diameter=base_marker_diameter,
                # blue color for predicted foot positions
                color=np.array([0.0, 0.0, 1.0, 1.0]),
                geom_ids=pred_foot_marker_ids
            )

            viewer.sync()

        euler_ref_traj_np = np.array(euler_ref_traj)
        euler_traj_np = np.array(euler_traj)
        

        # plot the reference and actual euler angles over time
        import matplotlib.pyplot as plt
        time_array = np.arange(euler_ref_traj_np.shape[0]) * (period * model.opt.timestep)
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(time_array, euler_ref_traj_np[:, 0], label="Reference Roll")
        plt.plot(time_array, euler_traj_np[:, 0], label="Actual Roll")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(time_array, euler_ref_traj_np[:, 1], label="Reference Pitch")
        plt.plot(time_array, euler_traj_np[:, 1], label="Actual Pitch")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(time_array, euler_ref_traj_np[:, 2], label="Reference Yaw")
        plt.plot(time_array, euler_traj_np[:, 2], label="Actual Yaw")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scene", type=str, default="flat")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
        scene=args.scene,
    )
