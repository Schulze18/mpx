import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, '..')))
import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import numpy as np
import mujoco
import mujoco.viewer
import numpy as np
import time
from gym_quadruped.utils.mujoco.visual import render_sphere ,render_vector
import mpx.utils.mpc_wrapper as mpc_wrapper
import mpx.config.config_spot_arm as config

model = mujoco.MjModel.from_xml_path(config.model_path)
data = mujoco.MjData(model)
sim_frequency = 200.0
model.opt.timestep = 1/sim_frequency

contact_id = []
for name in config.contact_frame:
    contact_id.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name))
mpc = mpc_wrapper.MPCControllerWrapper(config)
data.qpos = jnp.concatenate([config.p0, config.quat0,config.q0])

from timeit import default_timer as timer

ids = []
tau = jnp.zeros(config.n_joints)
paused = False
step_once = False

def key_callback(keycode):
    global paused, step_once
    key = chr(keycode).lower() if 0 <= keycode < 256 else ""
    if key == 'p':
        paused = not paused
        print(f"Paused: {paused}")
    elif key == 'n':
        step_once = True

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    mujoco.mj_step(model, data)
    viewer.sync()
    print("Controls: [P] pause/resume, [N] step one sim tick")
    delay = int(0*sim_frequency)
    print('Delay: ',delay)
    mpc.robot_height = config.robot_height
    mpc.reset(data.qpos.copy(),data.qvel.copy())
    counter = 0
    while viewer.is_running():
        if paused and not step_once:
            viewer.sync()
            time.sleep(0.01)
            continue

        step_once = False
        
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        foot_current = np.array([data.geom_xpos[i] for i in contact_id]).reshape(-1)
        if config.grf_as_state:
            x0_dbg = jnp.concatenate([qpos, qvel, foot_current, jnp.zeros(3 * config.n_contact)])
        else:
            x0_dbg = jnp.concatenate([qpos, qvel, foot_current])

        reference_dbg, _, _ = mpc._ref_gen(
            duty_factor=mpc.duty_factor,
            step_freq=mpc.step_freq,
            step_height=mpc.step_height,
            t_timer=mpc.contact_time.copy(),
            x=x0_dbg,
            foot=foot_current,
            input=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.robot_height]),
            liftoff=mpc.liftoff,
            contact=jnp.zeros(config.n_contact),
            clearence_speed=mpc.clearence_speed,
        )
        contact_ref_dbg = np.array(reference_dbg[0, 13 + config.n_joints + 3 * config.n_contact:13 + config.n_joints + 4 * config.n_contact])
        grf_ref_dbg = np.array(reference_dbg[0, 13 + config.n_joints + 4 * config.n_contact:13 + config.n_joints + 7 * config.n_contact])
        foot_ref = np.array(reference_dbg[0, 13 + config.n_joints:13 + config.n_joints + 3 * config.n_contact])
        # print(f"contact_ref {contact_ref_dbg} grf_ref {grf_ref_dbg}")

        if counter % (sim_frequency / config.mpc_frequency) == 0 or counter == 0:
            
            # if counter != 0:
            #     for i in range(delay):
            #         qpos = data.qpos.copy()
            #         qvel = data.qvel.copy()
            #         tau_fb = -3*(qvel[6:6+config.n_joints])
            #         # tau_fb = 10*(q-qpos[7:7+config.n_joints]) -2*(qvel[6:6+config.n_joints])
            #         data.ctrl = tau + tau_fb
            #         mujoco.mj_step(model, data)
            #         counter += 1
            start = timer()
            ref_base_lin_vel = jnp.array([0.1,0.0,0])
            ref_base_ang_vel = jnp.array([0,0,0.3])
            
            # x0 = jnp.concatenate([qpos, qvel,jnp.zeros(3*config.n_contact)])
            input = np.array([ref_base_lin_vel[0],ref_base_lin_vel[1],ref_base_lin_vel[2],
                           ref_base_ang_vel[0],ref_base_ang_vel[1],ref_base_ang_vel[2],
                           config.robot_height])
            
            #set this to the current contact state to use the blind step adaptation
            contact = np.zeros(config.n_contact)
        
            start = timer()
            tau, q, dq  = mpc.run(qpos,qvel,input,contact)  
            stop = timer()
            print(f"Time elapsed: {stop-start} qbase_vel {qvel[:6]} {qpos[7:14]} tau {tau} foot_current {foot_current} foot_ref{foot_ref} contact {contact} ")  
            
            # print(f"Time elapsed: {stop-start} qbase_vel {qvel[:6]} {qpos[7:14]} tau {tau} foot_current {foot_current} contact_ref {contact_ref_dbg} grf_ref {grf_ref_dbg} grf_opt {mpc.grf}")  
            # print(f"Time elapsed: {stop-start} qarm {qpos[7:14]} tau {tau} foot_current {foot_current} contact_ref {contact_ref_dbg} grf_ref {grf_ref_dbg} grf_opt {mpc.grf}")                   
        counter += 1        
        tau_fb = 10*(q-qpos[7:7+config.n_joints])-3*(qvel[6:6+config.n_joints])
        # tau_fb = -3*(qvel[6:6+config.n_joints])
        data.ctrl = tau + tau_fb
        mujoco.mj_step(model, data)
        viewer.sync()
        
    
    

