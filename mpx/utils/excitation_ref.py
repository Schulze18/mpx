import jax
import jax.numpy as jnp
from jax import random

def generate_constraints(order: int):
    """
    Trajectories are parameterized as
        q = \sum_k A_k*1/wk sin(k*wt) - B_k*1/wk cos(k*wt)
        qd = \sum_k A_k cos(k*wt) + B_k sin(k*wt)
        qdd = -\sum_k A_k*wk sin(k*wt) + B_k*wk cos(k*wt)

    Initial position, velocity, acceleration constraints: q(0) = qd(0) = qdd(0) = 0
        cA:  sum_k A_1 + ... + A_k + ... + A_L = 0
        cB1: sum_k B_1/1 + ... B_k/k + ... + B_L/L = 0
        cB2: sum_k B_1*1 + ... B_k*k + ... + B_L*L = 0
        where l is the order, so for B, to simplify the
        constraints, we can rewrite the constraints
        [
            [1, 1/2, 1/3, ..., 1/l],
            [1, 2, 3, ..., l]
        ] * B = 0, where B = [B_1, B_2, ..., B_L]
        to the reduced echelon_form so that we can generate
        parameters that satisfy the constraints

    Param:
        order: int, order of the Fourier series
    Return:
        cA: constraints vector, A[0] = -jnp.dot(cA,A)
        cB: constraints matrix, B[1] = -jnp.dot(cB[0][2:], B[2:])
                                B[0] = -jnp.dot(cB[1][1:], B[1:])
    """
    cA = jnp.ones(order)
    cB = jnp.array(
        [[1 / i for i in range(1, order + 1)], [i for i in range(1, order + 1)]]
    )
    cB_reduced_echelon_form = jnp.array([(cB[0] - cB[1]) / (cB[0] - cB[1])[1], cB[1]])
    return cA, cB_reduced_echelon_form

def generate_random_param(order: int, njoints: int, param_range = None, key = None):
    """
    Generate constant terms for the Fourier basis, each joint has
    different frequencies

    Param:
        order, njoints, param_range, key: JAX random key
    Return:
        params: an array that includes parameters, shape (2, order, njoints)
            A: parameters for the sin basis, shape (order, njoints)
            B: parameters for the cos basis, shape (order, njoints)
    """
    if key is None:
        key = random.PRNGKey(0)
    
    key1, key2, key3 = random.split(key, 3)
    
    if param_range is None:
        params = (
            random.normal(key1, (2, order, njoints)) * 0.02 # 0.02 works for duration 100, 0.04 works for duration (40, 80], 0.08 works for duration <= 40
                                                    # for single joint, duration 100, 0.08 works
        )  # use tiny variance to tweak the params
    else:
        rand_comp = random.uniform(key1, (2, order, njoints), minval=-1.0, maxval=1.0)
        param_norm = 1.0 * (param_range * 0.5 * 1.0 / order)
        # print(f'param_norm {param_norm}')
        params = rand_comp * param_norm

    cA, cB = generate_constraints(order)
    A, B = params[0], params[1]
    A = A.at[-1].set(-jnp.dot(cA[:-1], A[:-1]))
    B = B.at[1].set(-jnp.dot(cB[0][2:], B[2:]))
    B = B.at[0].set(-jnp.dot(cB[1][1:], B[1:]))
    phi0 = random.uniform(key2, (njoints,), minval=0.0, maxval=2 * jnp.pi)
    params = [A, B, phi0]
    return params

def generate_fourier_traj(t, params, return_vel=False):
    """
    Generate Fourier series trajectory over time array.
    
    Param:
        t: time array of shape (T,)
        order: int, order of the Fourier series
        params: list [A, B, phi0, duration, q0]
            A: shape (order, njoints)
            B: shape (order, njoints)
            phi0: shape (njoints,)
            duration: float, period of the motion
            q0: shape (njoints,) or (T, njoints), initial position
    
    Return:
        q: trajectory of shape (T, njoints)
    """
    A, B = params[0], params[1]
    phi0 = params[2]
    duration = params[3]
    q0 = params[4]
    omega_f = 2 * jnp.pi / duration
    
    njoints = A.shape[1]
    nt = t.shape[0]
    
    # Initialize trajectory: (T, njoints)
    if q0.ndim == 1:
        q = jnp.tile(q0[None, :], (nt, 1))
    else:
        q = q0

    order = A.shape[0]
    
    # Debug
    # import sys
    # print(f"[excitation_ref] A shape: {A.shape}, B shape: {B.shape}", file=sys.stderr)
    # print(f"[excitation_ref] omega_f: {omega_f}, duration: {duration}", file=sys.stderr)
    # print(f"[excitation_ref] q initial shape: {q.shape}", file=sys.stderr)
    
    # Compute Fourier series for each harmonic
    for k in range(1, order + 1):
        # t shape: (T,), expand to (T, 1) for broadcasting
        t_exp = t[:, None]  # (T, 1)
        
        # Compute sine and cosine components
        sin_term = jnp.sin(2 * omega_f * k * t_exp + phi0[None, :])  # (T, njoints)
        cos_term = jnp.cos(2 * omega_f * k * t_exp + phi0[None, :])  # (T, njoints)
        
        # Add Fourier components: A_k/w_k * sin(...) - B_k/w_k * cos(...)
        q = q + (sin_term * A[k - 1, :][None, :] / (2 * omega_f * k) - 
                 cos_term * B[k - 1, :][None, :] / (2 * omega_f * k))
        # if k == 1:
        #     import sys
        #     print(f"[excitation_ref] k={k}: sin_term[0]: {sin_term[0]}, A[0]: {A[k-1]}, contribution: {(sin_term[0] * A[k - 1, :] / (2 * omega_f * k))}", file=sys.stderr)
    
    if return_vel:
        qd = jnp.zeros_like(q)
        for k in range(1, order + 1):
            t_exp = t[:, None]  # (T, 1)
            # sin_term = jnp.sin(omega_f * k * t_exp + phi0[None, :])  # (T, njoints)
            # cos_term = jnp.cos(omega_f * k * t_exp + phi0[None, :])  # (T, njoints)
            qd = qd + (cos_term * A[k - 1, :][None, :] -
                       sin_term * B[k - 1, :][None, :])
        return q, qd
    return q
