from typing import Tuple
import numpy as np


def piecewise_constant(setpoints, setpoints_duration, Ts):
    """
    Defines the sampled version with sample time Ts of a piecewise constant reference,

                 _
                | setpoints(0)      0   <= t < T_1
                | setpoints(1)      T_1 <= t < T_1 + T_2
      ref(t) = <
                | ...
                | setpoints(n-1)    T_1 + ... T_(n-1) <= t <= T_1 + ... + T_n
                 _

    where T_1, ..., T_n are the duration of each setpoint, i.e. the entries of setpoints_duration

    """
    # compute the number of samples for each setpoint
    n_samples = np.rint(setpoints_duration / Ts).astype(int)

    # compute the reference for each setpoint
    ref = setpoints[0] * np.ones((n_samples[0],))

    for i in range(1, len(setpoints)):
        ref = np.append(ref, setpoints[i] * np.ones((n_samples[i],)))

    # add one sample to account for the non-strict inequality on both sides
    # in the definition of the last setpoint
    ref = np.append(ref, ref[-1])

    # compute final time instant
    Tf = np.round(np.sum(n_samples) * Ts, 3)

    return (ref, Tf)


def compute_num_steps(ts_sim, Ts, Tf) -> Tuple[int, int, int]:
    """

    Args:
        ts_sim: simulation time step
        Ts: controller sample time
        Tf: final time instant

    Returns: (N_steps, N_steps_dt, n_update)
        N_steps:
        N_steps_dt:
        n_update:
    """
    # check consistency
    if not (Ts / ts_sim).is_integer():
        raise ValueError(
            "The sample time Ts has to be an integer multiple of the simulation time step ts_sim"
        )
    if not (Tf / ts_sim).is_integer():
        raise ValueError(
            "The simulation time Tf has to be an integer multiple of the simulation time step ts_sim"
        )

    # compute the number of simulation steps
    N_steps = int(Tf / ts_sim)

    # compute the number of steps for the discrete-time part of the loop
    N_steps_dt = int(Tf / Ts)

    # number of simulation steps every which to update the discrete-time part of the loop
    n_update = int(Ts / ts_sim)

    return (N_steps, N_steps_dt, n_update)


def get_uniform_grid(T, N):
    # compute length of each shooting interval
    time_step = T / N

    # compute the shooting nodes
    shooting_nodes = np.linspace(0, time_step * N, N + 1)

    return shooting_nodes


def get_nonuniform_grid(T, Ts, intermediate_nodes):
    # start with a uniform grid
    uniform_grid = get_uniform_grid(T, int(T / Ts))

    # construct the list of indices to select,
    # including the first and the last sample
    indices = np.insert(intermediate_nodes, 0, 0)
    indices = np.append(indices, -1)

    # get the shooting nodes as the instants of the uniform grid
    # corresponding to the selected indices
    shooting_nodes = uniform_grid[indices]

    return shooting_nodes
