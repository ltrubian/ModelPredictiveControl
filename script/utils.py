from typing import Literal, Tuple, get_args

import numpy as np
from numpy.typing import NDArray

_REFERENCE_TYPE = Literal["swing-up", "horizontal", "empty-2s"]
_INITIAL_TYPE = Literal["up", "down", "off-balance"]


def get_reference(
    Ts: float, N: int, nx: int, nu: int, ref_type: _REFERENCE_TYPE
) -> Tuple[NDArray, int]:
    """Return preset references. It uses the provided piecewise_constant.

    Args:
        Ts: controller sampling time
        N: shooting intervals
        nx: controller n° states
        nu: controller n° inputs
        type: (swing-up/horizontal)

    Returns:
        y_ref: reference signal for the controller
        Tf: last time instant of the simulation
    """

    match ref_type:
        case "swing-up":
            angle_ref, Tf = piecewise_constant(
                np.array([np.pi, 0]), np.array([5, 10]), Ts
            )
            # - provide a reference for all variables
            y_ref = np.column_stack(
                (
                    np.zeros((len(angle_ref), 1)),
                    angle_ref.reshape(-1, 1),
                    np.zeros((len(angle_ref), nx + nu - 2)),
                )
            )
        case "horizontal":
            pos_ref, Tf = piecewise_constant(
                np.array([0, 1, -1, 0]), np.array([2.5, 5, 7.5, 10]), Ts
            )
            # - provide a reference for all variables
            y_ref = np.column_stack(
                (
                    pos_ref.reshape(-1, 1),
                    np.zeros((len(pos_ref), 1)),
                    np.zeros((len(pos_ref), nx + nu - 2)),
                )
            )
        case "empty-2s":
            pos_ref, Tf = piecewise_constant(np.array([0]), np.array([2]), Ts)
            # - provide a reference for all variables
            y_ref = np.column_stack(
                (
                    pos_ref.reshape(-1, 1),
                    np.zeros((len(pos_ref), 1)),
                    np.zeros((len(pos_ref), nx + nu - 2)),
                )
            )

        case _:
            raise ValueError(f"{ref_type = } is not in {get_args(_REFERENCE_TYPE)}")

    # - add N samples at the end (replicas of the last sample) for reference look-ahead
    y_ref = np.vstack((y_ref, np.repeat(y_ref[-1].reshape(1, -1), N, axis=0)))
    return (y_ref, Tf)


def get_initial_condition(nx: int, ini_type: _INITIAL_TYPE) -> NDArray:
    """Return preset initial condition
    The only difference between them is the angle of the pendulum

    Args:
        - nx: controller n° states
        - ini_type: (up/down/off-balance)

    Returns:
        - x0: initial condition
    """

    x0 = np.zeros((nx,))
    match ini_type:
        case "up":
            pass
        case "down":
            x0[1] = np.pi
        case "off-balance":
            x0[1] = np.pi / 3
        case _:
            raise ValueError(f"{ini_type = } is not in {get_args(_INITIAL_TYPE)}")
    return x0


def piecewise_constant(
    setpoints: NDArray[np.float64], setpoints_duration: NDArray[np.float64], Ts: float
) -> Tuple[NDArray, int]:
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


def compute_num_steps(ts_sim: float, Ts: float, Tf: float) -> Tuple[int, int, int]:
    """compute controller step and simulation for the loop dynamics

    Args:
        ts_sim: simulation time step
        Ts: controller sample time
        Tf: final time instant

    Returns: (N_steps, N_steps_dt, n_update)
        N_steps: number of simulation steps
        N_steps_dt: number of steps for the discrete-time part of the loop
        n_update: number of simulation steps
    """

    # check consistency
    if not round(Ts / ts_sim, 4).is_integer():
        raise ValueError(
            "The sample time Ts has to be an integer multiple of the simulation time step ts_sim"
        )
    if not round(Tf / ts_sim, 4).is_integer():
        raise ValueError(
            "The simulation time Tf has to be an integer multiple of the simulation time step ts_sim"
        )

    # compute the number of simulation steps
    N_steps = int(round(Tf / ts_sim, 4))

    # compute the number of steps for the discrete-time part of the loop
    N_steps_dt = int(round(Tf / Ts, 4))

    # number of simulation steps every which to update the discrete-time part of the loop
    n_update = int(round(Ts / ts_sim, 4))

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
