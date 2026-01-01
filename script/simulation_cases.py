from numpy.typing import NDArray
from typing import Literal, Tuple, get_args
import numpy as np
from .utils import piecewise_constant

_REFERENCE_TYPE = Literal["swing-up", "horizontal", "empty-2s"]
_INITIAL_TYPE = Literal["up", "down", "off-balance"]


def get_reference(
    Ts: float, N: int, nx: int, nu: int, ref_type: _REFERENCE_TYPE
) -> Tuple[NDArray, int]:
    """Return preset references

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
