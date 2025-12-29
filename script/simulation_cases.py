from numpy.typing import NDArray
from typing import Literal, Tuple, get_args
import numpy as np
from utils import piecewise_constant

_REFERENCE_TYPE = Literal["swing-up", "horizontal"]


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
        case _:
            raise ValueError(f"{ref_type = } is not in {get_args(_REFERENCE_TYPE)}")

    # - add N samples at the end (replicas of the last sample) for reference look-ahead
    y_ref = np.vstack((y_ref, np.repeat(y_ref[-1].reshape(1, -1), N, axis=0)))
    return (y_ref, Tf)
