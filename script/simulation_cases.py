import numpy as np
from utils import piecewise_constant


def get_reference(Ts, nx, nu, N, type):
    angle_ref, Tf = piecewise_constant(np.array([np.pi, 0]), np.array([5, 10]), Ts)

    # - provide a reference for all variables
    y_ref = np.column_stack(
        (
            np.zeros((len(angle_ref), 1)),
            angle_ref.reshape(-1, 1),
            np.zeros((len(angle_ref), nx + nu - 2)),
        )
    )

    # - add N samples at the end (replicas of the last sample) for reference look-ahead
    y_ref = np.vstack((y_ref, np.repeat(y_ref[-1].reshape(1, -1), N, axis=0)))
    return (y_ref, Tf)
