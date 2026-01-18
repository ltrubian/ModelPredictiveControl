import numpy as np
from numpy.typing import NDArray


def stepinfo(
    signal: NDArray,
    reference: NDArray,
    Tsim: float,
    stepindex: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Compute classical step information for every refstate
    Args:
        signal:     states values
        reference:  same dimensions of the signal
        Tsim:       Simulation sampling time
        stepindex:  index right before the reference changes values
    Returns:
        underpeak:  max movement in "opposite" direction
        underpeak_time: time instant of underpeak
        peak:       max movement over the target
        peak_time:  time instant of peak
        overshoot:  peak relative to the step
        rise_time:  how much time is necessary to go from 90% to 10% error
        settling_time:  first t such abs(errors) < 2%

    """
    n: int = signal.shape[1]
    start_v = reference[stepindex, :]
    end_v = reference[stepindex + 1, :]
    step: NDArray = np.abs(end_v - start_v)

    abserrors: NDArray = np.abs(signal - reference)

    # rise time
    en_tr = np.argmax((abserrors > 0) & (abserrors <= step * 0.1), axis=0)
    st_tr = np.argmax((abserrors > 0) & (abserrors <= step * 0.9), axis=0)
    rise_time = (en_tr - st_tr) * Tsim
    # underpeak: movement in opposite direction wrt final destination
    underpeak_index = np.argmax(np.abs(signal - end_v), axis=0)
    underpeak_time = underpeak_index * Tsim
    underpeak = signal[(underpeak_index, range(n))] - start_v

    # overshoot in absolute value and percentage
    peakindexe: NDArray = np.argmax(np.abs(signal - start_v), axis=0)
    peak: NDArray = abserrors[(peakindexe, range(n))]
    peak_time: NDArray = peakindexe * Tsim
    overshoot = np.repeat(np.inf, n)
    nz = np.nonzero(step)
    overshoot[nz] = peak[nz] / step[nz] * 100

    # settling time
    settling_time: NDArray = (
        signal.shape[0]
        - np.argmax(np.abs(np.flipud(signal) - end_v) > 0.02 * step, axis=0)
        - 1
    ) * Tsim

    return (
        underpeak,
        underpeak_time,
        peak,
        peak_time,
        overshoot,
        rise_time,
        settling_time,
    )
