from typing import Tuple
import numpy as np
from numpy.typing import NDArray


# reference: NDArray = np.repeat(reference, round(Ts / Tsim))
# np.nonzero(np.ediff1d(array[:, state]))[0]
def stepinfo(
    signal: NDArray,
    reference: NDArray,
    Tsim: float,
    stepindex: int,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
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

    """
    n: int = signal.shape[1]
    start_v = reference[stepindex, :]
    end_v = reference[stepindex + 1, :]
    step: NDArray = np.abs(end_v - start_v)

    abserrors: NDArray = np.abs(signal - reference)

    # rise time
    en_tr = np.argmax((abserrors > 0) & (abserrors <= step * 0.1), axis=0)
    st_tr = np.argmax((abserrors > 0) & (abserrors <= step * 0.9), axis=0)
    rise_times = (en_tr - st_tr) * Tsim
    # underpeak: movement in opposite direction wrt final destination
    underpeak_index = np.argmax(np.abs(signal - end_v), axis=0)
    underpeak_time = underpeak_index * Tsim
    underpeak = signal[(underpeak_index, range(n))] - start_v

    # overshoot in absolute value and percentage
    peakindexes: NDArray = np.argmax(np.abs(signal - start_v), axis=0)
    peak: NDArray = abserrors[(peakindexes, range(n))]
    peak_times: NDArray = peakindexes * Tsim
    overshoots = np.where(step != 0, peak / step * 100, np.inf)

    # settling time
    print(
        f"ok = {
            (
                signal.shape[0]
                - np.argmax(np.abs(np.flipud(signal) - end_v) > 0.02 * step, axis=0)
            )
            * Tsim
        } \n {
            signal[
                signal.shape[0]
                - np.argmax(np.abs(np.flipud(signal) - end_v) > 0.02 * step, axis=0)
                - 1,
                :,
            ]
        }"
    )

    return (underpeak, underpeak_time, peak, peak_times, overshoots, rise_times)
