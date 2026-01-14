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
) -> Tuple[NDArray, NDArray, NDArray]:
    """Compute classical step information for every refstate
    Args:
        signal:     states values
        reference:  same dimensions of the signal
        Tsim:       Simulation sampling time
        stepindex:  index right before the reference changes values
    Returns:
        Overshoots:
    """
    n: int = signal.shape[1]
    step: NDArray = reference[stepindex + 1, :] - reference[stepindex, :]
    errors: NDArray = signal - reference
    abserrors: NDArray = np.abs(errors)

    peakindexes: NDArray = np.argmax(
        np.abs(signal[stepindex:, :] - reference[0, :]), axis=0
    )
    peak: NDArray = abserrors[(peakindexes + stepindex, range(n))]
    peakTimes: NDArray = (peakindexes + stepindex) * Tsim
    overshoots = peak / step

    return (peak, peakTimes, overshoots)
