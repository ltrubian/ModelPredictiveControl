from script.loop_simulations import (
    closed_loop_simulation_extended,
    closed_loop_simulation,
)
import numpy as np

EXPER: list = [
    {
        "simulation": "task 1, default non-linear",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
        ],
    },
    {
        "simulation": "task 1, extended controlloer",
        "function": "closed_loop_simulation_extended",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.05]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.05]), "R": 0.01},
        ],
    },
    {
        "simulation": "task 2, modified system",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "off-balance", "ctrl_on": False},
        "specific": [{"integ_type_sim": "ERK"}, {"integ_type_sim": "IRK"}],
    },
]

if __name__ == "__main__":
    TASK: int = 1
    EX: int = 0
    curr_exp = EXPER[TASK]
    args = curr_exp["common"] | curr_exp["specific"][EX]

    match curr_exp["function"]:
        case "closed_loop_simulation":
            closed_loop_simulation(**args, save_video=False)
        case "closed_loop_simulation_extended":
            closed_loop_simulation_extended(**args, save_video=False)
