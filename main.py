from script.loop_simulations import (
    closed_loop_simulation_extended,
    closed_loop_simulation,
)
import numpy as np

EXPER: list = [
    [
        {
            "function": "closed_loop_simulation",
            "common": {"ini_type": "down", "ref_type": "swing"},
            "specific": [
                {"R": 0.02},
                {"R": 0.02},
                {"R": 0.02},
            ],
        },
        {
            "function": "closed_loop_simulation_extended",
            "common": {"ini_type": "down", "ref_type": "swing"},
            "specific": [
                {"R": 0.02},
                {"R": 0.02},
                {"R": 0.02},
            ],
        },
    ],
    [
        {
            "function": "closed_loop_simulation_extended",
            "common": {"ini_type": "down", "ref_type": "swing"},
            "specific": [
                {"R": 0.02},
                {"R": 0.02},
                {"R": 0.02},
            ],
        }
    ],
    [],
]

if __name__ == "__main__":
    TASK: int = 0
    EX: int = 0
    curr_exp = EXPER[TASK][EX]
    args = curr_exp["common"] | curr_exp["specific"]

    match curr_exp["function"]:
        case "closed_loop_simulation":
            pass
        case "closed_loop_simulation_extended":
            pass
    closed_loop_simulation(
        ini_type="off-balance",
        mod_type_ocp="spring",
        mod_type_sim="spring",
        ctrl_on=False,
        integ_type_sim="IRK",
        save_video=False,
    )
    # closed_loop_simulation_extended(save_video=False)
