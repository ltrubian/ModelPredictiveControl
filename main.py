import pprint
from script.loop_simulations import (
    closed_loop_simulation_extended,
    closed_loop_simulation,
)
import numpy as np
import argparse

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
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01]), "R": 0.01},
        ],
    },
    {
        "simulation": "task 2, default system",
        "function": "closed_loop_simulation",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "non-linear",
            "ref_type": "empty-2s",
            "Ts": 1,  # this allows to pass (unnecessary) the checks on controller sampling
        },
        "specific": [
            {"ts_sim": 1e-5, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-4, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-3, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-2, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-1, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-5, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-4, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-3, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-2, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-1, "integ_type_sim": "IRK"},
        ],
    },
    {
        "simulation": "task 2, system with spring",
        "function": "closed_loop_simulation",
        "common": {
            "ini_type": "off-balance",
            "ctrl_on": False,
            "mod_type_sim": "spring",
            "ref_type": "empty-2s",
            "Ts": 1,  # this allows to pass (unnecessary) the checks on controller sampling
        },
        "specific": [
            {"ts_sim": 1e-5, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-4, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-3, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-2, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-1, "integ_type_sim": "ERK"},
            {"ts_sim": 1e-5, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-4, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-3, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-2, "integ_type_sim": "IRK"},
            {"ts_sim": 1e-1, "integ_type_sim": "IRK"},
        ],
    },
    {
        "simulation": "task 2, default non-linear, higher sampling time",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "down", "ref_type": "swing-up", "Ts": 0.1, "N": 20},
        "specific": [
            {"integ_type_ocp": "ERK"},
            {"integ_type_ocp": "IRK"},
        ],
    },
    {
        "simulation": "task 3, NMPC vs linear MPC",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "up", "ref_type": "horizontal"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
    },
    {
        "simulation": "task 3, swing-up maneuver with linear MPC",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
    },
]

if __name__ == "__main__":
    SUMMARY: list = [
        f""" -s {x} -c [0-{len(EXPER[x]["specific"]) - 1}] {EXPER[x]["simulation"]} """
        for x in range(len(EXPER))
    ]
    parser = argparse.ArgumentParser(
        description="""It runs simulaitn of inverted pendulum in some predefined and tested scenarios""",
        epilog="\n".join(SUMMARY),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s", "--simulation", type=int, default=0, help="select simulation case"
    )
    parser.add_argument(
        "-c",
        "--case",
        type=int,
        default=0,
        help="select one of the available set of parameters",
    )
    args_parsed = parser.parse_args()

    TASK: int = args_parsed.simulation
    EX: int = args_parsed.case

    if not (0 <= TASK < len(EXPER)) or not (0 <= EX < len(EXPER[TASK]["specific"])):
        raise ValueError("check --help for valid indexes of simulations and cases")

    curr_exp = EXPER[TASK]
    args = curr_exp["common"] | curr_exp["specific"][EX]

    print(f"Simulation {curr_exp['simulation']} with parameters: ")
    pprint.pp(args)

    match curr_exp["function"]:
        case "closed_loop_simulation":
            closed_loop_simulation(**args, save_video=False)
        case "closed_loop_simulation_extended":
            closed_loop_simulation_extended(**args, save_video=False)
