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
        },
        "specific": [
            {"t_sim": 1e-5, "integ_type_sim": "ERK"},
            {"t_sim": 1e-4, "integ_type_sim": "ERK"},
            {"t_sim": 1e-3, "integ_type_sim": "ERK"},
            {"t_sim": 1e-2, "integ_type_sim": "ERK"},
            {"t_sim": 1e-1, "integ_type_sim": "ERK"},
            {"t_sim": 1e-5, "integ_type_sim": "IRK"},
            {"t_sim": 1e-4, "integ_type_sim": "IRK"},
            {"t_sim": 1e-3, "integ_type_sim": "IRK"},
            {"t_sim": 1e-2, "integ_type_sim": "IRK"},
            {"t_sim": 1e-1, "integ_type_sim": "IRK"},
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
        },
        "specific": [
            {"t_sim": 1e-5, "integ_type_sim": "ERK"},
            {"t_sim": 1e-4, "integ_type_sim": "ERK"},
            {"t_sim": 1e-3, "integ_type_sim": "ERK"},
            {"t_sim": 1e-2, "integ_type_sim": "ERK"},
            {"t_sim": 1e-1, "integ_type_sim": "ERK"},
            {"t_sim": 1e-5, "integ_type_sim": "IRK"},
            {"t_sim": 1e-4, "integ_type_sim": "IRK"},
            {"t_sim": 1e-3, "integ_type_sim": "IRK"},
            {"t_sim": 1e-2, "integ_type_sim": "IRK"},
            {"t_sim": 1e-1, "integ_type_sim": "IRK"},
        ],
    },
    {
        "simulation": "task 2, default non-linear, higher sampling time",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "down", "ref_type": "swing-up", "Ts": 0.1, "N": 20},
        "specific": [
            {"integ_type_ocp": "IRK"},
            {"integ_type_ocp": "ERK"},
        ],
    },
]

if __name__ == "__main__":
    SUMMARY: list = [
        f""" {x}) {EXPER[x]["simulation"]} cases [0-{len(EXPER[x]["specific"])}]"""
        for x in range(len(EXPER))
    ]
    parser = argparse.ArgumentParser(
        prog="Interted pendulum, simulation and control",
        description="""It runs simulaitn of inverted pendulum in some predefined and testes scenarios""",
        epilog="\n".join(SUMMARY),
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
    curr_exp = EXPER[TASK]
    args = curr_exp["common"] | curr_exp["specific"][EX]

    print(f"Simulation {curr_exp['simulation']} with parameters: ")
    pprint.pp(args)

    match curr_exp["function"]:
        case "closed_loop_simulation":
            closed_loop_simulation(**args, save_video=False)
        case "closed_loop_simulation_extended":
            closed_loop_simulation_extended(**args, save_video=False)
