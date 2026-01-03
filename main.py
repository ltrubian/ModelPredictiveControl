import argparse
import pprint

import numpy as np

from script.loop_simulations import (
    closed_loop_simulation,
    closed_loop_simulation_extended,
)

from script.utils import _REFERENCE_TYPE

SCENARIOS: list = [
    {
        "simulation": "task 1, default non-linear",
        "function": "closed_loop_simulation",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.05},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.10},
        ],
    },
    {
        "simulation": "task 1, extended controlloer",
        "function": "closed_loop_simulation_extended",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.05]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.10]), "R": 0.01},
        ],
    },
    {
        "simulation": "task 2, (only) simulate default system",
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
        "simulation": "task 2, (only) simulate system with spring",
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
    print(f"{type(_REFERENCE_TYPE)}")
    help_summary: list = [
        f""" -s {x} -c [0-{len(SCENARIOS[x]["specific"]) - 1}] -> {SCENARIOS[x]["simulation"]}\n"""
        f"""\t\t varying: {list(SCENARIOS[x]["specific"][0].keys())}"""
        for x in range(len(SCENARIOS))
    ]
    parser = argparse.ArgumentParser(
        description="""It runs simulaitn of inverted pendulum in some predefined and tested scenarios""",
        epilog="Here the possible choices of arguments\n" + "\n".join(help_summary),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-s", "--simulation", type=int, default=0, help="select simulation to run"
    )
    parser.add_argument(
        "-c",
        "--case",
        type=int,
        default=0,
        help="select one of the available set of parameters",
    )
    args_parsed = parser.parse_args()

    simulation: int = args_parsed.simulation
    case: int = args_parsed.case

    if not (0 <= simulation < len(SCENARIOS)) or not (
        0 <= case < len(SCENARIOS[simulation]["specific"])
    ):
        raise ValueError("check --help for valid indexes of simulations and cases")

    curr_exp = SCENARIOS[simulation]
    args = curr_exp["common"] | curr_exp["specific"][case]

    print(f"Simulation {curr_exp['simulation']} with parameters: ")
    pprint.pp(args)

    match curr_exp["function"]:
        case "closed_loop_simulation":
            closed_loop_simulation(**args, save_video=False)
        case "closed_loop_simulation_extended":
            closed_loop_simulation_extended(**args, save_video=False)
