import argparse
import pprint

import numpy as np

from script.loop_simulations import (
    closed_loop_simulation,
)
from script.analysis import stepinfo

SCENARIOS: list = [
    {
        "simulation": "task 1, default non-linear",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "non-linear",
        },
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.10},
            {"Q": np.diag([10, 10, 0.1, 0.1]), "R": 0.20},
        ],
    },
    {
        "simulation": "task 1, extended controlloer",
        "common": {
            "ini_type": "down",
            "ref_type": "swing-up",
            "mod_type_ocp": "extended",
        },
        "specific": [
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.01]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.10]), "R": 0.01},
            {"Q": np.diag([10, 10, 0.1, 0.1, 0.20]), "R": 0.01},
        ],
    },
    {
        "simulation": "task 2, (only) simulate default system",
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
        "common": {"ini_type": "down", "ref_type": "swing-up", "Ts": 0.1, "N": 20},
        "specific": [
            {"integ_type_ocp": "ERK"},
            {"integ_type_ocp": "IRK"},
        ],
    },
    {
        "simulation": "task 3, NMPC vs linear MPC",
        "common": {"ini_type": "up", "ref_type": "horizontal"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
    },
    {
        "simulation": "task 3, swing-up maneuver with linear MPC",
        "common": {"ini_type": "down", "ref_type": "swing-up"},
        "specific": [
            {"mod_type_ocp": "linear"},
            {"mod_type_ocp": "non-linear"},
        ],
    },
]

if __name__ == "__main__":
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
        "-s",
        "--simulation",
        type=int,
        default=0,
        help="select simulation to run",
        choices=list(range(len(SCENARIOS))),
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

    (simX, simU, y_ref, cpt, cpt_sim, n_update, N, ts_sim) = closed_loop_simulation(
        **args, save_video=False
    )

    signal = simX[1:, :]
    reference = np.repeat(y_ref[: -N - 1, :-1], n_update, axis=0)
    stepindex = np.nonzero(np.ediff1d(reference[:, 1]))[0][0]
    (underpeak, underpeak_time, peak, peak_times, overshoots, rise_times) = stepinfo(
        signal, reference, ts_sim, stepindex
    )
    print(
        f"{peak = } \n{peak_times = } \n {overshoots = } \n{rise_times = } \n{underpeak = } \n{underpeak_time = } \n"
    )
