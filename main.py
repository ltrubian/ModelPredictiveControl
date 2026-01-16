import argparse
import pprint
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from scenarios import SCENARIOS
from script.analysis import stepinfo
from script.loop_simulations import (
    closed_loop_simulation,
)
from script.plot_utils import plot_cpt, plot_results

if __name__ == "__main__":
    help_summary: list = [
        f""" -s {x:>2} -c [0-{len(SCENARIOS[x]["specific"]) - 1}] -> {SCENARIOS[x]["simulation"]}\n"""
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
    parser.add_argument(
        "-a",
        "--analysis",
        action="store_true",
        default=False,
        help="make comparison of all the cases for the simulation 's' ('c' will be ignored)",
    )
    args_parsed = parser.parse_args()
    simulation: int = args_parsed.simulation
    case: int = args_parsed.case
    analysis = args_parsed.analysis

    if not (0 <= simulation < len(SCENARIOS)) or not (
        0 <= case < len(SCENARIOS[simulation]["specific"])
    ):
        raise ValueError("check --help for valid indexes of simulations and cases")

    # set cases to be run: all or just the one selected
    curr_exp = SCENARIOS[simulation]
    if not analysis:
        curr_exp["specific"] = [curr_exp["specific"][case]]
        curr_exp["labels"] = [curr_exp["labels"][case]]
    n_exp = len(curr_exp["specific"])

    # array to collect all simulations results
    expX = np.zeros(1)
    expU = np.zeros(1)
    expR = np.zeros(1)
    expCtime = np.zeros(1)
    expStime = np.zeros(1)
    Ts_sim = np.zeros(n_exp)

    ########## SIMULATIONS ##########
    for jj in range(n_exp):
        args = curr_exp["common"] | curr_exp["specific"][jj]
        print(f"Simulation {curr_exp['simulation']} with parameters: ")
        pprint.pp(args)

        (simX, simU, y_ref, cpt, cpt_sim, n_update, N, ts_sim) = closed_loop_simulation(
            **args, save_video=False
        )
        # correctly initialize the vectors
        if expX.shape[0] < 2:
            expR = y_ref[:-N, :4]
            expX = np.zeros((simX.shape[0], 4, n_exp))
            expU = np.zeros((simU.shape[0], n_exp))
            expCtime = np.zeros((simU.shape[0], n_exp))
            expStime = np.zeros((simX.shape[0] - 1, n_exp))

        expX[:, :, jj] = np.vstack(
            (
                np.repeat(
                    simX[:-1, :4], (expX.shape[0] - 1) / (simX.shape[0] - 1), axis=0
                ),
                simX[-1, :4],
            )
        )
        expU[:, jj : 1 + jj] = simX[:-1:n_update, 4:] if simX.shape[1] > 4 else simU
        expCtime[:, jj] = cpt
        expStime[:, jj] = np.repeat(cpt_sim, expStime.shape[0] / cpt_sim.shape[0])
        Ts_sim[jj] = ts_sim

    control_on: bool = (
        False
        if ("Ts" in curr_exp["common"].keys()) and (curr_exp["common"]["Ts"] >= 1)
        else True
    )
    Tf = Ts_sim[0] * (expX.shape[0] - 1)
    time_dt = np.linspace(0, Tf, expU.shape[0] + 1)
    time = np.linspace(0, Tf, expX.shape[0])
    ########## NUMERICAL STATES PERFORMANCE ##########
    if control_on:
        str_res = (
            "underpeak",
            "Upeak time",
            "peak",
            "peak time",
            "overshoot",
            "rise time",
            "settl time",
        )

        for state in range(4):
            multdeg = 180 / np.pi if state % 2 else 1

            signal = expX[1:, state, :] * multdeg
            reference = np.repeat(expR[:-1, state], n_update, axis=0) * multdeg
            reference = np.repeat(reference[:, np.newaxis], n_exp, axis=1)

            diffs = np.nonzero(np.ediff1d(reference[:, 0]))[0]
            stepindex = diffs[0] if len(diffs) != 0 else 0
            results: Tuple = stepinfo(signal, reference, ts_sim, stepindex)

            print(f"{' state ' + str(state) + ' ':#^20}")
            print(f"{'labels':>15}", end="")
            for stat in str_res:
                print(f"{stat:>10} ", end="")
            print("")
            for i in range(signal.shape[1]):
                print(f"{curr_exp['labels'][i]:>15}", end="")
                for stat in results:
                    print(f"{stat[i]:10.2f} ", end="")
                print("")

    ########## PLOTS ##########
    try:
        plot_results(
            time,
            time_dt,
            expX,
            expU,
            expR,
            labels=curr_exp["labels"],
            ctrl_on=control_on,
        )

        plot_cpt(time_dt, expCtime, Tf / expU.shape[0], curr_exp["labels"])
        plot_cpt(time, expStime, ts_sim, curr_exp["labels"])
        plt.show()
    except KeyboardInterrupt:
        pass
