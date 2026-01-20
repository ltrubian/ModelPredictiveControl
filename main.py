import argparse
import pprint

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from scenarios import SCENARIOS
from script.analysis import stepinfo
from script.loop_simulations import (
    closed_loop_simulation,
)
from script.plot_utils import plot_cpt, plot_pred_traj, plot_results

if __name__ == "__main__":
    help_summary: list = [
        f""" -s {x:>2} -c [0-{len(SCENARIOS[x]["specific"]) - 1}] """
        f"""-> {SCENARIOS[x]["simulation"]}\n"""
        f"""\t\t varying: {list(SCENARIOS[x]["specific"][0].keys())}"""
        for x in range(len(SCENARIOS))
    ]
    parser = argparse.ArgumentParser(
        description="""It runs simulaitn of inverted pendulum in some """
        """predefined and tested scenarios""",
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
        help="""make comparison of all the cases for the simulation 's'"""
        """('c' will be ignored)""",
    )
    parser.add_argument(
        "--save-fig",
        action="store_true",
        default=False,
        help="""save figures in ./images/sim_<simulation>/""",
    )
    args_parsed = parser.parse_args()
    simulation: int = args_parsed.simulation
    case: int = args_parsed.case
    analysis: bool = args_parsed.analysis

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

    # #################################
    # ########## SIMULATIONS ##########
    # #################################

    # array to collect all simulations results
    expX = np.zeros(1)
    expU = np.zeros(1)
    expR = np.zeros(1)
    expX_opt = np.zeros(1)
    expU_opt = np.zeros(1)
    expCtime = np.zeros(1)
    expStime = np.zeros(1)
    Ts_sim = np.zeros(n_exp)

    for jj in range(n_exp):
        args = curr_exp["common"] | curr_exp["specific"][jj]
        print(f"Simulation {curr_exp['simulation']} with parameters: ")
        pprint.pp(args)

        (simX, simU, y_ref, x_opt, u_opt, cpt, cpt_sim, n_update, N, ts_sim) = (
            closed_loop_simulation(**args, save_video=False)
        )
        # correctly initialize the vectors
        if expX.shape[0] < 2:
            expR = y_ref[:-N, :4]
            expX = np.zeros((simX.shape[0], 4, n_exp))
            expU = np.zeros((simU.shape[0], n_exp))
            expCtime = np.zeros((simU.shape[0], n_exp))
            expStime = np.zeros((simX.shape[0] - 1, n_exp))
            expX_opt = np.zeros((x_opt.shape[0], 4, x_opt.shape[2], n_exp))

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
        expX_opt[:, :, :, jj] = x_opt[:, :4, :]
        Ts_sim[jj] = ts_sim

    # ##################################################
    # ########## NUMERICAL STATES PERFORMANCE ##########
    # ##################################################
    control_on: bool = (
        False
        if ("Ts" in curr_exp["common"].keys()) and (curr_exp["common"]["Ts"] >= 1)
        else True
    )
    Tf = Ts_sim[0] * (expX.shape[0] - 1)
    time_dt = np.linspace(0, Tf, expU.shape[0] + 1)
    time = np.linspace(0, Tf, expX.shape[0])
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
        stepstate: int = 1 if curr_exp["common"]["ref_type"] == "swing-up" else 0
        diffs: NDArray = np.nonzero(
            np.ediff1d(np.repeat(expR[:-1, stepstate], n_update, axis=0))
        )[0]
        stepindex: int = diffs[0] if len(diffs) != 0 else 0

        for state in range(4):
            multdeg: float = 180 / np.pi if state % 2 else 1

            signal: NDArray = expX[1:, state, :] * multdeg
            reference: NDArray = np.repeat(expR[:-1, state], n_update, axis=0) * multdeg
            reference: NDArray = np.repeat(reference[:, np.newaxis], n_exp, axis=1)

            if len(diffs) > 1:
                signal: NDArray = signal[: diffs[1] - 1, :]
                reference: NDArray = reference[: diffs[1] - 1, :]
            results: tuple = stepinfo(signal, reference, ts_sim, stepindex)

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

    # ###########################
    # ########## PLOTS ##########
    # ###########################
    try:
        xlimits: tuple | None = (
            (4.5, 10) if curr_exp["common"]["ref_type"] == "swing-up" else None
        )

        folder = f"./images/sim_{simulation:0>2}/"
        save: bool = args_parsed.save_fig
        plot_results(
            time,
            time_dt,
            expX,
            expU,
            expR,
            labels=curr_exp["labels"],
            ctrl_on=control_on,
            folder=folder,
            xlimits=xlimits,
            save=save,
        )

        if control_on:
            plot_cpt(
                time_dt,
                expCtime,
                None,
                curr_exp["labels"],
                folder=folder,
                prefix="ctrl_",
                xlimits=xlimits,
                save=save,
            )
            k: int = np.argwhere(np.round(time_dt, 3) == 5.5)[0]
            plot_pred_traj(
                time,
                time_dt,
                simX,
                simU,
                expX_opt[:, :4, k, :].reshape(-1, 4, n_exp),
                expU_opt,
                k,
                xlimits=(5, 7.5),
                save=save,
                folder=folder,
                labels=curr_exp["labels"],
                shooting_nodes=None,
            )
        plot_cpt(
            time,
            expStime,
            None,
            curr_exp["labels"],
            folder=folder,
            prefix="sim_",
            xlimits=xlimits,
            save=save,
        )
        plt.show()
    except KeyboardInterrupt:
        pass
