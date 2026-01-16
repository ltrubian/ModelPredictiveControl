from typing import Tuple

import numpy as np
from acados_template import (
    AcadosOcpSolver,
    AcadosSimSolver,
)
from numpy.typing import NDArray
from tqdm import tqdm

from script.inverted_pendulum_model import _MODEL_TYPE, get_inverted_pendulum_model
from script.solvers_description import (
    _INTEGRATOR_TYPE,
    create_ocp_solver_description,
    create_sim_solver_description,
)
from script.utils import (
    _INITIAL_TYPE,
    _REFERENCE_TYPE,
    compute_num_steps,
    get_initial_condition,
    get_reference,
)


def closed_loop_simulation(
    shifting: bool = False,
    ref_preview: bool = False,
    save_video: bool = False,
    ctrl_on: bool = True,
    ts_sim: float = 0.001,
    # - controller sample time [s]
    Ts: float = 0.02,
    # - number of shooting time intervals
    N: int = 100,
    # define simulation fundamental time step [s]
    Q: NDArray = np.diag([10, 10, 0.1, 0.1]),
    R: float = 0.01,
    ini_type: _INITIAL_TYPE = "down",
    ref_type: _REFERENCE_TYPE = "swing-up",
    mod_type_sim: _MODEL_TYPE = "non-linear",
    mod_type_ocp: _MODEL_TYPE = "non-linear",
    integ_type_sim: _INTEGRATOR_TYPE = "ERK",
    integ_type_ocp: _INTEGRATOR_TYPE = "ERK",
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, int, int, float]:
    """Runs a closede loop simulatoin with the 'extended' controller.

    Args:
        shifting:       shifting of the state for the ocp solver
        ref_preview:    allow controller reference preview
        save_video:     save video of the system
        ctrl_on:        activate controller
        ts_sim:         sampling time of the simulation
        Ts:             sampling time of the controller
        N:              number of shooting intervals
        Q:              state weight
        R:              input weight
        ini_type:       initial condition (up/down/off-balance)
        ref_type:       reference type (swing-up/horizontal/empty)
        mod_type_sim:   model type to simulate
        mod_type_ocp:   model type used by the controller
        integ_type_sim: integrator type (IRK/ERK) for the simulation solver
        integ_type_ocp: integrator type (IRK/ERK) for the controller solver
    """

    # model used to simulate the system
    sim_model = get_inverted_pendulum_model(type=mod_type_sim)

    # setup controller parameters
    model = get_inverted_pendulum_model(type=mod_type_ocp)
    # - prediction horizon length [s]
    T = N * Ts

    # get state and control dimensions
    nx, nu = model.x.rows(), model.u.rows()

    # initial condition
    x0 = get_initial_condition(nx, ini_type)

    # define reference
    y_ref, Tf = get_reference(Ts, N, nx, nu, ref_type)
    # compute the number of steps for simulation
    N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

    acados_integrator = AcadosSimSolver(
        create_sim_solver_description(
            sim_model, ts_sim, integrator_type=integ_type_sim
        ),
        verbose=False,
    )

    # create OCP solver
    ocp = create_ocp_solver_description(
        model,
        N,
        T,
        x0,
        constraints="input-U" if mod_type_ocp != "extended" else "state-F",
        integrator_type=integ_type_ocp,
        Q=Q,
        R=R,
    )
    acados_ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    # initialize solver
    for stage in range(N):
        acados_ocp_solver.set(stage, "x", x0)
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    acados_ocp_solver.set(N, "x", x0)

    # define iteration counter for the discrete-time part of the control loop
    k = 0

    # create variables to store state and control trajectories
    simX = np.zeros((N_steps + 1, nx))
    simU = np.zeros((N_steps_dt, nu))
    # set intial state
    simX[0, :] = x0

    ############ EXTENDED START ############
    if mod_type_ocp == "extended":
        # set initial input, PURE EXTENDED
        inputk1 = simX[0, 4:]
    ############ EXTENDED END ##############

    # create variables to store, at each iteration, previous optimal solution
    x_opt = np.zeros((N + 1, nx, N_steps_dt))
    u_opt = np.zeros((N, nu, N_steps_dt))

    # variable to store total CPU time
    cpt = np.zeros((N_steps_dt,))
    cpt_sim = np.zeros((N_steps,))

    # variable to store solver status
    status = np.zeros((N_steps_dt,))

    # simulation loop
    for i in tqdm(
        range(N_steps), desc="Simulation", ascii=False, ncols=75, colour="green"
    ):
        # check whether to update the discrete-time part of the loop
        if i % n_update == 0 and ctrl_on:
            # update reference
            for j in range(N):
                acados_ocp_solver.set(
                    j, "yref", y_ref[k + (j if ref_preview else 0), :]
                )
            acados_ocp_solver.set(
                N, "yref", y_ref[k + (N if ref_preview else 0), 0:-nu]
            )

            # if performing shifting, explicitly initialize solver
            # (otherwise, it will be automatically intialized with the previous solution)
            if shifting and k > 0:
                for stage in range(N):
                    acados_ocp_solver.set(stage, "x", x_opt[stage + 1, :, k - 1])
                    acados_ocp_solver.set(
                        stage, "u", u_opt[min([stage + 1, N - 1]), :, k - 1]
                    )

                acados_ocp_solver.set(N, "x", x_opt[N, :, k - 1])

            ############ EXTENDED START ############
            if mod_type_ocp == "extended":
                # update the control for EXTENDED
                simX[i, 4:] = inputk1
                acados_ocp_solver.set(0, "x", simX[i, :])
            ############ EXTENDED END ##############

            # update the control
            simU[k, :] = acados_ocp_solver.solve_for_x0(
                simX[i, :],
                fail_on_nonzero_status=False,
                print_stats_on_failure=False,
            )

            ############ EXTENDED START ############
            if mod_type_ocp == "extended":
                # update next input via integration, PURE EXTENDED
                inputk1 = simX[i, 4:] + Ts * simU[k, :]
            ############ EXTENDED END ##############

            # store CPU time required for solving the problem
            cpt[k] = acados_ocp_solver.get_stats("time_tot")

            # store solver status
            status[k] = acados_ocp_solver.get_status()

            # store optimal solution
            for stage in range(N):
                x_opt[stage, :, k] = acados_ocp_solver.get(stage, "x")
                u_opt[stage, :, k] = acados_ocp_solver.get(stage, "u")

            x_opt[N, :, k] = acados_ocp_solver.get(N, "x")

            # update discrete-time iteration counter
            k += 1

        ############ EXTENDED START ############
        if mod_type_ocp == "extended":
            # simulate system
            simX[i + 1, 0:4] = acados_integrator.simulate(simX[i, 0:4], simX[i, 4:])
            # unpdate the state with the actual input, PURE EXTENDED
            simX[i + 1, 4:] = simX[i, 4:]
        else:
            # simulate system
            simX[i + 1, :] = acados_integrator.simulate(simX[i, :], simU[k - 1, :])
        ############ EXTENDED END ##############

        cpt_sim[i] = acados_integrator.get("CPUtime")

    # visualize results
    print("Average total controller CPU time: " + str(np.mean(cpt) * 1000) + " ms")
    print("Average total simulation CPU time: " + str(np.mean(cpt_sim) * 1000) + " ms")

    time_dt = np.linspace(0, Ts * N_steps_dt, N_steps_dt + 1)

    nonzero_status = np.argwhere(status != 0)

    if nonzero_status.size != 0:
        print("\nSolver returned non-zero status at the following iterations:")
        for k in nonzero_status:
            print(
                f"* k = {int(k.item())} [t = {time_dt[k].item():.3f} s] - status {int(status[k].item())}"
            )

    return (simX, simU, y_ref, cpt, cpt_sim, n_update, N, ts_sim)

    # time = np.linspace(0, ts_sim * N_steps, N_steps + 1)
    # try:
    #    plot_results(time, time_dt, simX, simU, y_ref, ctrl_on=ctrl_on)
    #    if ctrl_on:
    #        plot_cpt(time_dt, cpt, Ts)
    #        plot_pred_traj(
    #            time,
    #            time_dt,
    #            simX,
    #            simU,
    #            x_opt,
    #            u_opt,
    #            np.argwhere(np.round(time_dt, 3) == 5),
    #        )
    #    else:
    #        plot_cpt(time, cpt_sim, ts_sim)

    #    if save_video:
    #        inverted_pendulum_animation(simX[:, 0], simX[:, 1], ts_sim)

    #    plt.show()

    # except KeyboardInterrupt:
    #    pass
