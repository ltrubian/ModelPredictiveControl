import matplotlib.pyplot as plt
import numpy as np
from acados_template import (
    AcadosOcpSolver,
    AcadosSimSolver,
)
from tqdm import tqdm

from script.inverted_pendulum_model import get_inverted_pendulum_model
from script.plot_utils import (
    inverted_pendulum_animation,
    plot_cpt,
    plot_pred_traj,
    plot_results,
)
from script.simulation_cases import (
    get_initial_condition,
    get_reference,
    _INITIAL_TYPE,
    _REFERENCE_TYPE,
)
from script.solvers_description import (
    create_ocp_solver_description,
    create_sim_solver_description,
)
from script.utils import compute_num_steps


def closed_loop_simulation(
    shifting: bool = False,
    ref_preview: bool = False,
    save_video=False,
    Q=np.diag([10, 10, 0.1, 0.1, 0.01]),
    R=0.01,
    ini_type: _INITIAL_TYPE = "down",
    ref_type: _REFERENCE_TYPE = "swing-up",
):
    # define simulation fundamental time step [s]
    ts_sim = 0.001

    # model used to simulate the system
    sim_model = get_inverted_pendulum_model()

    # setup controller parameters
    model = get_inverted_pendulum_model(type="extended")
    # - controller sample time [s]
    Ts = 0.02
    # - number of shooting time intervals
    N = 100
    # - prediction horizon length [s]
    T = N * Ts

    # get state and control dimensions
    nx, nu = model.x.rows(), model.u.rows()

    # initial condition
    x0 = get_initial_condition(nx, ini_type)

    # define reference
    # - define reference for the angle and, accordingly, the simulation time Tf
    y_ref, Tf = get_reference(Ts, N, nx, nu, ref_type)
    # compute the number of steps for simulation
    N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

    acados_integrator = AcadosSimSolver(
        create_sim_solver_description(sim_model, ts_sim), verbose=False
    )

    # create OCP solver
    ocp = create_ocp_solver_description(
        model,
        N,
        T,
        x0,
        apply_state_constraints=True,
        integrator_type="ERK",
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
    simU = np.zeros((N_steps_dt + 1, nu))
    deltaU = np.zeros((N_steps_dt, nu))
    # set intial state
    simX[0, :] = x0

    # create variables to store, at each iteration, previous optimal solution
    x_opt = np.zeros((N + 1, nx, N_steps_dt))
    u_opt = np.zeros((N, nu, N_steps_dt))

    # variable to store total CPU time
    cpt = np.zeros((N_steps_dt,))

    # variable to store solver status
    status = np.zeros((N_steps_dt,))

    simU[0, :] = x0[4:]
    # simulation loop
    for i in tqdm(
        range(N_steps), desc="Simulation", ascii=False, ncols=75, colour="green"
    ):
        # check whether to update the discrete-time part of the loop
        if i % n_update == 0:
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

            # update the control
            deltaU[k, :] = acados_ocp_solver.solve_for_x0(
                simX[i, :], fail_on_nonzero_status=False, print_stats_on_failure=False
            )

            # store CPU time required for solving the problem
            cpt[k] = acados_ocp_solver.get_stats("time_tot")

            # store solver status
            status[k] = acados_ocp_solver.get_status()

            # store optimal solution
            for stage in range(N):
                x_opt[stage, :, k] = acados_ocp_solver.get(stage, "x")
                u_opt[stage, :, k] = acados_ocp_solver.get(stage, "u")

            x_opt[N, :, k] = acados_ocp_solver.get(N, "x")

            # update next input via integration
            simU[k + 1, :] = simU[k, :] + Ts * deltaU[k, :]

            # update discrete-time iteration counter
            k += 1

        # simulate system
        simX[i + 1, 0:4] = acados_integrator.simulate(simX[i, 0:4], simU[k - 1, :])
        simX[i + 1, 4:] = simU[k - 1, :]

    # visualize results
    print("Average total CPU time: " + str(np.mean(cpt) * 1000) + " ms")

    time = np.linspace(0, ts_sim * N_steps, N_steps + 1)
    time_dt = np.linspace(0, Ts * N_steps_dt, N_steps_dt + 1)

    nonzero_status = np.argwhere(status != 0)

    if nonzero_status.size != 0:
        print("\nSolver returned non-zero status at the following iterations:")
        for k in nonzero_status:
            print(
                f"* k = {int(k.item())} [t = {time_dt[k].item():.3f} s] - status {int(status[k].item())}"
            )

    try:
        plot_results(time, time_dt, simX, simU[:-1, :], y_ref)
        plot_cpt(time_dt, cpt, Ts)

        plot_pred_traj(
            time,
            time_dt,
            simX,
            simU[:-1, :],
            x_opt,
            u_opt,
            np.argwhere(np.round(time_dt, 3) == 5),
        )

        if save_video:
            inverted_pendulum_animation(simX[:, 0], simX[:, 1], ts_sim)

        plt.show()

    except KeyboardInterrupt:
        pass
