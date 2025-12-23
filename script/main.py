from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from tqdm import tqdm

from inverted_pendulum_model import get_inverted_pendulum_model
from utils import piecewise_constant, compute_num_steps, get_nonuniform_grid
from plot_utils import (
    plot_results,
    plot_pred_traj,
    plot_cpt,
    plot_grid,
    inverted_pendulum_animation,
)


def create_ocp_solver_description(
    model, N, T, x0, shooting_nodes=None, apply_state_constraints=False
) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # define system dynamics model
    ocp.model = model

    # set prediction horizon:
    # tf - prediction horizon length [s]
    # N  - number of intervals in which the prediction horizon is divided
    ocp.solver_options.tf = T
    ocp.solver_options.N_horizon = N

    # NOTE: on older acados versions, use instead
    # ocp.dims.N = N

    # set shooting nodes, if provided
    # (otherwise set automatically assuming uniform grid)
    if shooting_nodes is not None:
        ocp.solver_options.shooting_nodes = shooting_nodes

    # get state, control and cost dimensions
    nx = model.x.rows()
    nu = model.u.rows()

    ny = nx + nu
    ny_e = nx

    # define cost type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # define cost weigth matrices
    Q = np.diag([10, 10, 0.1, 0.1])
    R = 0.01

    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_e = T / N * Q

    # define matrices characterizing the cost
    ocp.cost.Vx = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.Vx_e = np.eye(nx)

    # alternatively, for the NONLINEAR_LS cost type
    # ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    # ocp.model.cost_y_expr_e = model.x

    # initialize variables for reference
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # bounds on control input
    ocp.constraints.lbu = np.array([-20])
    ocp.constraints.ubu = np.array([20])
    ocp.constraints.idxbu = np.array([0])

    # if specified, apply the bounds on position with slack variables
    if apply_state_constraints:
        # bounds on position (0 component of state vector)
        ocp.constraints.idxbx = np.array([0])
        ocp.constraints.lbx = np.array([-1])
        ocp.constraints.ubx = np.array([1])

        # bounds on terminal state x_N
        ocp.constraints.idxbx_e = np.array([0])
        ocp.constraints.lbx_e = np.array([-1])
        ocp.constraints.ubx_e = np.array([1])

        # indices among the bounds on state for which use a slack variable
        ocp.constraints.idxsbx = np.array([0])
        ocp.constraints.idxsbx_e = np.array([0])

        # define weight on slack variables
        ocp.cost.Zl = np.array([1e4])
        ocp.cost.Zl_e = np.array([1e4])
        ocp.cost.Zu = np.array([1e4])
        ocp.cost.Zu_e = np.array([1e4])

        ocp.cost.zl = np.array([1e3])
        ocp.cost.zl_e = np.array([1e3])
        ocp.cost.zu = np.array([1e3])
        ocp.cost.zu_e = np.array([1e3])

    # initialize constraint on initial condition
    ocp.constraints.x0 = x0

    # set solver options
    ocp.solver_options.qp_solver = (
        "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM
    )
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"  # ERK, IRK
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI

    # to configure partial condensing
    # ocp.solver_options.qp_solver_cond_N = int(N/10)

    # some more advanced settings (refer to the documentation to see them all)
    # - maximum number of SQP iterations (default: 100)
    ocp.solver_options.nlp_solver_max_iter = 100
    # - maximum number of iterations for the QP solver (default: 50)
    ocp.solver_options.qp_solver_iter_max = 50

    # - configure warm start of the QP solver (0: no, 1: warm start, 2: hot start)
    # (depends on the specific solver)
    ocp.solver_options.qp_solver_warm_start = 0

    return ocp


def closed_loop_simulation(save_video=False):
    # define simulation fundamental time step [s]
    ts_sim = 0.001

    # model used to simulate the system
    sim_model = get_inverted_pendulum_model(l=0.8)
    sim_model.name = "sim_model"

    # initial condition
    x0 = np.array([0, np.pi, 0, 0])

    # setup controller parameters
    # - system model
    model = get_inverted_pendulum_model(type="linear")
    # - controller sample time [s]
    Ts = 0.02
    # - number of shooting time intervals
    N = 100
    # - prediction horizon length [s]
    T = N * Ts

    # get state and control dimensions
    nx, nu = model.x.rows(), model.u.rows()

    # define reference
    # - define reference for the angle and, accordingly, the simulation time Tf
    angle_ref, Tf = piecewise_constant(np.array([np.pi, 0]), np.array([5, 10]), Ts)

    # - provide a reference for all variables
    y_ref = np.column_stack(
        (
            np.zeros((len(angle_ref), 1)),
            angle_ref.reshape(-1, 1),
            np.zeros((len(angle_ref), nx + nu - 2)),
        )
    )

    # - add N samples at the end (replicas of the last sample) for reference look-ahead
    y_ref = np.vstack((y_ref, np.repeat(y_ref[-1].reshape(1, -1), N, axis=0)))

    # compute the number of steps for simulation
    N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

    # configure whether to apply shifting and to enable reference look-ahead
    shifting = False
    ref_preview = False

    # setup simulation of system dynamics
    sim = AcadosSim()
    sim.model = sim_model
    sim.solver_options.T = ts_sim
    sim.solver_options.integrator_type = "ERK"

    acados_integrator = AcadosSimSolver(sim, verbose=False)

    # create OCP solver
    ocp = create_ocp_solver_description(model, N, T, x0, apply_state_constraints=False)
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

    # create variables to store, at each iteration, previous optimal solution
    x_opt = np.zeros((N + 1, nx, N_steps_dt))
    u_opt = np.zeros((N, nu, N_steps_dt))

    # variable to store total CPU time
    cpt = np.zeros((N_steps_dt,))

    # variable to store solver status
    status = np.zeros((N_steps_dt,))

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
            simU[k, :] = acados_ocp_solver.solve_for_x0(
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

            # update discrete-time iteration counter
            k += 1

        # simulate system
        simX[i + 1, :] = acados_integrator.simulate(simX[i, :], simU[k - 1, :])

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
        plot_results(time, time_dt, simX, simU)
        plot_cpt(time_dt, cpt, Ts)

        plot_pred_traj(
            time,
            time_dt,
            simX,
            simU,
            x_opt,
            u_opt,
            np.argwhere(np.round(time_dt, 3) == 5),
        )

        if save_video:
            inverted_pendulum_animation(simX[:, 0], simX[:, 1], ts_sim)

        plt.show()

    except KeyboardInterrupt:
        pass


def closed_loop_simulation_nug(save_video=False):
    # define simulation fundamental time step [s]
    ts_sim = 0.001

    # model used to simulate the system
    sim_model = get_inverted_pendulum_model(l=0.8)
    sim_model.name = "sim_model"

    # initial condition
    x0 = np.array([0, np.pi, 0, 0])

    # setup controller parameters
    # - system model
    model = get_inverted_pendulum_model()
    # - controller sample time [s]
    Ts = 0.02
    # - prediction horizon length [s]
    T = 2
    # - indices of intermediate shooting nodes (between 0 and T) with respect to a uniform grid
    intermediate_nodes = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    # - shooting nodes
    shooting_nodes = get_nonuniform_grid(T, Ts, intermediate_nodes)
    # - number of shooting time intervals
    N = len(shooting_nodes) - 1

    # get state and control dimensions
    nx, nu = model.x.rows(), model.u.rows()

    # define reference
    # - define reference for the angle and, accordingly, the simulation time Tf
    angle_ref, Tf = piecewise_constant(np.array([np.pi, 0]), np.array([5, 10]), Ts)

    # - provide a reference for all variables
    y_ref = np.hstack(
        (
            np.zeros((len(angle_ref), 1)),
            angle_ref.reshape(-1, 1),
            np.zeros((len(angle_ref), nx + nu - 2)),
        )
    )

    # - add T/Ts samples at the end (replicas of the last sample) for reference look-ahead
    y_ref = np.vstack((y_ref, np.repeat(y_ref[-1].reshape(1, -1), int(T / Ts), axis=0)))

    # compute the number of steps for simulation
    N_steps, N_steps_dt, n_update = compute_num_steps(ts_sim, Ts, Tf)

    # configure whether to enable reference look-ahead
    ref_preview = False

    # setup simulation of system dynamics
    sim = AcadosSim()
    sim.model = sim_model
    sim.solver_options.T = ts_sim
    sim.solver_options.integrator_type = "ERK"

    acados_integrator = AcadosSimSolver(sim, verbose=False)

    # create OCP solver
    ocp = create_ocp_solver_description(model, N, T, x0, shooting_nodes)
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

    # create variables to store, at each iteration, previous optimal solution
    x_opt = np.zeros((N + 1, nx, N_steps_dt))
    u_opt = np.zeros((N, nu, N_steps_dt))

    # variable to store total CPU time
    cpt = np.zeros((N_steps_dt,))

    # variable to store solver status
    status = np.zeros((N_steps_dt,))

    # simulation loop
    for i in tqdm(
        range(N_steps), desc="Simulation", ascii=False, ncols=75, colour="green"
    ):
        # check whether to update the discrete-time part of the loop
        if i % n_update == 0:
            # update reference
            if ref_preview:
                y_ref_k = y_ref[k : k + int(T / Ts) + 1, :][
                    [0] + intermediate_nodes + [-1]
                ]
            else:
                y_ref_k = np.repeat(y_ref[k, :].reshape(1, -1), N + 1, axis=0)

            for j in range(N):
                acados_ocp_solver.set(j, "yref", y_ref_k[j])
            acados_ocp_solver.set(N, "yref", y_ref_k[N, 0:-nu])

            # update the control
            simU[k, :] = acados_ocp_solver.solve_for_x0(
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

            # update discrete-time iteration counter
            k += 1

        # simulate system
        simX[i + 1, :] = acados_integrator.simulate(simX[i, :], simU[k - 1, :])

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
        # plot state and control evolution
        plot_results(time, time_dt, simX, simU)
        # plot CPU time
        plot_cpt(time_dt, cpt, Ts)

        # visualize the adopted non-uniform grid
        plot_grid(shooting_nodes, title="Adopted non-uniform grid")

        # plot state and control evolution along with prediction in a given discrete time instant
        plot_pred_traj(
            time,
            time_dt,
            simX,
            simU,
            x_opt,
            u_opt,
            np.argwhere(np.round(time_dt, 3) == 5),
            shooting_nodes,
        )

        # make animation
        if save_video:
            inverted_pendulum_animation(simX[:, 0], simX[:, 1], ts_sim)

        plt.show()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    closed_loop_simulation()

    # to make simulation with non-uniform grid
    # closed_loop_simulation_nug()
