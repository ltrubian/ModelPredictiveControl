from typing import Literal, Optional, get_args

import numpy as np
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosSim,
)
from numpy.typing import NDArray
from scipy.linalg import block_diag

_INTEGRATOR_TYPE = Literal["ERK", "IRK"]
_CONSTRAINTS = Literal["input-U", "state-F"]


def create_ocp_solver_description(
    model: AcadosModel,
    N: int,
    T: float,
    x0: NDArray[np.float64],
    Q: NDArray[np.float64] = np.diag([10, 10, 0.1, 0.1]),
    R: float = 0.01,
    shooting_nodes: Optional[NDArray[np.float64]] = None,
    constraints: _CONSTRAINTS = "input-U",
    integrator_type: _INTEGRATOR_TYPE = "ERK",
) -> AcadosOcp:
    """create optimal control problem description

    Args:
        model: predictive model of the controller
        N: number of intervals in which the prediction horizon is divided
        T: prediction horizon length [s]
        x0: initial condition
        Q: state cost weight matrix
        R: input cost weight
        shooting_nodes:
        constraints: (input-U/state-F) constraints on extended state or input
        integrator_type: numerical integrator type (Explicit/Implicit RK)

    Returns:
        AcadosOcp (acados optimal control problem)

    """

    assert integrator_type in get_args(_INTEGRATOR_TYPE), (
        f"{integrator_type=} not in {get_args(_INTEGRATOR_TYPE)}"
    )
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

    match constraints:
        case "input-U":
            # bounds on control input
            ocp.constraints.lbu = np.array([-20])
            ocp.constraints.ubu = np.array([20])
            ocp.constraints.idxbu = np.array([0])

        case "state-F":
            # bounds on position (0 component of state vector)
            ocp.constraints.idxbx = np.array([4])
            ocp.constraints.lbx = np.array([-20])
            ocp.constraints.ubx = np.array([20])

            # bounds on terminal state x_N
            ocp.constraints.idxbx_e = np.array([4])
            ocp.constraints.lbx_e = np.array([-20])
            ocp.constraints.ubx_e = np.array([20])
        case _:
            raise ValueError(f"{constraints=} is not in {get_args(_CONSTRAINTS)}")

    # initialize constraint on initial condition
    ocp.constraints.x0 = x0

    # set solver options
    ocp.solver_options.qp_solver = (
        "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES, PARTIAL_CONDENSING_HPIPM
    )
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = integrator_type  # ERK, IRK
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


def create_sim_solver_description(
    model: AcadosModel,
    Ts: float,
    name: Optional[str] = None,
    integrator_type: _INTEGRATOR_TYPE = "ERK",
) -> AcadosSim:
    """create simulation description object

    Args:
        model: model to simulate
        T: prediction horizon length [s]
        integrator_type: numerical integrator type (Explicit/Implicit Runge-Kutta)

    Returns:
        AcadosSim (acados simulation object)

    """

    assert integrator_type in get_args(_INTEGRATOR_TYPE), (
        f"{integrator_type=} not in {get_args(_INTEGRATOR_TYPE)}"
    )
    if name is None:
        name: str = "sim_" + model.name
    model.name: str = name
    # create sim object
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = Ts
    sim.solver_options.integrator_type = integrator_type

    return sim
