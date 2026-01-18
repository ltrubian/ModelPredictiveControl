from acados_template import AcadosModel
import casadi as ca
from typing import Literal, get_args

_MODEL_TYPE = Literal["linear", "non-linear", "spring", "extended"]


def get_inverted_pendulum_model(
    M=1, m=0.1, l=0.8, g=9.81, k=10**4, type: _MODEL_TYPE = "non-linear"
) -> AcadosModel:
    # set model name
    model_name = str("inverted_pendulum_" + type).replace("-", "_")

    # the physical parameters of the system are passed as parameters
    # g - gravity acceleration [m/s**2]
    # M - cart mass [kg]
    # m - pendulum mass [kg]
    # l - pendulum length [m]
    # k - spring elastic constant [N/m]

    # setup states and controls symbolic variables
    p = ca.SX.sym("p", 1, 1)
    theta = ca.SX.sym("theta", 1, 1)
    v = ca.SX.sym("v", 1, 1)
    omega = ca.SX.sym("omega", 1, 1)

    x = ca.vertcat(p, theta, v, omega)
    u = ca.SX.sym("u", 1, 1)

    # setup symbolic variables for xdot (to be used with IRK integrator)
    p_dot = ca.SX.sym("p_dot", 1, 1)
    theta_dot = ca.SX.sym("theta_dot", 1, 1)
    v_dot = ca.SX.sym("v_dot", 1, 1)
    omega_dot = ca.SX.sym("omega_dot", 1, 1)

    xdot = ca.vertcat(p_dot, theta_dot, v_dot, omega_dot)

    # define dynamics
    match type:
        case "non-linear":
            a = (
                -m * l * ca.sin(theta) * omega**2
                + m * g * ca.cos(theta) * ca.sin(theta)
                + u
            )
            b = (
                -m * l * ca.cos(theta) * ca.sin(theta) * omega**2
                + u * ca.cos(theta)
                + (M + m) * g * ca.sin(theta)
            )
            c = M + m - m * (ca.cos(theta)) ** 2

            # explicit ODE right hand side (to be used with ERK integrator)
            f_expl = ca.vertcat(v, omega, a / c, b / (l * c))
            # implicit dynamics (to be used with IRK integrator)
            f_impl = xdot - f_expl

        case "linear":
            a = (theta * m * g + u) / M
            b = ((theta * g) * (M + m) + u) / (M * l)
            # explicit ODE right hand side (to be used with ERK integrator)
            f_expl = ca.vertcat(v, omega, a, b)

            # implicit dynamics (to be used with IRK integrator)
            f_impl = xdot - f_expl

        case "extended":
            # Add the new state for the control input u and update x, xdot
            F = ca.SX.sym("F", 1, 1)
            F_dot = ca.SX.sym("F_dot", 1, 1)
            x = ca.vertcat(p, theta, v, omega, F)
            xdot = ca.vertcat(p_dot, theta_dot, v_dot, omega_dot, F_dot)

            a = (
                -m * l * ca.sin(theta) * omega**2
                + m * g * ca.cos(theta) * ca.sin(theta)
                + F
            )
            b = (
                -m * l * ca.cos(theta) * ca.sin(theta) * omega**2
                + F * ca.cos(theta)
                + (M + m) * g * ca.sin(theta)
            )
            c = M + m - m * (ca.cos(theta)) ** 2

            # explicit ODE right hand side (to be used with ERK integrator)
            f_expl = ca.vertcat(v, omega, a / c, b / (l * c), u)
            # implicit dynamics (to be used with IRK integrator)
            f_impl = xdot - f_expl

        case "spring":
            a = (
                -m * l * ca.sin(theta) * omega**2
                + m * g * ca.cos(theta) * ca.sin(theta)
                + u
                - k * p
            )
            b = (
                -m * l * ca.cos(theta) * ca.sin(theta) * omega**2
                + (u - k * p) * ca.cos(theta)
                + (M + m) * g * ca.sin(theta)
            )
            c = M + m - m * (ca.cos(theta)) ** 2

            # explicit ODE right hand side (to be used with ERK integrator)
            f_expl = ca.vertcat(v, omega, a / c, b / (l * c))
            # implicit dynamics (to be used with IRK integrator)
            f_impl = xdot - f_expl
        case _:
            raise ValueError(
                f"""Invalid model type {type}, must be one of
                    {get_args(_MODEL_TYPE)}"""
            )

    # create acados model and fill in all the required fields
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model
