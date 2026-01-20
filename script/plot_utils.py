import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    time,
    time_dt,
    state,
    control,
    ref,
    labels,
    ctrl_on=True,
    xlimits=None,
    save=False,
    folder="./images/",
):
    # latexify_plot()
    labels: list[str] = [*labels, "reference"] if ctrl_on else labels
    # plot state
    (fig, _) = plt.subplots(2, 2)

    # - plot cart position
    plt.subplot(2, 2, 1)
    plt.plot(time, state[:, 0])
    if ctrl_on:
        plt.plot(time_dt, ref[: len(time_dt), 0])
    # plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$p$ [m]")
    plt.grid(True)
    if xlimits is not None:
        plt.xlim(xlimits)

    # - plot pendulum angle
    plt.subplot(2, 2, 2)
    plt.plot(time, np.rad2deg(state[:, 1]))
    if ctrl_on:
        plt.plot(time_dt, np.rad2deg(ref[: len(time_dt), 1]))
    # plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\theta$ [deg]")
    plt.gca().grid(True)
    if xlimits is not None:
        plt.xlim(xlimits)

    # - plot cart velocity
    plt.subplot(2, 2, 3)
    plt.plot(time, state[:, 2])
    if ctrl_on:
        plt.plot(time_dt, ref[: len(time_dt), 2])
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$v$ [m/s]")
    plt.grid(True)
    if xlimits is not None:
        plt.xlim(xlimits)

    # - plot pendulum angular velocity
    plt.subplot(2, 2, 4)
    plt.plot(time, np.rad2deg(state[:, 3]))
    if ctrl_on:
        plt.plot(time_dt, ref[: len(time_dt), 3])
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\omega$ [deg/s]")
    plt.grid(True)
    if xlimits is not None:
        plt.xlim(xlimits)

    fig.legend(labels, loc="upper center", ncol=1 + len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        fig.savefig(folder + "states.png", transparent=True, format="png")

    if ctrl_on:
        # plot control input
        plt.figure()
        plt.step(time_dt, np.vstack((control, control[-1])), where="post")
        plt.legend(labels, loc="upper center", ncol=1 + len(labels))
        plt.gca().set_xlabel("time [s]")
        plt.gca().set_ylabel("$F$ [N]")
        plt.grid(True)
        if xlimits is not None:
            plt.xlim(xlimits)
        if save:
            plt.savefig(folder + "input.png", transparent=True, format="png")


def plot_cpt(
    t, cpt, Ts=None, labels=[], xlimits=None, save=False, folder="./images/", prefix=""
):
    # latexify_plot()

    plt.figure()
    plt.step(t, np.vstack((cpt, cpt[-1])) * 1000, where="post")
    plt.legend(labels, loc="upper center", ncol=len(labels))
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("cpt [ms]")
    plt.grid(True)
    if xlimits is not None:
        plt.xlim(xlimits)
    if Ts is not None:
        plt.hlines(Ts * 1000, t[0], t[-1], linestyles="dashed", alpha=0.7)
    if save:
        plt.savefig(folder + prefix + "cputime.png", transparent=True, format="png")


def plot_pred_traj(
    time,
    time_dt,
    state,
    control,
    x_opt,
    u_opt,
    k,
    labels,
    shooting_nodes=None,
    save=False,
    folder="./images/",
    prefix="",
    xlimits=None,
):
    labels: list[str] = ["state", *labels]  # latexify_plot()

    (fig, _) = plt.subplots(2, 2)

    # get number of shooting time intervals
    N = x_opt.shape[0]

    # construct the time vector for prediction
    if shooting_nodes is None:
        # if the shooting nodes are not provided, assume uniform grid
        Ts = np.diff(time_dt)[-1]
        time_pred = time_dt[k].item() + np.arange(0, N) * Ts
    else:
        time_pred = time_dt[k].item() + shooting_nodes

    plt.subplot(2, 2, 1)
    plt.plot(time, state[:, 0], color="green")
    plt.step(time_pred, x_opt[:, 0], where="post")
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$p$ [m]")
    if xlimits is not None:
        plt.xlim(xlimits)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time, np.rad2deg(state[:, 1]), color="green")
    plt.step(time_pred, np.rad2deg(x_opt[:, 1]), where="post")
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\theta$ [deg]")
    if xlimits is not None:
        plt.xlim(xlimits)
    plt.gca().grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time, state[:, 2], color="green")
    plt.step(time_pred, x_opt[:, 2], where="post")
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$v$ [m/s]")
    if xlimits is not None:
        plt.xlim(xlimits)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time, np.rad2deg(state[:, 3]), color="green")
    plt.step(time_pred, np.rad2deg(x_opt[:, 3]), where="post")
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\omega$ [deg/s]")
    if xlimits is not None:
        plt.xlim(xlimits)
    plt.grid(True)

    fig.legend(labels, loc="upper center", ncol=1 + len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save:
        fig.savefig(folder + "state_pred.png", transparent=True, format="png")


#    plt.figure()
#    plt.step(time_dt, np.append(control, control[-1]), where="post")
#    plt.step(
#        time_pred,
#        np.append(u_opt[:, 0, k], u_opt[-1, 0, k]).reshape(-1, 1),
#        where="post",
#        color="red",
#    )
#    plt.gca().set_xlabel("time [s]")
#    plt.gca().set_ylabel("$F$ [N]")
#    plt.grid(True)
