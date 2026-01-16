import matplotlib.pyplot as plt
import numpy as np

# from acados_template import latexify_plot
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from tqdm import tqdm


def plot_results(time, time_dt, state, control, ref, labels, ctrl_on=True):
    # latexify_plot()
    labels = labels + ["reference"] if ctrl_on else labels
    # plot state
    (fig, _) = plt.subplots(2, 2)

    # - plot cart position
    plt.subplot(2, 2, 1)
    plt.plot(time, state[:, 0])
    if ctrl_on:
        plt.plot(time_dt, ref[: len(time_dt), 0])
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$p$ [m]")
    plt.grid(True)

    # - plot pendulum angle
    plt.subplot(2, 2, 2)
    plt.plot(time, np.rad2deg(state[:, 1]))
    if ctrl_on:
        plt.plot(time_dt, np.rad2deg(ref[: len(time_dt), 1]))
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\theta$ [deg]")
    plt.gca().grid(True)

    # - plot cart velocity
    plt.subplot(2, 2, 3)
    plt.plot(time, state[:, 2])
    if ctrl_on:
        plt.plot(time_dt, ref[: len(time_dt), 2])
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$v$ [m/s]")
    plt.grid(True)

    # - plot pendulum angular velocity
    plt.subplot(2, 2, 4)
    plt.plot(time, np.rad2deg(state[:, 3]))
    if ctrl_on:
        plt.plot(time_dt, ref[: len(time_dt), 3])
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\omega$ [deg/s]")
    plt.grid(True)

    fig.legend(labels, loc="lower center", ncol=1 + len(labels), draggable=True)

    if ctrl_on:
        # plot control input
        plt.figure()
        plt.step(time_dt, np.vstack((control, control[-1])), where="post")
        plt.legend(labels, loc="upper center", ncol=1 + len(labels))
        plt.gca().set_xlabel("time [s]")
        plt.gca().set_ylabel("$F$ [N]")
        plt.grid(True)


def plot_pred_traj(time, time_dt, state, control, x_opt, u_opt, k, shooting_nodes=None):
    # latexify_plot()

    plt.subplots(2, 2)

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
    plt.plot(time, state[:, 0])
    plt.step(time_pred, x_opt[:, 0, k].reshape(-1, 1), where="post", color="red")
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$p$ [m]")
    plt.grid(True)

    plt.subplot(2, 2, 2)

    plt.plot(time, np.rad2deg(state[:, 1]))
    plt.step(
        time_pred, np.rad2deg(x_opt[:, 1, k].reshape(-1, 1)), where="post", color="red"
    )
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\theta$ [deg]")
    plt.gca().grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time, state[:, 2])
    plt.step(time_pred, x_opt[:, 2, k].reshape(-1, 1), where="post", color="red")
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$v$ [m/s]")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time, np.rad2deg(state[:, 3]))
    plt.step(
        time_pred, np.rad2deg(x_opt[:, 3, k].reshape(-1, 1)), where="post", color="red"
    )
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$\\omega$ [deg/s]")
    plt.grid(True)

    plt.tight_layout()

    plt.figure()
    plt.step(time_dt, np.append(control, control[-1]), where="post")
    plt.step(
        time_pred,
        np.append(u_opt[:, 0, k], u_opt[-1, 0, k]).reshape(-1, 1),
        where="post",
        color="red",
    )
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("$F$ [N]")
    plt.grid(True)


def plot_cpt(t, cpt, Ts=None, labels=[]):
    # latexify_plot()

    plt.figure()
    plt.step(t, np.vstack((cpt, cpt[-1])) * 1000, where="post")
    plt.legend(labels, loc="upper center", ncol=len(labels))
    plt.gca().set_xlabel("time [s]")
    plt.gca().set_ylabel("cpt [ms]")
    plt.grid(True)

    if Ts is not None:
        plt.hlines(Ts * 1000, t[0], t[-1], linestyles="dashed", alpha=0.7)


def inverted_pendulum_animation(p, theta, ts, filename=None):
    # latexify_plot()

    # define colors
    cart_color = [0.6549, 0.7804, 0.9059]
    pendulum_color = [0.2549, 0.4118, 0.8824]
    mass_color = [0.2510, 0.8784, 0.8157]

    # initialize plot
    fig, ax = plt.subplots()
    (plot,) = ax.plot([], [], color=pendulum_color, linewidth=2, zorder=3)

    # set plot style
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.set_xlim((-3, 3))
    ax.set_ylim(ymin=-1, ymax=1)
    ax.set_aspect("equal")
    # ax.set_title('\\bfseries Inverted Pendulum simulation')
    ax.set_title("Inverted Pendulum simulation", fontweight="bold")
    ax.grid(True)

    # set number of frames per second
    fps = 20
    # create FFMpegWriter object
    animation_writer = FFMpegWriter(fps)

    # compute decimation factor
    df = int(1 / (ts * fps))

    # define parameters
    cart_width = 1
    cart_height = 0.5
    l = 0.8

    # draw cart
    r = Rectangle(
        (p[0] - cart_width / 2, -cart_height / 2),
        cart_width,
        cart_height,
        color=cart_color,
        zorder=2,
    )
    ax.add_patch(r)

    # draw line for base
    ax.hlines(-cart_height / 2, -2.5, 2.5, colors="black")

    # compute x,y coordinates of the end of the pendulum
    x_pendulum = p - l * np.sin(theta)
    y_pendulum = l * np.cos(theta)

    # draw pendulum mass
    mass = Circle((y_pendulum[0], y_pendulum[0]), 0.1, color=mass_color, zorder=4)
    ax.add_patch(mass)

    with animation_writer.saving(
        fig, filename if filename is not None else "simulation.mp4", dpi=300
    ):
        for i in tqdm(
            range(0, len(p), df),
            desc="Generating Animation",
            ascii=False,
            ncols=75,
            colour="green",
        ):
            # update cart, pendulum and mass position
            r.set(xy=(p[i] - cart_width / 2, -cart_height / 2))
            plot.set_data([p[i], x_pendulum[i]], [0, y_pendulum[i]])
            mass.set(center=(x_pendulum[i], y_pendulum[i]))

            # update figure
            fig.canvas.draw()

            # save figure as video frame
            animation_writer.grab_frame()

    plt.close(fig)
