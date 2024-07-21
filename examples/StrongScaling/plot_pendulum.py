import subprocess

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

# from IPython.display import HTML # for animation

# Constants and model parameters
NUM_X = 401
NUM_XD = 401
NUM_A = 201
MAX_X = 2 * np.pi
MAX_XD = 10.0
MAX_A = 3.0
R = 1  # penalty for control effort
Q = 2  # penalty for state deviation
DT = 0.01
G = 9.81  # gravitational acceleration
L = 1.0  # pendulum length
M = 1.0  # pendulum mass
NUM_STATES = NUM_X * NUM_XD
NUM_ACTIONS = NUM_A


# Helper functions
def x2s(x, xd):
    return int(x * NUM_XD + xd)


def s2x(s):
    return s // NUM_XD, s % NUM_XD

def reshape_data(data, shape):
    return np.rot90(data.reshape(shape), k=1)


def create_tick_labels(values, num_ticks):
    indices = np.linspace(0, len(values) - 1, num_ticks, dtype=int)
    return indices, [f"{val:.2f}" for val in values[indices]]


def plot_heatmap(ax, data, title, x_val, xd_val):
    im = ax.imshow(data, cmap="jet", interpolation="nearest")
    ax.set_title(title)
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_xlabel("$x$")
    yticks, yticklabels = create_tick_labels(xd_val[::-1], 5)
    xticks, xticklabels = create_tick_labels(x_val, 5)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(xticklabels)
    return im


def create_initial_state(x_0_v, xd_0_v):
    indices, weights = interpolate(x_0_v, xd_0_v, x_vals, xd_vals)
    s0 = np.zeros(NUM_STATES)
    for (x_i, xd_i), w in zip(indices, weights):
        s0[x2s(x_i, xd_i)] = w
    return s0


def simulate(s0, P_pi, T):
    trajectory = []
    s = s0
    trajectory.append(s2x(np.argmax(s)))
    for _ in range(T):
        s = s.T @ P_pi
        trajectory.append(s2x(np.argmax(s)))
    return trajectory


def create_animation(traj, ax1, ax2):
    (line1,) = ax1.plot([], [], "r-", lw=2)
    (line2,) = ax2.plot([], [], "r-", lw=2)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        x, y = zip(*traj[: i + 1])
        line1.set_data(x, y)
        line2.set_data(x, y)
        return line1, line2

    return animation.FuncAnimation(
        ax1.figure, animate, init_func=init, frames=len(traj), interval=500, blit=True
    )


def main():
    # Preprocessing
    global x_vals, xd_vals, a_vals
    x_vals = np.linspace(0, MAX_X, NUM_X)  # (!) starting from 0
    xd_vals = np.linspace(-MAX_XD, MAX_XD, NUM_XD)
    a_vals = np.linspace(-MAX_A, MAX_A, NUM_A)

    # Postprocessing and visualization
    costs = reshape_data(np.loadtxt("bin/costs.out"), (NUM_X, NUM_XD))
    policy = np.loadtxt("bin/policy.out", dtype=int)
    user_input = reshape_data(a_vals[policy], (NUM_X, NUM_XD))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = plot_heatmap(ax1, costs, "Costs", x_vals, xd_vals)
    im2 = plot_heatmap(ax2, user_input, "Policy", x_vals, xd_vals)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    # plt.show()
    plt.savefig("pendulum.png", dpi=300)

    # Simulate state trajectory
    # row_indices = np.arange(NUM_STATES) * NUM_ACTIONS + policy
    # P_pi = P[row_indices, :]
    # x_0_v, xd_0_v = 1.5, -2.75
    # T = 100
    # s0 = create_initial_state(x_0_v, xd_0_v)
    # trajectory = simulate(s0, P_pi, T)

    # # Create and display animation
    # ani = create_animation(trajectory, ax1, ax2)
    # HTML(ani.to_jshtml())


if __name__ == "__main__":
    main()
