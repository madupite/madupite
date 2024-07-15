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


def interpolate(x, y, grid_x, grid_y):
    x_i = np.searchsorted(grid_x, x, side="right")
    y_i = np.searchsorted(grid_y, y, side="right")
    x_i, y_i = np.clip(x_i, 1, len(grid_x) - 1), np.clip(y_i, 1, len(grid_y) - 1)
    xl_v, xr_v = grid_x[x_i - 1], grid_x[x_i]
    yl_v, yr_v = grid_y[y_i - 1], grid_y[y_i]
    wx1, wy1 = (x - xl_v) / (xr_v - xl_v), (y - yl_v) / (yr_v - yl_v)
    wx0, wy0 = 1 - wx1, 1 - wy1
    indices = [(x_i - 1, y_i - 1), (x_i, y_i - 1), (x_i - 1, y_i), (x_i, y_i)]
    weights = [wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1]
    return indices, weights


def step(s, a):
    """
    Calculate the next state for a given state-action pair.

    Parameters:
    s: int, current state index(!) (value in [0, NUM_STATES))
    a: int, action index (value in [0, NUM_ACTIONS))

    Returns:
    x_next, xd_next: float(!), coordinate values of the next state in [-MAX_X, MAX_X] and [-MAX_XD, MAX_XD]
    """
    x_t_i, xd_t_i = s2x(s)
    x_t_v = x_vals[x_t_i]
    xd_t_v = xd_vals[xd_t_i]
    a_t_v = a_vals[a]

    x_tpp_v = (x_t_v + xd_t_v * DT) % MAX_X  # periodc boundary condition!
    xdd_t_v = -G / L * np.sin(x_t_v) + a_t_v / (M * L**2)
    xd_tpp_v = np.clip(xd_t_v + xdd_t_v * DT, -MAX_XD, MAX_XD)

    return x_tpp_v, xd_tpp_v


def stage_cost(x_i, xd_i, a):
    return Q * ((x_vals[x_i] - np.pi) ** 2 + xd_vals[xd_i] ** 2) + R * a_vals[a] ** 2


def construct_transition_probability_matrix():
    data, row_idx, col_idx = [], [], []
    for s_t_i in range(NUM_STATES):
        for a_t_i in range(NUM_ACTIONS):
            x_tpp_v, xd_tpp_v = step(s_t_i, a_t_i)
            indices, weights = interpolate(x_tpp_v, xd_tpp_v, x_vals, xd_vals)
            for (x_tpp_i, xd_tpp_i), w in zip(indices, weights):
                s_tpp_i = x2s(x_tpp_i, xd_tpp_i)
                data.append(w)
                row_idx.append(s_t_i * NUM_ACTIONS + a_t_i)
                col_idx.append(s_tpp_i)
    return csr_matrix(
        (data, (row_idx, col_idx)), shape=(NUM_STATES * NUM_ACTIONS, NUM_STATES)
    )


def construct_stage_cost_matrix():
    g_dense = np.empty((NUM_STATES, NUM_ACTIONS))
    for s in range(NUM_STATES):
        x_t_i, xd_t_i = s2x(s)
        for a in range(NUM_ACTIONS):
            g_dense[s, a] = stage_cost(x_t_i, xd_t_i, a)
    return csr_matrix(g_dense)


def write_sparse_matrix_as_petsc_binary(sparse_matrix, filename):
    sparse_matrix.sort_indices()
    with open(filename, "wb") as f:
        f.write(b"\x00\x12\x7b\x50")
        f.write(np.array(sparse_matrix.shape, dtype=">i").tobytes())
        f.write(np.array(sparse_matrix.nnz, dtype=">i").tobytes())
        f.write(np.array(np.diff(sparse_matrix.indptr), dtype=">i").tobytes())
        f.write(np.array(sparse_matrix.indices, dtype=">i").tobytes())
        f.write(np.array(sparse_matrix.data, dtype=">d").tobytes())


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

    # Construct matrices
    # P = construct_transition_probability_matrix()
    # g = construct_stage_cost_matrix()

    # Save matrices for madupite
    # write_sparse_matrix_as_petsc_binary(
    #     P, f"data/pend_P_{NUM_STATES}_{NUM_ACTIONS}.bin"
    # )
    # write_sparse_matrix_as_petsc_binary(
    #     g, f"data/pend_g_{NUM_STATES}_{NUM_ACTIONS}.bin"
    # )
    # print(
    #     f"Matrices saved to data/pend_P_{NUM_STATES}_{NUM_ACTIONS}.bin and data/pend_g_{NUM_STATES}_{NUM_ACTIONS}.bin"
    # )

    # Call madupite solver
    flags = []
    # flags = [
    #     "-file_stats",
    #     "di_stats.json",
    #     "-file_cost",
    #     "di_cost.out",
    #     "-file_policy",
    #     "di_policy.out",
    #     "-num_states",
    #     str(NUM_STATES),
    #     "-num_actions",
    #     str(NUM_ACTIONS),
    #     "-discount_factor",
    #     "0.999",
    # ]
    build_dir = "./"
    try:
        result = subprocess.run(
            ["mpirun", "-n", "4", "./bin/pendulum", *flags],
            cwd=build_dir,
            capture_output=True,
            check=True,
            text=True,
        )
        print("Madupite Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.returncode, e.output, e.stderr)

    # Postprocessing and visualization
    costs = reshape_data(np.loadtxt("pend_cost.out"), (NUM_X, NUM_XD))
    policy = np.loadtxt("pend_policy.out", dtype=int)
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
