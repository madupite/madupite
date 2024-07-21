import time

import numba
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
@numba.jit(nopython=True)
def x2s(x, xd):
    return int(x * NUM_XD + xd)


@numba.jit(nopython=True)
def s2x(s):
    return s // NUM_XD, s % NUM_XD


@numba.jit(nopython=True)
def interpolate(x, y, grid_x, grid_y):
    x_i = np.searchsorted(grid_x, x, side="right")
    y_i = np.searchsorted(grid_y, y, side="right")

    # Replace np.clip with a manual implementation
    if x_i < 1:
        x_i = 1
    elif x_i >= len(grid_x) - 1:
        x_i = len(grid_x) - 1

    if y_i < 1:
        y_i = 1
    elif y_i >= len(grid_y) - 1:
        y_i = len(grid_y) - 1

    xl_v, xr_v = grid_x[x_i - 1], grid_x[x_i]
    yl_v, yr_v = grid_y[y_i - 1], grid_y[y_i]
    wx1, wy1 = (x - xl_v) / (xr_v - xl_v), (y - yl_v) / (yr_v - yl_v)
    wx0, wy0 = 1 - wx1, 1 - wy1
    indices = [(x_i - 1, y_i - 1), (x_i, y_i - 1), (x_i - 1, y_i), (x_i, y_i)]
    weights = [wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1]
    return indices, weights


@numba.jit(nopython=True)
def step(s, a):
    x_t_i, xd_t_i = s2x(s)
    x_t_v = x_vals[x_t_i]
    xd_t_v = xd_vals[xd_t_i]
    a_t_v = a_vals[a]

    x_tpp_v = (x_t_v + xd_t_v * DT) % MAX_X
    xdd_t_v = -G / L * np.sin(x_t_v) + a_t_v / (M * L**2)
    xd_tpp_v = xd_t_v + xdd_t_v * DT

    # Replace np.clip with a manual implementation
    if xd_tpp_v < -MAX_XD:
        xd_tpp_v = -MAX_XD
    elif xd_tpp_v > MAX_XD:
        xd_tpp_v = MAX_XD

    return x_tpp_v, xd_tpp_v


@numba.jit(nopython=True)
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


def main():
    # Preprocessing
    global x_vals, xd_vals, a_vals
    x_vals = np.linspace(0, MAX_X, NUM_X)  # (!) starting from 0
    xd_vals = np.linspace(-MAX_XD, MAX_XD, NUM_XD)
    a_vals = np.linspace(-MAX_A, MAX_A, NUM_A)

    # Construct matrices
    p_start = time.time()
    P = construct_transition_probability_matrix()
    p_end = time.time()
    print(f"Time to construct P:    {p_end - p_start:.4f}s")
    g_start = time.time()
    g = construct_stage_cost_matrix()
    g_end = time.time()
    print(f"Time to construct g:    {g_end - g_start:.4f}s")

    # Save matrices for madupite
    p_start = time.time()
    write_sparse_matrix_as_petsc_binary(
        P, f"data/pend_P_{NUM_STATES}_{NUM_ACTIONS}.bin"
    )
    p_end = time.time()
    print(f"Time to save P:         {p_end - p_start:.4f}s")
    g_start = time.time()
    write_sparse_matrix_as_petsc_binary(
        g, f"data/pend_g_{NUM_STATES}_{NUM_ACTIONS}.bin"
    )
    g_end = time.time()
    print(f"Time to save g:         {g_end - g_start:.4f}s")
    print(
        f"Matrices saved to data/pend_P_{NUM_STATES}_{NUM_ACTIONS}.bin and data/pend_g_{NUM_STATES}_{NUM_ACTIONS}.bin"
    )


if __name__ == "__main__":
    main()
