import os

import numpy as np
from numba import njit

import madupite as md

# Constants and model parameters
NUM_X = 201  # Number of discretized positions (angles)
NUM_XD = 201  # Number of discretized angular velocities
NUM_A = 9  # Number of discretized actions/torque

MAX_X = 2 * np.pi  # Maximum position (angle)
MAX_XD = 10.0  # Maximum angular velocity
MAX_A = 3.0  # Maximum torque
R = 1  # Penalty for control effort
Q = 2  # Penalty for state deviation
DT = 0.01  # Time step
G = 9.81  # Gravitational acceleration
L = 1.0  # Pendulum length
M = 1.0  # Pendulum mass
NUM_STATES = NUM_X * NUM_XD  # Total number of states
NUM_ACTIONS = NUM_A  # Total number of actions


# Helper functions that map a 2D state onto a 1D index
@njit
def x2s(x, xd):
    return int(x * NUM_XD + xd)


@njit
def s2x(s):
    return s // NUM_XD, s % NUM_XD


@njit
def numba_clip(value, min_value, max_value):
    """Clip value within min_value and max_value."""
    return min(max(value, min_value), max_value)


@njit
def interpolate(x, y, grid_x, grid_y):
    """Interpolate between grid points."""
    x_i = np.searchsorted(grid_x, x, side="right")
    y_i = np.searchsorted(grid_y, y, side="right")
    x_i, y_i = numba_clip(x_i, 1, len(grid_x) - 1), numba_clip(y_i, 1, len(grid_y) - 1)
    xl_v, xr_v = grid_x[x_i - 1], grid_x[x_i]
    yl_v, yr_v = grid_y[y_i - 1], grid_y[y_i]
    wx1, wy1 = (x - xl_v) / (xr_v - xl_v), (y - yl_v) / (yr_v - yl_v)
    wx0, wy0 = 1 - wx1, 1 - wy1
    indices = [(x_i - 1, y_i - 1), (x_i, y_i - 1), (x_i - 1, y_i), (x_i, y_i)]
    weights = [wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1]
    return indices, weights


@njit
def step(s, a):
    """Calculate the next state for a given state-action pair."""
    x_t_i, xd_t_i = s2x(s)
    x_t_v = x_vals[x_t_i]
    xd_t_v = xd_vals[xd_t_i]
    a_t_v = a_vals[a]

    # Compute next state based on dynamics
    x_tpp_v = (x_t_v + xd_t_v * DT) % MAX_X  # Periodic boundary condition
    xdd_t_v = -G / L * np.sin(x_t_v) + a_t_v / (M * L**2)
    xd_tpp_v = numba_clip(xd_t_v + xdd_t_v * DT, -MAX_XD, MAX_XD)

    return x_tpp_v, xd_tpp_v


def stage_cost(s, a):
    """Compute stage cost for a given state-action pair."""
    x_i, xd_i = s2x(s)
    return Q * ((x_vals[x_i] - np.pi) ** 2 + xd_vals[xd_i] ** 2) + R * a_vals[a] ** 2


def transition_probability(s, a):
    """Compute transition probabilities for a given state-action pair."""
    x_tpp_v, xd_tpp_v = step(s, a)
    indices, weights = interpolate(x_tpp_v, xd_tpp_v, x_vals, xd_vals)
    s_indices = [x2s(x_i, xd_i) for x_i, xd_i in indices]
    return weights, s_indices


def main():
    # Initialize grid values
    global x_vals, xd_vals, a_vals
    x_vals = np.linspace(0, MAX_X, NUM_X)  # Starting from 0
    xd_vals = np.linspace(-MAX_XD, MAX_XD, NUM_XD)
    a_vals = np.linspace(-MAX_A, MAX_A, NUM_A)

    # Initialize Madupite instance
    madupite_instance = md.initialize_madupite()

    # Create MDP instance
    mdp = md.MDP(madupite_instance)

    # Set MDP options
    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-max_iter_pi", "100")
    mdp.setOption("-max_iter_ksp", "20000")
    mdp.setOption("-alpha", "1e-5")
    mdp.setOption("-atol_pi", "1e-7")
    mdp.setOption("-ksp_type", "tfqmr")
    mdp.setOption("-discount_factor", "0.99")
    mdp.setOption("-file_stats", "data/pend_stats.json")
    mdp.setOption("-file_policy", "data/pend_policy.out")
    mdp.setOption("-file_cost", "data/pend_cost.out")
    mdp.setOption("-export_optimal_transition_probabilities", "data/pend_Ppi.out")
    mdp.setOption("-export_optimal_stage_costs", "data/pend_gpi.out")

    # Create the stage cost matrix using the stage cost function
    filename_P = f"data/pend_P_{NUM_STATES}_{NUM_ACTIONS}.bin"
    filename_g = f"data/pend_g_{NUM_STATES}_{NUM_ACTIONS}.bin"
    if os.path.exists(filename_P):
        P_mat = md.Matrix.fromFile(
            comm=md.getCommWorld(),
            name="P",
            filename=filename_P,
            category=md.MatrixCategory.Dynamics,
            type=md.MatrixType.Sparse,
        )
    else:
        P_mat = md.createTransitionProbabilityTensor(
            name="P",
            numStates=NUM_STATES,
            numActions=NUM_ACTIONS,
            func=transition_probability,
        )
        P_mat.writeToFile(filename_P, md.MatrixType.Sparse, True)

    if os.path.exists(filename_g):
        g_mat = md.Matrix.fromFile(
            comm=md.getCommWorld(),
            name="g",
            filename=filename_g,
            category=md.MatrixCategory.Cost,
            type=md.MatrixType.Dense,
        )
    else:
        g_mat = md.createStageCostMatrix(
            name="g", numStates=NUM_STATES, numActions=NUM_ACTIONS, func=stage_cost
        )
        g_mat.writeToFile(filename_g, md.MatrixType.Dense, True)

    # Set matrices in the MDP
    mdp.setStageCostMatrix(g_mat)
    mdp.setTransitionProbabilityTensor(P_mat)

    if md.mpi_rank_size()[0] == 0:
        print("Solving...", flush=True)

    # Solve the MDP
    mdp.solve()

    del mdp


if __name__ == "__main__":
    main()
