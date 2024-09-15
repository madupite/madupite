import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import madupite as md
import time

NUM_X = 41
NUM_XD = 41
NUM_A = 11
MAX_X = 3.0
MAX_XD = 4.0
MAX_A = 1.0
DT = 0.1

# derived constants - do not modify
NUM_STATES = NUM_X * NUM_XD
NUM_ACTIONS = NUM_A

x_vals = np.linspace(-MAX_X, MAX_X, NUM_X)
xd_vals = np.linspace(-MAX_XD, MAX_XD, NUM_XD)
a_vals = np.linspace(-MAX_A, MAX_A, NUM_A)

# Some helper functions that we need later. We use subscript `i` to denote indices (int) and `v` to denote values (float) as well as `t` ($t$) and `tpp` ($t+1$) for better readability.

def x2s(x, xd):
    """Convert [x, xd] to 1-dim state (row-major)."""
    return int(x * NUM_XD + xd)

def s2x(s):
    """Convert 1-dim state to [x, xd]."""
    return s // NUM_XD, s % NUM_XD

def interpolate(x, y, grid_x, grid_y):
    """
    Bilinear interpolation of a 2D grid at point (x, y).
    Returns the indices of the 4 grid points surrounding (x, y) and their weights.
    """
    x_i = np.searchsorted(grid_x, x, side='right')
    y_i = np.searchsorted(grid_y, y, side='right')

    x_i = np.clip(x_i, 1, len(grid_x) - 1)
    y_i = np.clip(y_i, 1, len(grid_y) - 1)

    xl_v, xr_v = grid_x[x_i-1], grid_x[x_i]
    yl_v, yr_v = grid_y[y_i-1], grid_y[y_i]

    wx1 = (x - xl_v) / (xr_v - xl_v)
    wx0 = 1 - wx1
    wy1 = (y - yl_v) / (yr_v - yl_v)
    wy0 = 1 - wy1

    indices = [(x_i-1, y_i-1), (x_i, y_i-1), (x_i-1, y_i), (x_i, y_i)]
    weights = [wx0*wy0, wx1*wy0, wx0*wy1, wx1*wy1]

    return indices, weights

# Next, we define the model dynamics, i.e. calculate the next state given the current state and action.

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

    x_tpp_v = np.clip(x_t_v + xd_t_v * DT, -MAX_X, MAX_X)
    xd_tpp_v = np.clip(xd_t_v + a_t_v * DT, -MAX_XD, MAX_XD)

    return x_tpp_v, xd_tpp_v   

# Now we can construct the transition probability function. The function returns the list of possible next states and the corresponding probabilities for a given current state and action.

def transition_probabilities(s_t_i, a_t_i):
    x_tpp_v, xd_tpp_v = step(s_t_i, a_t_i)
    indices, weights = interpolate(x_tpp_v, xd_tpp_v, x_vals, xd_vals)

    return weights, [x2s(x, xd) for x, xd in indices]

# Next, we define the stage cost function. In this case, the goal is to get the object to stop at $x = 0$, i.e. $x = 0$ and $\dot{x} = 0$. We can define the stage cost as the squared distance to the goal, which makes the function independent of the action.
# The stage cost function is given by:

def stage_cost(s_t_i, a_t_i):
    x_i, xd_i = s2x(s_t_i)
    return x_vals[x_i]**2 + xd_vals[xd_i]**2

# Functions for plotting
def reshape_data(data, shape):
    return np.rot90(data.reshape(shape), k=1)

def create_tick_labels(values, num_ticks):
    indices = np.linspace(0, len(values) - 1, num_ticks, dtype=int)
    return indices, [f"{val:.2f}" for val in values[indices]]

def plot_heatmap(ax, data, title, x_val, xd_val):
    im = ax.imshow(data, cmap='jet', interpolation='nearest')
    ax.set_title(title)
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_xlabel(r"$x$")
    
    yticks, yticklabels = create_tick_labels(xd_val[::-1], 5)
    xticks, xticklabels = create_tick_labels(x_val, 5)
    
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(xticklabels)
    
    return im

def plot():
    costs = reshape_data(np.loadtxt("di_cost.out"), (NUM_X, NUM_XD))
    policy = np.loadtxt("di_policy.out", dtype=int)
    user_input = reshape_data(a_vals[policy], (NUM_X, NUM_XD))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    im1 = plot_heatmap(ax1, costs, "Costs", x_vals, xd_vals)
    im2 = plot_heatmap(ax2, user_input, "Policy", x_vals, xd_vals)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    plt.savefig("di_plot.png")


def main():
    # We first create the transition probability tensor whose elements are given by the transition probability function above. In order to this efficiently, we first preallocate the tensor: Because of the bilinear interpolation in the 2D space, we have at most 4 entries per row (state-action pair). Hence, we preallocate 4 elements in the diagonal and 4 elements in the off-diagonal. We then fill the tensor with the transition probabilities.

    prealloc = md.MatrixPreallocation()
    prealloc.d_nz = 4
    prealloc.o_nz = 4

    P = md.createTransitionProbabilityTensor(
        numStates=NUM_STATES,
        numActions=NUM_ACTIONS,
        func=transition_probabilities,
        preallocation=prealloc
    )

    # We repeat the same for the stage costs. Since this matrix is stored as a dense matrix, no preallocation is necessary.

    g = md.createStageCostMatrix(
        numStates=NUM_STATES,
        numActions=NUM_ACTIONS,
        func=stage_cost
    )

    # Now we can create the MDP object, configure the optimization parameters and finally solve the MDP.

    mdp = md.MDP()
    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-discount_factor", "0.99")
    mdp.setOption("-file_cost", "di_cost.out")
    mdp.setOption("-file_policy", "di_policy.out")
    mdp.setOption("-file_stats", "di_stats.json")
    mdp.setOption("-verbose", "true") # enable console output
    mdp.setOption("-overwrite", "true") # overwrite existing files (so we don't need to adapt the filenames in the plotting script below)

    mdp.setTransitionProbabilityTensor(P)
    mdp.setStageCostMatrix(g)

    mdp.solve()



if __name__ == "__main__":
    main()

    # plot only on one process
    rank, _ = md.mpi_rank_size()
    if rank == 0:
        plot()