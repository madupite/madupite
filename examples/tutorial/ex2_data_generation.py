import os

import numpy as np
import scipy.sparse

import madupite as md

# --------------
# Data generation for Example 2
# --------------

# Using the same model as in Example 1, this script shows how to preprocess data into PETSc binary files.

# Do not use MPI for this.


def P_stochastic(state, action):
    if action == 0:  # stay
        return [0.1, 0.8, 0.1], [(state - 1) % 50, state, (state + 1) % 50]
    if action == 1:  # left
        return [0.1, 0.9], [state, (state - 1) % 50]
    if action == 2:  # right
        return [0.1, 0.9], [state, (state + 1) % 50]


def reward(state, action):
    return 1 if state == 42 else 0


def generate_P_matrix(num_states, num_actions):
    """
    Here we generate a sparse matrix using only the nonzero values and indices. This is recommended for large matrices to save space.
    """

    values, colidxs, rowidxs = [], [], []
    for state in range(num_states):
        for action in range(num_actions):
            vals, idxs = P_stochastic(state, action)
            for val, idx in zip(vals, idxs):
                values.append(val)
                colidxs.append(idx)
                rowidxs.append(state * num_actions + action)

    P = scipy.sparse.csr_matrix(
        (values, (rowidxs, colidxs)), shape=(num_states * num_actions, num_states)
    )

    return P


def generate_r_matrix(num_states, num_actions):
    """
    Here we show that also dense matrices can be generated.
    """

    r = np.empty((num_states, num_actions))
    for state in range(num_states):
        for action in range(num_actions):
            r[state, action] = reward(state, action)

    return r


def main():
    # Create output directory
    if not os.path.exists("data"):
        os.makedirs("data")

    # Model parameters
    num_states = 50
    num_actions = 3

    P = generate_P_matrix(num_states, num_actions)
    r = generate_r_matrix(num_states, num_actions)

    # Save matrices
    md.writePETScBinary(P, "data/P.bin")
    md.writePETScBinary(r, "data/r.bin")


if __name__ == "__main__":
    main()
