import os

import madupite as md

# --------------
# Example 1
# --------------


def P_deterministic(state, action):
    if action == 0:  # stay
        return [1], [state]
    if action == 1:  # left
        return [1], [(state - 1) % 50]
    if action == 2:  # right
        return [1], [(state + 1) % 50]


def P_stochastic(state, action):
    if action == 0:  # stay
        return [0.1, 0.8, 0.1], [(state - 1) % 50, state, (state + 1) % 50]
    if action == 1:  # left
        return [0.1, 0.9], [state, (state - 1) % 50]
    if action == 2:  # right
        return [0.1, 0.9], [state, (state + 1) % 50]


def r(state, action):
    return 1 if state == 42 else 0


def main():
    # Initialize madupite
    instance = md.initialize_madupite()
    rank, size = md.mpi_rank_size()

    # Create output directory (only on one process)
    if rank == 0 and not os.path.exists("out"):
        os.makedirs("out")

    # Preallocation
    prealloc_deterministic = md.MatrixPreallocation()
    prealloc_deterministic.d_nz = 1
    prealloc_deterministic.o_nz = 1

    prealloc_stochastic = md.MatrixPreallocation()
    prealloc_stochastic.d_nz = 3
    prealloc_stochastic.o_nz = 3

    # Create the transition probability tensor and the stage cost matrix
    P_mat_deterministic = md.createTransitionProbabilityTensor(
        name="prob_ex1_deterministic",
        numStates=50,
        numActions=3,
        func=P_deterministic,
        preallocation=prealloc_deterministic,
    )

    P_mat_stochastic = md.createTransitionProbabilityTensor(
        name="prob_ex1_stochastic",
        numStates=50,
        numActions=3,
        func=P_stochastic,
        preallocation=prealloc_stochastic,
    )

    r_mat = md.createStageCostMatrix(
        name="reward_ex1", numStates=50, numActions=3, func=r
    )

    # Create MDP object and set options
    mdp = md.MDP(instance)
    mdp.setOption("-mode", "MAXREWARD")
    mdp.setOption("-discount_factor", "0.99")
    mdp.setOption("-file_stats", "out/ex1_stats.json")
    mdp.setOption("-file_policy", "out/ex1_policy_deterministic.json")
    mdp.setOption("-file_cost", "out/ex1_reward_deterministic.out")

    # Set transition probabilities and rewards
    mdp.setTransitionProbabilityTensor(P_mat_deterministic)
    mdp.setStageCostMatrix(r_mat)

    # Solve
    mdp.solve()

    # Second run using stochastic transition probabilities, reuse the same MDP object and options
    mdp.setTransitionProbabilityTensor(P_mat_stochastic)
    mdp.setOption("-file_policy", "out/ex1_policy_stochastic.json")
    mdp.setOption("-file_cost", "out/ex1_reward_stochastic.out")
    mdp.solve()


if __name__ == "__main__":
    main()
