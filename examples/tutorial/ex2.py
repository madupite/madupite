import os

import madupite as md

# --------------
# Example 2
# --------------


def main():
    # Create output directory
    if not os.path.exists("out"):
        os.makedirs("out")

    # Initialize madupite
    instance = md.initialize_madupite()

    # Create the transition probability tensor and the stage cost matrix
    P = md.Matrix.fromFile(
        comm=md.getCommWorld(),
        name="prob_ex2",
        filename="data/P.bin",
        category=md.MatrixCategory.Dynamics,
        type=md.MatrixType.Sparse,
    )

    r = md.Matrix.fromFile(
        comm=md.getCommWorld(),
        name="reward_ex2",
        filename="data/r.bin",
        category=md.MatrixCategory.Cost,
        type=md.MatrixType.Dense,
    )

    # Create MDP object and set options
    mdp = md.MDP(instance)
    mdp.setOption("-mode", "MAXREWARD")
    mdp.setOption("-discount_factor", "0.99")
    mdp.setOption("-file_stats", "out/ex2_stats.json")
    mdp.setOption("-file_policy", "out/ex2_policy.json")
    mdp.setOption("-file_cost", "out/ex2_reward.out")

    # Set transition probabilities and rewards
    mdp.setTransitionProbabilityTensor(P)
    mdp.setStageCostMatrix(r)

    # Solve
    mdp.solve()


if __name__ == "__main__":
    main()
