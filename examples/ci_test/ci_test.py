import madupite


def rewardfunc(s, a):
    return 1.0 if s == 41 else 0.0


def probfunc(s, a):
    if a == 0:  # stay
        values = [0.9, 0.05, 0.05]
        indices = [s, (s - 1 + 50) % 50, (s + 1) % 50]
    elif a == 1:  # left
        values = [0.1, 0.9]
        indices = [s, (s - 1 + 50) % 50]
    elif a == 2:  # right
        values = [0.1, 0.9]
        indices = [s, (s + 1) % 50]
    else:
        raise ValueError("invalid action index")
    return values, indices


def main():
    instance = madupite.initialize_madupite()

    mdp = madupite.MDP(instance)

    mdp.setOption("-mode", "MAXREWARD")
    mdp.setOption("-discount_factor", "0.9999")
    mdp.setOption("-max_iter_pi", "200")
    mdp.setOption("-max_iter_ksp", "1000")
    mdp.setOption("-alpha", "1e-4")
    mdp.setOption("-atol_pi", "1e-8")
    mdp.setOption("-file_stats", "ci_stats.json")
    mdp.setOption("-file_cost", "ci_reward.out")
    mdp.setOption("-file_policy", "ci_policy.out")
    mdp.setOption("-ksp_type", "gmres")

    num_states = 50
    num_actions = 3
    prealloc = madupite.MatrixPreallocation()
    prealloc.o_nz = 3
    prealloc.d_nz = 3
    g = madupite.createStageCostMatrix(
        name="g", numStates=num_states, numActions=num_actions, func=rewardfunc
    )
    P = madupite.createTransitionProbabilityTensor(
        name="P",
        numStates=num_states,
        numActions=num_actions,
        func=probfunc,
        preallocation=prealloc,
    )
    mdp.setStageCostMatrix(g)
    mdp.setTransitionProbabilityTensor(P)
    mdp.solve()

    g = madupite.Matrix.fromFile(
        comm=madupite.getCommWorld(),
        name="g",
        filename="100_50_0.1/g.bin",
        category=madupite.MatrixCategory.Cost,
        type=madupite.MatrixType.Dense,
    )
    P = madupite.Matrix.fromFile(
        comm=madupite.getCommWorld(),
        name="P",
        filename="100_50_0.1/P.bin",
        category=madupite.MatrixCategory.Dynamics,
        type=madupite.MatrixType.Sparse,
    )

    mdp.setStageCostMatrix(g)
    mdp.setTransitionProbabilityTensor(P)
    mdp.solve()


if __name__ == "__main__":
    main()
