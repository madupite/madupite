import sys

import madupite


def main():
    # Initialize MPI, PETSc and Madupite, passing command line arguments.
    madupite_instance = madupite.Madupite.initialize(sys.argv)

    # Create MDP instance
    mdp = madupite.MDP(madupite_instance)

    # Set options
    mdp.setOption("-mode", "MINCOST")
    mdp.setOption("-max_iter_pi", "100")
    mdp.setOption("-max_iter_ksp", "2000")
    mdp.setOption("-alpha", "1e-5")
    mdp.setOption("-atol_pi", "1e-7")
    mdp.setOption("-ksp_type", "tfqmr")
    mdp.setOption("-discount_factor", "0.999")
    mdp.setOption("-file_stats", "pend_stats.json")
    mdp.setOption("-file_policy", "pend_policy.out")
    mdp.setOption("-file_cost", "pend_cost.out")
    mdp.setOption("-export_optimal_transition_probabilities", "pend_Ppi.out")
    mdp.setOption("-export_optimal_stage_costs", "pend_gpi.out")

    # Get PETSc communicator
    comm = madupite.PETSC_COMM_WORLD

    # Load matrices from files
    g_mat = madupite.Matrix.fromFile(
        comm,
        "g_file",
        "../examples/pendulum/data/pend_g_441_9.bin",
        madupite.MatrixCategory.Cost,
        madupite.MatrixType.Dense,
    )
    P_mat = madupite.Matrix.fromFile(
        comm,
        "P_file",
        "../examples/pendulum/data/pend_P_441_9.bin",
        madupite.MatrixCategory.Dynamics,
    )

    # Set matrices
    mdp.setStageCostMatrix(g_mat)
    mdp.setTransitionProbabilityTensor(P_mat)

    # Solve MDP
    mdp.solve()


if __name__ == "__main__":
    main()
