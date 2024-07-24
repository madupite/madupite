#include "MDP.h"
#include <cassert>
#include <petsc.h>
#include <string>

int main(int argc, char** argv)
{
    // Initialize MPI, PETSc and Madupite, passing command line arguments.
    auto madupite = Madupite::initialize(&argc, &argv);

    MDP mdp(madupite);
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-max_iter_pi", "100");
    mdp.setOption("-max_iter_ksp", "2000");
    mdp.setOption("-alpha", "1e-5");
    mdp.setOption("-atol_pi", "1e-7");
    mdp.setOption("-ksp_type", "tfqmr");
    mdp.setOption("-discount_factor", "0.999");
    mdp.setOption("-file_stats", "pend_stats.json");
    mdp.setOption("-file_policy", "pend_policy.out");
    mdp.setOption("-file_cost", "pend_cost.out");
    mdp.setOption("-export_optimal_transition_probabilities", "pend_Ppi.out");
    mdp.setOption("-export_optimal_stage_costs", "pend_gpi.out");
    // mdp.setOption("-num_states", "441");
    // mdp.setOption("-num_actions", "9");

    auto comm = PETSC_COMM_WORLD;

    auto g_mat = Matrix::fromFile(comm, "g_file", "data/pend_g_441_9.bin", MatrixCategory::Cost, MatrixType::Dense);
    auto P_mat = Matrix::fromFile(comm, "P_file", "data/pend_P_441_9.bin", MatrixCategory::Dynamics);

    mdp.setStageCostMatrix(&g_mat);
    mdp.setTransitionProbabilityTensor(&P_mat);

    mdp.solve();
}
