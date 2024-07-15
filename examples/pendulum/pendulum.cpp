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
    mdp.setOption("-max_iter_pi", "200");
    mdp.setOption("-max_iter_ksp", "2000");
    mdp.setOption("-alpha", "1e-2");
    mdp.setOption("-atol_pi", "1e-7");
    mdp.setOption("-ksp_type", "tfqmr");
    mdp.setOption("-discount_factor", "0.999");
    mdp.setOption("-file_stats", "pend_stats.json");
    mdp.setOption("-file_policy", "pend_policy.out");
    mdp.setOption("-file_cost", "pend_cost.out");
    mdp.setOption("-num_states", "160801");
    mdp.setOption("-num_actions", "201");

    mdp.setSourceStageCostMatrix("data/pend_g_160801_201.bin");
    mdp.setSourceTransitionProbabilityTensor("data/pend_P_160801_201.bin");

    mdp.solve();
}
