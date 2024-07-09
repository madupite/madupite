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
    mdp.setOption("-discount_factor", "0.9");
    mdp.setOption("-file_stats", "di_stats.json");
    mdp.setOption("-file_policy", "di_policy.out");
    mdp.setOption("-file_cost", "di_cost.out");
    mdp.setOption("-num_states", "4331");
    mdp.setOption("-num_actions", "11");

    mdp.setSourceStageCostMatrix("data/di_g_4331_11.bin");
    mdp.setSourceTransitionProbabilityTensor("data/di_P_4331_11.bin");

    mdp.solve();
}