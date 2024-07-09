//
// Benchmarking Distributed Inexact Policy Iteration for Large-Scale Markov Decision Processes
// Bachelor Thesis - Robin Sieber - 2023 - ETH Zürich
//

#include "MDP.h"
#include <iostream>
#include <random>

double g(int s, int a) { return 1.0; }

double P(int s, int a, int s_prime)
{
    if (s == s_prime) {
        return 1.0;
    }
    return 0.0;
}

int main(int argc, char** argv)
{
    // Initialize MPI, PETSc and Madupite, passing command line arguments.
    auto madupite = Madupite::initialize(&argc, &argv);

    // Initialize MDP.
    // madupite can be obtained also with Madupite::get()
    MDP mdp(madupite);
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-discount_factor", "0.9");
    mdp.setOption("-max_iter_pi", "20");
    mdp.setOption("-max_iter_ksp", "1000");
    mdp.setOption("-alpha", "1e-4");
    mdp.setOption("-atol_pi", "1e-10");
    mdp.setOption("-file_stats", "stats.json");
    mdp.setOption("-file_policy", "policy.out");
    mdp.setOption("-file_cost", "cost.out");
    // mdp.setOption("-numStates", "100");
    // mdp.setOption("-numActions", "20");

    mdp.setSourceStageCostMatrix("100_50_0.1/g.bin");
    mdp.setSourceTransitionProbabilityTensor("100_50_0.1/P.bin");

    std::cout << "File loaded." << std::endl;
    // mdp.generateCostMatrix(g);
    // mdp.generateTransitionProbabilityTensor(P, 1, nullptr, 1, nullptr);

    mdp.solve();
    std::cout << "Inext policy iteration #1 done." << std::endl;
    mdp.setOption("-discount_factor", "0.999"); // doesn't work anymore -> change setOptions to update member variables
    mdp.setOption("-ksp_type", "tfqmr");
    mdp.solve();
    std::cout << "Inext policy iteration #2 done." << std::endl;
    return 0;
}
