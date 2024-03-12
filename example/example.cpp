//
// Benchmarking Distributed Inexact Policy Iteration for Large-Scale Markov Decision Processes
// Bachelor Thesis - Robin Sieber - 2023 - ETH Zürich
//

#include <petsc.h>
#include <iostream>
#include "MDP.h"

#include<random>

double g(int s, int a) {
    return 1.0;
}

double P(int s, int a, int s_prime) {
    if (s == s_prime) {
        return 1.0;
    }
    return 0.0;
}


int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR);
    
    MDP mdp;
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-discount_factor", "0.9");
    mdp.setOption("-max_iter_pi", "20");
    mdp.setOption("-max_iter_ksp", "1000");
    mdp.setOption("-num_pi_runs", "1");
    mdp.setOption("-rtol_ksp", "1e-4");
    mdp.setOption("-atol_pi", "1e-10");
    mdp.setOption("-file_probabilities", "../example/100_50_0.1/P.bin");
    mdp.setOption("-file_costs", "../example/100_50_0.1/g.bin");
    mdp.setOption("-file_stats", "stats.json");
    mdp.setOption("-file_policy", "policy.out");
    mdp.setOption("-file_cost", "cost.out");
    // mdp.setOption("-numStates", "100");
    // mdp.setOption("-numActions", "20");

    mdp.setValuesFromOptions();
    mdp.loadFromBinaryFile();

    // mdp.generateCostMatrix(g);
    // mdp.generateTransitionProbabilityTensor(P, 1, nullptr, 1, nullptr);

    mdp.inexactPolicyIteration();
    std::cout << "done nr. 1" << std::endl;
    mdp.setOption("-discount_factor", "0.999"); // doesn't work anymore -> change setOptions to update member variables
    mdp.setOption("-ksp_type", "tfqmr");
    mdp.setValuesFromOptions();
    mdp.inexactPolicyIteration();
    std::cout << "done nr. 2" << std::endl;
    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
