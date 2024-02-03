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
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    
    MDP mdp;
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-discountFactor", "0.9");
    mdp.setOption("-maxIter_PI", "20");
    mdp.setOption("-maxIter_KSP", "1000");
    mdp.setOption("-numPIRuns", "1");
    mdp.setOption("-rtol_KSP", "1e-4");
    mdp.setOption("-atol_PI", "1e-10");
    // mdp.setOption("-file_P", "../example/100_50_0.1/P.bin");
    // mdp.setOption("-file_g", "../example/100_50_0.1/g.bin");
    mdp.setOption("-file_stats", "stats.json");
    mdp.setOption("-file_policy", "policy.out");
    mdp.setOption("-file_cost", "cost.out");
    mdp.setOption("-numStates", "100");
    mdp.setOption("-numActions", "20");

    mdp.setValuesFromOptions();
    // mdp.loadFromBinaryFile();

    mdp.generateCostMatrix(g);
    mdp.generateTransitionProbabilityTensor(P, 1, nullptr, 1, nullptr);

    mdp.inexactPolicyIteration();
    mdp.setOption("-discountFactor", "0.999"); // doesn't work anymore -> change setOptions to update member variables
    mdp.inexactPolicyIteration();
    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
