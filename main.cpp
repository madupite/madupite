//
// Benchmarking Distributed Inexact Policy Iteration for Large-Scale Markov Decision Processes
// Bachelor Thesis - Robin Sieber - 2023 - ETH Zürich
//

#include <petsc.h>
#include <iostream>
#include "MDP.h"

int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    
    MDP mdp;
    mdp.setValuesFromOptions();
    mdp.loadFromBinaryFile();
    mdp.inexactPolicyIteration();
    mdp.setOption("-discountFactor", "0.999");
    mdp.inexactPolicyIteration();
    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
