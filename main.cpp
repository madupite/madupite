//
// Benchmarking Distributed Inexact Policy Iteration for Large-Scale Markov Decision Processes
// Bachelor Thesis - Robin Sieber - 2023 - ETH Zürich
//

#include <petscvec.h>
#include <mpi.h>
#include <iostream>
#include <random>

#include "utils/Timer.h"
#include "utils/Logger.h"
#include "MDP.h"

int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    Timer t;

    // Setup MDP
    t.start();
    MDP mdp;
    mdp.setValuesFromOptions();
    mdp.loadFromBinaryFile(mdp.file_P_, mdp.file_g_);
    t.stop("MDP setup + loading took: ");


    t.start();
    mdp.inexactPolicyIteration();
    //mdp.benchmarkIPI(V0, optimalPolicy, optimalCost);
    t.stop("iPI took: ");

    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
