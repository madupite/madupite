//
// Created by robin on 02.04.23.
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

    Vec V0;
    VecCreateMPI(PETSC_COMM_WORLD, mdp.localNumStates_, mdp.numStates_, &V0);
    VecSet(V0, 1.0);

    IS optimalPolicy;
    Vec optimalCost;

    t.start();
    //mdp.inexactPolicyIteration(V0, optimalPolicy, optimalCost);
    mdp.benchmarkIPI(V0, optimalPolicy, optimalCost);
    t.stop("iPI took: ");

    t.start();
    mdp.writeVec(optimalCost, mdp.file_cost_);
    mdp.writeIS(optimalPolicy, mdp.file_policy_);
    t.stop("Writing took: ");

    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
