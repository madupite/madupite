//
// Created by robin on 02.04.23.
//

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petsc.h>

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

    //MDP mdp(5000, 20, 0.9); // sparsity factor = 0.1
    //PetscReal sparsityFactor = 0.03;
    MDP mdp(100, 10, 0.9); // sparsity factor = 0.1
    PetscReal sparsityFactor = 0.1;
    //MDP mdp(2000, 40, 0.9);
    //PetscReal sparsityFactor = 0.03;
    //MDP mdp(500, 50, 0.9); // sparsity factor = 0.01
    //PetscReal sparsityFactor = 0.05;
    //MDP mdp(5000, 40, 0.9); // sparsity factor = 0.01
    //PetscReal sparsityFactor = 0.01;
    //MDP mdp(3000, 50, 0.9); // sparsity factor = 0.02
    PetscPrintf(PETSC_COMM_WORLD, "States: %d, Actions: %d, Sparsity factor: %f\n", mdp.numStates_, mdp.numActions_, sparsityFactor);

    std::string path = "../data/" + std::to_string(mdp.numStates_) + "_" + std::to_string(mdp.numActions_) + "_" + std::to_string(sparsityFactor) + "/";


    Timer t;
    t.start();
    mdp.loadFromBinaryFile(path + "P.bin", path + "g.bin", path + "nnz.bin");
    t.stop("Loading took: ");

    Vec V0;
    VecCreateMPI(PETSC_COMM_WORLD, mdp.localNumStates_, mdp.numStates_, &V0);
    VecSet(V0, 1.0);

    IS optimalPolicy;
    Vec optimalCost;
    t.start();
    mdp.inexactPolicyIteration(V0, 15, 0.001, optimalPolicy, optimalCost);
    t.stop("iPI took: ");

    // output solutions
    PetscPrintf(PETSC_COMM_WORLD, "Optimal cost:\n");
    VecView(optimalCost, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Optimal policy:\n");
    ISView(optimalPolicy, PETSC_VIEWER_STDOUT_WORLD);

    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
