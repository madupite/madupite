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

    // Get the indices from the index set
    const PetscInt *indices;
    ISGetIndices(optimalPolicy, &indices);

// Get the local size of the index set
    PetscInt localSize;
    ISGetLocalSize(optimalPolicy, &localSize);

// Gather all indices on process 0
    PetscInt *allIndices = NULL;
    PetscInt *recvcounts = NULL;
    PetscInt *displs = NULL;
    if (mdp.rank_ == 0) {
        allIndices = new PetscInt[mdp.numStates_]; // globalSize should be the total number of indices across all processes
        recvcounts = new int[mdp.size_];
        displs = new int[mdp.size_];
    }
    MPI_Gather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if (mdp.rank_ == 0) {
        displs[0] = 0;
        for (int i = 1; i < mdp.size_; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    MPI_Gatherv(indices, localSize, MPI_INT, allIndices, recvcounts, displs, MPI_INT, 0, PETSC_COMM_WORLD);

// Process 0 writes the indices to a CSV file
    if (mdp.rank_ == 0) {
        std::ofstream file("indices.csv");
        for (int i = 0; i < mdp.numStates_; i++) {
            file << allIndices[i] << ' ';
        }
        file.close();

        delete[] allIndices;
        delete[] recvcounts;
        delete[] displs;
    }

// Restore the indices
    ISRestoreIndices(optimalPolicy, &indices);


    VecDestroy(&V0);
    ISDestroy(&optimalPolicy);
    VecDestroy(&optimalCost);
    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
