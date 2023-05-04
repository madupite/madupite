//
// Created by robin on 02.04.23.
//

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <iostream>
#include <mpi.h>
#include <random>
#include "Timer.h"
#include "MDP.h"


int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

    // print how many processors are used
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if(rank == 0) {
        int size;
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        std::cout << "Number of processors: " << size << std::endl;
    }

    MDP mdp(200, 20, 0.9); // sparsity factor = 0.1
    //MDP mdp(500, 30, 0.9); // sparsity factor = 0.02
    std::cout << mdp.numStates_ << std::endl;
    std::cout << mdp.numActions_ << std::endl;
    std::cout << mdp.discountFactor_ << std::endl;

    PetscReal sparsityFactor = 0.1;
    PetscInt seed = 8624;
    std::string P_filename = "../data/P_" + std::to_string(mdp.numStates_) + "_" + std::to_string(mdp.numActions_) + "_" + std::to_string(sparsityFactor) + "_" + std::to_string(seed) + ".bin";
    std::string g_filename = "../data/g_" + std::to_string(mdp.numStates_) + "_" + std::to_string(mdp.numActions_) + "_" + std::to_string(seed) + ".bin";

    mdp.loadFromBinaryFile(P_filename, g_filename);

    Vec V;
    VecCreate(PETSC_COMM_WORLD, &V);
    VecSetType(V, VECSEQ);
    VecSetSizes(V, PETSC_DECIDE, mdp.numStates_);
    VecSet(V, 1.0);
    PetscInt *policy = new PetscInt[mdp.numStates_];

    Timer t;
    t.start();
    auto result = mdp.inexactPolicyIteration(V, 10, 0.001);
    t.stop("iPI took: ");
    std::cout << "Policy: " << std::endl;
    for(auto x : result) std::cout << x << " ";


    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
