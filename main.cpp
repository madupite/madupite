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

#include "Timer.h"
#include "MDP.h"


int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

    // print how many processors are used
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    PetscPrintf(PETSC_COMM_WORLD, "Number of processors: %d\n", size);



    MDP mdp(5000, 20, 0.9); // sparsity factor = 0.1
    PetscReal sparsityFactor = 0.03;
    //MDP mdp(500, 50, 0.9); // sparsity factor = 0.01
    //PetscReal sparsityFactor = 0.05;
    //MDP mdp(5000, 40, 0.9); // sparsity factor = 0.01
    //PetscReal sparsityFactor = 0.01;
    //MDP mdp(3000, 50, 0.9); // sparsity factor = 0.02
    std::cout << mdp.numStates_ << std::endl;
    std::cout << mdp.numActions_ << std::endl;
    std::cout << mdp.discountFactor_ << std::endl;


    PetscInt seed = 8624;
    std::string path = "../data/" + std::to_string(mdp.numStates_) + "_" + std::to_string(mdp.numActions_) + "_" + std::to_string(sparsityFactor) + "/";


    Timer t;
    t.start();
    mdp.loadFromBinaryFile(path + "P.bin", path + "g.bin", path + "nnz.bin");
    t.stop("Loading took: ");


    // solve MDP
    Vec V;
    VecCreate(PETSC_COMM_WORLD, &V);
    VecSetType(V, VECSEQ);
    VecSetSizes(V, PETSC_DECIDE, mdp.numStates_);
    VecSet(V, 1.0);
    //PetscInt *policy = new PetscInt[mdp.numStates_];

    //Timer t;
    t.start();
    auto result = mdp.inexactPolicyIteration(V, 5, 0.001);
    t.stop("iPI took: ");
    std::cout << "Policy: " << std::endl;
    for(auto x : result) std::cout << x << " ";


    mdp.~MDP();

    /*
    PetscInt *policy = new PetscInt[mdp.numStates_];
    PetscInt iterations = 5;

    t.start();
    Vec V;
    VecCreate(PETSC_COMM_WORLD, &V);
    VecSetType(V, VECSEQ);
    VecSetSizes(V, PETSC_DECIDE, mdp.numStates_);
    PetscReal *values = new PetscReal[mdp.numStates_];
    std::mt19937_64 gen(23987);
    std::uniform_real_distribution<PetscReal> dis(0, 12);
    for(PetscReal *it = values; it != values + mdp.numStates_; ++it) {
        *it = dis(gen);
    }
    PetscInt *indices = new PetscInt[mdp.numStates_];
    std::iota(indices, indices + mdp.numStates_, 0);
    VecSetValues(V, mdp.numStates_, indices, values, INSERT_VALUES);
    t.stop("Initializing V took: ");
    // benchmarking two versions of extractGreedyPolicy

    t.start();
    for(PetscInt i = 0; i < iterations; ++i) {
        mdp.extractGreedyPolicy(V, policy, MDP::V1);
        std::fill(policy, policy + mdp.numStates_, 0); // reset
    }
    t.stop("5x V1 took: ");

    t.start();
    for(PetscInt i = 0; i < iterations; ++i) {
        mdp.extractGreedyPolicy(V, policy, MDP::V2);
        std::fill(policy, policy + mdp.numStates_, 0); // reset
    }
    t.stop("5x V2 took: ");

    delete[] values;
    delete[] policy;
*/
    // Finalize PETSc
    PetscFinalize();
    return 0;
}
