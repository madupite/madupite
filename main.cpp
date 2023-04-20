//
// Created by robin on 02.04.23.
//

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <iostream>
#include <mpi.h>
#include <random>
#include "PETScFunctions.h"
#include "Filewriter.h"
#include "TransitionMatrixGenerator.h"
#include "Timer.h"

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

    Timer t;

    const unsigned long states = 2000;
    const unsigned long actions = 50;
    const double sparsityFactor = 0.1;
    const int seed = 8624;

    // generate transition matrix
    Mat A;
    t.start();
    generateTransitionMatrix(A, states, actions, sparsityFactor, seed);
    t.stop("Time to generate transition matrix: ");

    // write transition matrix to file
    std::string filename = "../data/P_" + std::to_string(states) + "_" + std::to_string(actions) + "_" + std::to_string(sparsityFactor) + "_" + std::to_string(seed);
    std::cout << "Writing matrix to bin:" << std::endl;
    t.start();
    matrixToBin(A, filename + ".bin");
    t.stop("Time to write matrix to file: ");
    //std::cout << "Writing matrix to csv:" << std::endl;
    //matrixToAscii(A, filename + ".csv");

    std::cout << "Checking row-stochasticity..." << std::endl;
    t.start();
    double error = checkRowStochasticity(A, states, actions);
    std::cout << "Row-stochasticity error: " << error << std::endl;
    t.stop("Time to check row-stochasticity: ");

    // destroy matrix
    MatDestroy(&A);
    // Finalize PETSc
    PetscFinalize();

    return 0;
}
