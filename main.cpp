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

    /*
    int rows = 200;
    int cols = 200;
    int seed = 8624;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0, 1);

    auto MatValues = new PetscScalar[rows*cols];
    std::cout << "Generating matrix values..." << std::endl;
    for (int i = 0; i < rows*cols; i++) {
        double rand = dis(gen);
        if(dis(gen) < 0.3) rand = 0;
        else if (dis(gen) > 0.8) rand *= 10;
        MatValues[i] = rand;
    }
    std::cout << "Generating vector values..." << std::endl;
    auto VecValues = new double[rows];
    for (int i = 0; i < rows; i++) {
        double rand = dis(gen);
        if(rand < 0.4) rand *= 10; // perturb uniformity
        VecValues[i] = rand;
    }

    Mat A;
    std::cout << "Generating matrix..." << std::endl;
    generateDenseMatrix(A, rows, cols, MatValues);
    Vec b;
    Vec b2;
    std::cout << "Generating vector..." << std::endl;
    fillVector(b, rows, VecValues);
    VecDuplicate(b, &b2);
    VecCopy(b, b2);
    Vec x;
    Vec x2;
    VecDuplicate(b, &x);
    VecDuplicate(b, &x2);

    // Solve Ax = b using GMRES
    // Create the linear solver context
    std::cout << "Setup solver..." << std::endl;
    t.start();
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);

    // Set the GMRES method
    KSPSetType(ksp, KSPGMRES);

    // Set the matrix
    KSPSetOperators(ksp, A, A);

    // set tolerance
    const double rtol = 1e-14;
    const double atol = 0.0;
    const unsigned long maxIter = 1000;
    KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, maxIter);

    // Set the right-hand side
    KSPSetFromOptions(ksp);
    //KSPSetConvergenceTest(ksp, my_convergence_test, nullptr, nullptr);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    // use L2 norm (KSPMonitorTrueResidualNorm)

    // Solve the system
    std::cout << "Solve..." << std::endl;
    KSPSolve(ksp, b, x);
    t.stop("Solving using GMRES took: ");
    int iterations;
    KSPGetIterationNumber(ksp, &iterations);
    std::cout << "Number of iterations: " << iterations << std::endl;

    // solve exactly A x2 = b2 using LU factorization
    std::cout << "Solve exactly..." << std::endl;
    t.start();
    MatLUFactor(A, nullptr, nullptr, nullptr);
    MatSolve(A, b2, x2);
    t.stop("Solving exactly took: ");

    // Check the solution
    VecAXPY(x, -1, x2);
    PetscReal norm;
    VecNorm(x, NORM_2, &norm);
    std::cout << "Norm of error: " << norm << std::endl;


    // Free the memory
    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    VecDestroy(&x2);
    VecDestroy(&b2);
    */

    const unsigned long states = 500;
    const unsigned long actions = 40;
    const double sparsityFactor = 0.2;
    const int seed = 8624;


    // generate transition matrix
    Mat A;
    generateTransitionMatrix(A, states, actions, sparsityFactor, seed);

    // write transition matrix to file
    std::string filename = "../data/P_" + std::to_string(states) + "_" + std::to_string(actions);
    std::cout << "Writing matrix to csv:" << std::endl;
    matrixToAscii(A, filename + ".csv");
    std::cout << "Writing matrix to bin:" << std::endl;
    matrixToBin(A, filename + ".bin");

    // check row-stochasticity
    std::cout << "Checking row-stochasticity..." << std::endl;
    Vec ones;
    Vec rowSum;
    Vec solution;
    VecCreate(PETSC_COMM_WORLD, &ones);
    VecCreate(PETSC_COMM_WORLD, &rowSum);
    VecCreate(PETSC_COMM_WORLD, &solution);
    VecSetSizes(ones, states*actions, PETSC_DECIDE);
    VecSetSizes(rowSum, states, PETSC_DECIDE);
    VecSetSizes(solution, states, PETSC_DECIDE);
    VecSetFromOptions(ones);
    VecSetFromOptions(rowSum);
    VecSetFromOptions(solution);
    VecSet(ones, 1);
    VecSet(solution, actions);
    // compute row-wise sum
    MatMult(A, ones, rowSum);
    // subtract solution, print inf norm
    VecAXPY(rowSum, -1, solution);
    PetscReal norm;
    VecNorm(rowSum, NORM_INFINITY, &norm);
    std::cout << "Row-stochasticity error: " << norm << std::endl;



    // destroy matrix
    MatDestroy(&A);
    // Finalize PETSc
    PetscFinalize();

    return 0;
}
