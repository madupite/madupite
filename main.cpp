//
// Created by robin on 02.04.23.
//

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <iostream>
#include <random>
#include "PETScFunctions.h"
#include "Timer.h"

int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

    Timer t;

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

    // Set the right-hand side
    KSPSetFromOptions(ksp);
    KSPSetConvergenceTest(ksp, my_convergence_test, nullptr, nullptr);
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    // use L2 norm (KSPMonitorTrueResidualNorm)

    // Solve the system
    std::cout << "Solve..." << std::endl;
    KSPSolve(ksp, b, x);
    t.stop("Solving using GMRES took: ");

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

    // Finalize PETSc
    PetscFinalize();

    return 0;
}
