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
#include "StageCostGenerator.h"
#include "Timer.h"

// convergence test for GMRES that outputs the residual norm at each iteration
PetscErrorCode cvgTest(KSP ksp, PetscInt n, PetscReal rnorm, KSPConvergedReason *reason, void *ctx) {
    PetscPrintf(PETSC_COMM_WORLD, "Iteration %D, Residual norm %e\n", n, rnorm);
    if (rnorm < 1e-13) {
        PetscPrintf(PETSC_COMM_WORLD, "Converged! Residual norm = %e\n", rnorm);
        return KSP_CONVERGED_RTOL;
    } else {
        return KSP_CONVERGED_ITERATING;
    }
}

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
    PetscErrorCode ierr;

    /*
    const unsigned long states = 200;
    const unsigned long actions = 20;
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

    // generate stage costs
    Mat stageCosts;
    t.start();
    generateStageCosts(stageCosts, states, actions, seed + 123, 0.2);
    t.stop("Time to generate stage costs: ");
    filename = "../data/g_" + std::to_string(states) + "_" + std::to_string(actions) + "_" + std::to_string(seed);
    std::cout << "Writing stage costs to bin:" << std::endl;
    t.start();
    matrixToBin(stageCosts, filename + ".bin");
    t.stop("Time to write stage costs to file: ");
    //matrixToAscii(stageCosts, filename + ".csv");
    */

    // read transition matrix from file
    Mat transitionProbabilityTensor, stageCosts;
    t.start();
    matrixFromBin(transitionProbabilityTensor, "../data/P_200_20_0.100000_8624.bin");
    t.stop("Time to read matrix from file: ");
    t.start();
    matrixFromBin(stageCosts, "../data/g_200_20_8624.bin");
    t.stop("Time to read stage costs from file: ");

    PetscInt P_rows, P_cols, numStates, numActions;
    MatGetSize(transitionProbabilityTensor, &P_rows, &P_cols);
    MatGetSize(stageCosts, &numStates, &numActions);
    assert(P_rows == numStates);
    assert(P_cols == numStates * numActions);

    // DEBUG
    PetscInt cols[20] = {0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19};
    PetscInt row = 0;
    PetscReal vals[20];
    MatGetValues(transitionProbabilityTensor, 1, &row, 20, cols, vals);





    // solve linear system
    auto *policy = new PetscInt[numStates];
    for(PetscInt stateInd = 0; stateInd < numStates; ++stateInd) {
        policy[stateInd] = stateInd % numActions;
    }

    // generate transition probability matrix and stage costs from policy
    Vec g;
    VecCreateSeq(PETSC_COMM_WORLD, numStates, &g);
    constructStageCostsFromPolicy(stageCosts, g, policy);
    VecAssemblyBegin(g);
    VecAssemblyEnd(g);

    Mat P;
    MatCreateSeqDense(PETSC_COMM_WORLD, numStates, numStates, PETSC_NULL, &P);
    constructTransitionProbabilityMatrixFromPolicy(transitionProbabilityTensor, P, policy);
    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);

    // generate jacobian for linear system (I - gamma * P)
    PetscScalar gamma = 0.9;
    Mat jacobian;
    MatDuplicate(P, MAT_COPY_VALUES, &jacobian);
    MatScale(jacobian, -gamma);
    MatShift(jacobian, 1.0);
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);


    // create KSP solver
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOperators(ksp, jacobian, jacobian);
    KSPSetFromOptions(ksp);
    PetscReal rtol = 1e-13;
    PetscInt maxIter = 1000;
    KSPSetConvergenceTest(ksp, cvgTest, NULL, NULL);
    //KSPSetTolerances(ksp, rtol, 0.0, PETSC_DEFAULT, maxIter);

    Vec V;
    VecDuplicate(g, &V);
    // solve for V, store residual norm for every iteration
    std::vector<PetscReal> residualNorms(maxIter);
    ierr = KSPSolve(ksp, g, V);
    CHKERRQ(ierr);

    // destroy
    MatDestroy(&transitionProbabilityTensor);
    MatDestroy(&stageCosts);
    MatDestroy(&jacobian);
    VecDestroy(&V);
    VecDestroy(&g);
    KSPDestroy(&ksp);

    delete[] policy;

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
