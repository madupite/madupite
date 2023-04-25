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

/*PetscErrorCode cvgTest(KSP ksp, PetscInt n, PetscReal rnorm, KSPConvergedReason *reason, void *ctx) {
    PetscPrintf(PETSC_COMM_WORLD, "Iteration %d, Residual norm %e\n", n, rnorm);
    if (rnorm < 1e-13) {
        PetscPrintf(PETSC_COMM_WORLD, "Converged! Residual norm = %e\n", rnorm);
        *reason = KSP_CONVERGED_RTOL;
        return KSP_CONVERGED_RTOL;
    } else {
        *reason = KSP_DIVERGED_DTOL;
        return KSP_DIVERGED_DTOL;
    }
}*/

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


//    const unsigned long states = 3000;
//    const unsigned long actions = 50;
//    const double sparsityFactor = 0.02;
//    const int seed = 8624;

    const PetscInt states = 500;
    const PetscInt actions = 30;
    const double sparsityFactor = 0.02;
    const int seed = 8624;

    //PetscOptionsSetValue(NULL, "-ksp_monitor", NULL);
    PetscOptionsSetValue(NULL, "-ksp_converged_reason", NULL);
    PetscOptionsSetValue(NULL, "-ksp_monitor_true_residual", NULL);
    PetscOptionsSetValue(NULL, "-pc_type", "none"); // lu // svd
    //PetscOptionsSetValue(NULL, "-pc_svd_monitor", NULL);

/*
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
    std::string P_filename = "P_" + std::to_string(states) + "_" + std::to_string(actions) + "_" + std::to_string(sparsityFactor) + "_" + std::to_string(seed) + ".bin";
    std::string g_filename = "g_" + std::to_string(states) + "_" + std::to_string(actions) + "_" + std::to_string(seed) + ".bin";
    Mat transitionProbabilityTensor, stageCosts;
    t.start();
    matrixFromBin(transitionProbabilityTensor, "../data/" + P_filename);
    t.stop("Time to read matrix from file: ");
    t.start();
    matrixFromBin(stageCosts, "../data/" + g_filename);
    t.stop("Time to read stage costs from file: ");

    PetscInt P_rows, P_cols, numStates, numActions;
    MatGetSize(transitionProbabilityTensor, &P_rows, &P_cols);
    MatGetSize(stageCosts, &numStates, &numActions);
    assert(P_rows == numStates);
    assert(P_cols == numStates * numActions);

    // solve linear system
    auto *policy = new PetscInt[numStates];
    for(PetscInt stateInd = 0; stateInd < numStates; ++stateInd) {
        policy[stateInd] = (stateInd+3) % numActions;
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
    KSPSetOperators(ksp, jacobian, jacobian);
    KSPSetType(ksp, KSPGMRES);
    //PetscOptionsSetValue(NULL, "-ksp_view", NULL);
    //PetscOptionsSetValue(NULL, "-ksp_plot_eigenvalues", NULL);
    //PetscOptionsSetValue(NULL, "-ksp_monitor", NULL);
    //PetscOptionsSetValue(NULL, "-pc_type", "none");
    //PetscOptionsSetValue(NULL, "-ksp_converged_reason", NULL);
    KSPSetFromOptions(ksp);
    PetscReal rtol = 1e-14;
    PetscInt maxIter = 1.5 * numStates;
    //KSPSetConvergenceTest(ksp, cvgTest, PETSC_NULLPTR, PETSC_NULLPTR);
    KSPSetTolerances(ksp, rtol, 1e-20, PETSC_DEFAULT, maxIter);

    Vec V;
    VecCreateSeq(PETSC_COMM_WORLD, numStates, &V);

    // DEBUG: print dimensions of jacobian, g, V
    PetscInt jacobian_rows, jacobian_cols;
    MatGetSize(jacobian, &jacobian_rows, &jacobian_cols);
    PetscInt g_rows, V_rows;
    VecGetSize(g, &g_rows);
    VecGetSize(V, &V_rows);
    std::cout << "states: " << numStates << ", actions: " << numActions << std::endl;
    std::cout << "jacobian: " << jacobian_rows << " x " << jacobian_cols << std::endl;
    std::cout << "g: " << g_rows << std::endl;
    std::cout << "V: " << V_rows << std::endl;

    //ierr = MatMult(P, g, V);
    //CHKERRQ(ierr);

    // solve for V, store residual norm for every iteration
    std::cout << "Solving LSE with row-stochastic matrix:" << std::endl;
    ierr = KSPSolve(ksp, g, V);
    CHKERRQ(ierr);

    // generate random matrix -> non-clustered eigenvalues
    const PetscInt SEED = 98273;
    std::mt19937_64 gen(SEED);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    auto *values = new PetscScalar[numStates*numStates];
    for(PetscInt i = 0; i < numStates*numStates; ++i) {
        auto rand = dis(gen); // matrix will be in col-major order
        if(dis(gen) > sparsityFactor) {
            rand = 0;
        }
        values[i] = rand;
    }
    Mat A;
    MatCreateSeqDense(PETSC_COMM_WORLD, numStates, numStates, values, &A);

    std::cout << "Solving LSE with random matrix:" << std::endl;
    KSP ksp2;
    KSPCreate(PETSC_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, A, A);
    KSPSetType(ksp2, KSPGMRES);

    //PetscOptionsSetValue(NULL, "-ksp_view_eigenvalues", "draw");
    KSPSetFromOptions(ksp2);
    KSPSetTolerances(ksp2, rtol, 1e-20, PETSC_DEFAULT, maxIter);

    Vec V2;
    VecCreateSeq(PETSC_COMM_WORLD, numStates, &V2);

    ierr = KSPSolve(ksp2, g, V2);
    CHKERRQ(ierr);




    // destroy
    MatDestroy(&transitionProbabilityTensor);
    MatDestroy(&stageCosts);
    MatDestroy(&jacobian);
    MatDestroy(&P);
    MatDestroy(&A);
    VecDestroy(&V);
    VecDestroy(&V2);
    VecDestroy(&g);
    KSPDestroy(&ksp);

    delete[] policy;

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
// convergence test for GMRES that outputs the residual norm at each iteration
