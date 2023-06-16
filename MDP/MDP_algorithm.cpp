//
// Created by robin on 27.04.23.
//

#include "MDP.h"
#include <petscksp.h>
#include <mpi.h>
#include <iostream>
#include "../utils/Logger.h"
#include <chrono>
#include <algorithm>

// find $\argmin_{\pi} \{ g^\pi + \gamma P^\pi V \}$
// PRE: policy is a array of size localNumStates_ and must be allocated. Function will write into it but not allocate it.
PetscErrorCode MDP::extractGreedyPolicy(Vec &V, PetscInt *policy) {

    PetscErrorCode ierr;
    const PetscReal *costValues; // stores cost (= g + gamma PV) values for each state
    PetscReal *minCostValues; // stores minimum cost values for each state
    PetscMalloc1(localNumStates_, &minCostValues);
    std::fill(minCostValues, minCostValues + localNumStates_, std::numeric_limits<PetscReal>::max());

    Mat P;
    Vec g;

    for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
        //LOG("action " + std::to_string(actionInd) + " of " + std::to_string(numActions_) + "...");
        constructFromPolicy(actionInd, P, g); // creates P and g => need to destroy P and g by ourselves
        //LOG("Finished construction of P and g. Calculating g + gamma PV...");
        ierr = MatScale(P, discountFactor_); CHKERRQ(ierr);
        ierr = MatMultAdd(P, V, g, g); CHKERRQ(ierr);

        ierr = VecGetArrayRead(g, &costValues); CHKERRQ(ierr);

        //LOG("Performing minimization for all local states...");
        for (int localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
            if (costValues[localStateInd] < minCostValues[localStateInd]) {
                minCostValues[localStateInd] = costValues[localStateInd];
                policy[localStateInd] = actionInd;
            }
        }
        ierr = VecRestoreArrayRead(g, &costValues); CHKERRQ(ierr);
        //LOG("Finished minimization for all local states.");
        ierr = MatDestroy(&P); CHKERRQ(ierr);
        ierr = VecDestroy(&g); CHKERRQ(ierr);
    }
    ierr = PetscFree(costValues); CHKERRQ(ierr);

    return 0;
}

// user must destroy P and g by himself. Function will create them. [used in extractGreedyPolicy]
PetscErrorCode MDP::constructFromPolicy(const PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts) {
    //LOG("Entering constructFromPolicy [actionInd]");

    // compute where local ownership of new P_pi matrix starts
    PetscInt P_pi_start; // start of ownership of new matrix (to be created)
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_pi_start, NULL);
    P_pi_start /= numActions_;

    // allocate memory for values
    PetscInt *P_rowIndexValues;
    PetscMalloc1(localNumStates_, &P_rowIndexValues);
    PetscReal *g_pi_values;
    PetscMalloc1(localNumStates_, &g_pi_values);

    // compute global row indices for P and get values for g_pi
    PetscInt g_srcRow;
    for(PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        // compute values for row index set
        P_rowIndexValues[localStateInd] = P_start_ + localStateInd * numActions_ + actionInd;
        // get values for stageCosts
        g_srcRow  = g_start_ + localStateInd;
        MatGetValue(stageCostMatrix_, g_srcRow, actionInd, &g_pi_values[localStateInd]);
    }

    // generate index sets
    IS P_rowIndices;
    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, P_rowIndexValues, PETSC_COPY_VALUES, &P_rowIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    //LOG("Creating transitionProbabilities submatrix");
    MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, NULL, MAT_INITIAL_MATRIX, &transitionProbabilities);

    //LOG("Creating stageCosts vector");
    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_, numStates_, &stageCosts);
    const PetscInt *g_pi_rowIndexValues; // global indices
    ISGetIndices(g_pi_rowIndices, &g_pi_rowIndexValues);
    VecSetValues(stageCosts, localNumStates_, g_pi_rowIndexValues, g_pi_values, INSERT_VALUES);
    ISRestoreIndices(g_pi_rowIndices, &g_pi_rowIndexValues);

    ISDestroy(&P_rowIndices);
    ISDestroy(&g_pi_rowIndices);
    PetscFree(P_rowIndexValues);
    PetscFree(g_pi_values);
    return 0;
}

// user must destroy P and g by himself. Function will create them. [used in inexactPolicyIteration]
PetscErrorCode MDP::constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    //LOG("Entering constructFromPolicy [policy]");

    // compute where local ownership of new P_pi matrix starts
    PetscInt P_pi_start; // start of ownership of new matrix (to be created)
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_pi_start, NULL);
    P_pi_start /= numActions_;

    // allocate memory for values
    PetscInt *P_rowIndexValues;
    PetscMalloc1(localNumStates_, &P_rowIndexValues);
    PetscReal *g_pi_values;
    PetscMalloc1(localNumStates_, &g_pi_values);

    // compute global row indices for P and get values for g_pi
    PetscInt g_srcRow, actionInd;
    for(PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        actionInd = policy[localStateInd];
        // compute values for row index set
        P_rowIndexValues[localStateInd] = P_start_ + localStateInd * numActions_ + actionInd;
        // get values for stageCosts
        g_srcRow  = g_start_ + localStateInd;
        MatGetValue(stageCostMatrix_, g_srcRow, actionInd, &g_pi_values[localStateInd]);
    }

    // generate index sets
    IS P_rowIndices;
    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, P_rowIndexValues, PETSC_COPY_VALUES, &P_rowIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    //LOG("Creating transitionProbabilities submatrix");
    MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, NULL, MAT_INITIAL_MATRIX, &transitionProbabilities);

    //LOG("Creating stageCosts vector");
    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_, numStates_, &stageCosts);
    const PetscInt *g_pi_rowIndexValues; // global indices
    ISGetIndices(g_pi_rowIndices, &g_pi_rowIndexValues);
    VecSetValues(stageCosts, localNumStates_, g_pi_rowIndexValues, g_pi_values, INSERT_VALUES);
    ISRestoreIndices(g_pi_rowIndices, &g_pi_rowIndexValues);

    //LOG("Assembling transitionProbabilities and stageCosts");
    MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(stageCosts);
    MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(stageCosts);

    ISDestroy(&P_rowIndices);
    ISDestroy(&g_pi_rowIndices);
    PetscFree(P_rowIndexValues);
    PetscFree(g_pi_values);
    return 0;
}


PetscErrorCode MDP::iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, KSPContext &ctx) {
    PetscErrorCode ierr;
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);

    // ksp solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, jacobian, jacobian); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    //ierr = KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetConvergenceTest(ksp, &MDP::cvgTest, &ctx, NULL); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, stageCosts, V); CHKERRQ(ierr);
    //ierr = KSPGetIterationNumber(ksp, &iter); CHKERRQ(ierr);
    //LOG("KSP converged after " + std::to_string(iter) + " iterations");

    ierr = KSPGetIterationNumber(ksp, &ctx.kspIterations); CHKERRQ(ierr);
    //LOG("KSP iterations: " + std::to_string(ctx.kspIterations) + " (max: " + std::to_string(ctx.maxIter) + ")");
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    return ierr;
}

// defines the matrix-vector product for the jacobian shell
void MDP::jacobianMultiplication(Mat mat, Vec x, Vec y) {
    JacobianContext *ctx;
    MatShellGetContext(mat, (void **) &ctx); // todo static cast
    MatMult(ctx->P_pi, x, y);
    VecScale(y, -ctx->discountFactor);
    VecAXPY(y, 1.0, x);
}

// creates MPIAIJ matrix and computes jacobian = I - gamma * P_pi
PetscErrorCode MDP::createJacobian(Mat &jacobian, const Mat &transitionProbabilities, JacobianContext &ctx) {
    PetscErrorCode ierr;
    ierr = MatCreateShell(PETSC_COMM_WORLD, localNumStates_, localNumStates_, numStates_, numStates_, &ctx, &jacobian); CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobian, MATOP_MULT, (void (*)(void)) jacobianMultiplication); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode MDP::inexactPolicyIteration(Vec &V0, IS &policy, Vec &optimalCost) {
    LOG("Entering inexactPolicyIteration");
    jsonWriter_->add_solver_run();

    PetscErrorCode ierr;

    Vec V;
    VecDuplicate(V0, &V);
    VecCopy(V0, V);

    Mat transitionProbabilities, jacobian;
    Vec stageCosts;
    PetscInt *policyValues;
    PetscMalloc1(localNumStates_, &policyValues);

    PetscReal residualNorm;

    // initialize policy iteration
    extractGreedyPolicy(V, policyValues);
    constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
    JacobianContext ctxJac = {transitionProbabilities, discountFactor_};
    createJacobian(jacobian, transitionProbabilities, ctxJac);
    computeResidualNorm(jacobian, V, stageCosts, &residualNorm); // for KSP convergence test

    PetscLogDouble startTime, endTime;
    PetscInt i = 1;
    for(; i <= maxIter_PI_; ++i) {
        LOG("Iteration " + std::to_string(i) + " residual norm: " + std::to_string(residualNorm));
        PetscTime(&startTime);

        KSPContext ctx = {maxIter_KSP_, residualNorm * rtol_KSP_, -1};

        // solve linear system
        iterativePolicyEvaluation(jacobian, stageCosts, V, ctx);
        MatDestroy(&transitionProbabilities);
        MatDestroy(&jacobian); // avoid memory leak
        VecDestroy(&stageCosts);

        // compute jacobian wrt new policy
        extractGreedyPolicy(V, policyValues);
        constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
        ctxJac = {transitionProbabilities, discountFactor_};
        createJacobian(jacobian, transitionProbabilities, ctxJac);
        computeResidualNorm(jacobian, V, stageCosts, &residualNorm); // used for outer loop stopping criterion + RHS of KSP stopping criterion

        PetscTime(&endTime);

        jsonWriter_->add_iteration_data(i, ctx.kspIterations, (endTime - startTime) * 1000, residualNorm);

        if(residualNorm < atol_PI_) {
            break;
        }
    }
    if(i > maxIter_PI_) {
        LOG("Warning: maximum number of PI iterations reached");
    }

    jsonWriter_->write_to_file(file_stats_);

    MatDestroy(&transitionProbabilities);
    MatDestroy(&jacobian);
    VecDestroy(&stageCosts);

    // output results
    VecDuplicate(V, &optimalCost);
    VecCopy(V, optimalCost);
    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, policyValues, PETSC_COPY_VALUES, &policy);

    VecDestroy(&V);
    PetscFree(policyValues);

    return 0;
}

PetscErrorCode MDP::computeResidualNorm(Mat J, Vec V, Vec g, PetscReal *rnorm) {
    // compute residual norm ||g - J*V||_\infty
    PetscErrorCode ierr;
    Vec res;
    VecDuplicate(g, &res);
    MatMult(J, V, res);
    VecAXPY(res, -1, g);
    VecNorm(res, NORM_INFINITY, rnorm);
    VecDestroy(&res);
    return ierr;
}


PetscErrorCode MDP::cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx) {
    PetscErrorCode ierr;
    PetscReal threshold = static_cast<KSPContext*>(ctx)->threshold;
    PetscReal norm;

    Vec res;
    //ierr = VecDuplicate(ksp->vec_rhs, &res); CHKERRQ(ierr);
    ierr = KSPBuildResidual(ksp, NULL, NULL, &res); CHKERRQ(ierr);
    ierr = VecNorm(res, NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = VecDestroy(&res); CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_WORLD, "it = %d: residual norm = %f\n", it, norm);

    if(it == 0) *reason = KSP_CONVERGED_ITERATING;
    else if(norm < threshold) *reason = KSP_CONVERGED_RTOL;
    else if(it >= static_cast<KSPContext*>(ctx)->maxIter) *reason = KSP_DIVERGED_ITS;
    else *reason = KSP_CONVERGED_ITERATING;

    return 0;
}
