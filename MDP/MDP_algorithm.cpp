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
// idea: Mult gamma * P*V, reshape, add g, use MatGetRowMin
PetscErrorCode MDP::extractGreedyPolicy(const Vec &V, PetscInt *policy, PetscReal &residualNorm) {
    PetscErrorCode ierr;
//    Vec costVector_;
//    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_*numActions_, numStates_*numActions_, &costVector_);
    ierr = MatMult(transitionProbabilityTensor_, V, costVector_); CHKERRQ(ierr);
    VecScale(costVector_, discountFactor_);

    // reshape costVector_ into costMatrix_
    // create + preallocate new matrix
//    Mat costMatrix_;
//    MatCreate(PETSC_COMM_WORLD, &costMatrix_);
//    MatSetSizes(costMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_);
//    MatSetType(costMatrix_, MATDENSE);
//    MatSetUp(costMatrix_);
    //ierr = MatMPIAIJSetPreallocation(costMatrix_, localNumActions, NULL, numActions_ - localNumActions, NULL); CHKERRQ(ierr); // preallocate dense matrix

    // reshape costVector_ into costMatrix_ (fill matrix with values)
    const PetscReal *costVectorValues;
    VecGetArrayRead(costVector_, &costVectorValues);
    IS rows, cols;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &rows);
    ISCreateStride(PETSC_COMM_WORLD, numActions_, 0, 1, &cols);
    const PetscInt *rowIndices, *colIndices;
    ISGetIndices(rows, &rowIndices);
    ISGetIndices(cols, &colIndices);
    //ierr = MatSetValuesIS(costMatrix_, rows, cols, costVectorValues, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValues(costMatrix_, localNumStates_, rowIndices, numActions_, colIndices, costVectorValues, INSERT_VALUES); CHKERRQ(ierr);
    MatAssemblyBegin(costMatrix_, MAT_FINAL_ASSEMBLY);
    VecRestoreArrayRead(costVector_, &costVectorValues);
    ISRestoreIndices(rows, &rowIndices);
    ISRestoreIndices(cols, &colIndices);
    MatAssemblyEnd(costMatrix_, MAT_FINAL_ASSEMBLY);
    ISDestroy(&rows);
    ISDestroy(&cols);

    // add g to costMatrix_
    ierr = MatAXPY(costMatrix_, 1.0, stageCostMatrix_, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

    // find minimum for each row and compute residual norm
    Vec residual;
    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_, numStates_, &residual);
//    if(mode_ == mode::MINCOST) {
//        ierr = MatGetRowMin(costMatrix_, residual, policy); CHKERRQ(ierr);
//    }
//    else if(mode_ == mode::MAXREWARD) {
//        ierr = MatGetRowMax(costMatrix_, residual, policy); CHKERRQ(ierr);
//    }

    PetscInt nnz;
    const PetscInt *colIdx;
    const PetscReal *vals;

    if(mode_ == mode::MINCOST) {
        for(PetscInt rowInd = 0; rowInd < localNumStates_; ++rowInd) {
            MatGetRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals);
            const PetscReal *min = std::min_element(vals, vals + nnz);
            policy[rowInd] = colIdx[min - vals];
            VecSetValue(residual, rowInd + g_start_, *min, INSERT_VALUES);
            MatRestoreRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals);
        }
    }
    else if(mode_ == mode::MAXREWARD) {
        for(PetscInt rowInd = 0; rowInd < localNumStates_; ++rowInd) {
            MatGetRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals);
            const PetscReal *max = std::max_element(vals, vals + nnz);
            policy[rowInd] = colIdx[max - vals];
            VecSetValue(residual, rowInd + g_start_, *max, INSERT_VALUES);
            MatRestoreRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals);
        }
    }

    VecAssemblyBegin(residual);
    VecAssemblyEnd(residual);

    VecAXPY(residual, -1.0, V);
    VecNorm(residual, NORM_INFINITY, &residualNorm);
    VecDestroy(&residual);
//    VecDestroy(&costVector_);
//    MatDestroy(&costMatrix_);
    return 0;
}

// user must destroy P and g by himself. Function will create them. [used in inexactPolicyIteration]
PetscErrorCode MDP::constructFromPolicy(const PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    //LOG("Entering constructFromPolicy [policy]");
    PetscErrorCode ierr;
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
        ierr = MatGetValue(stageCostMatrix_, g_srcRow, actionInd, &g_pi_values[localStateInd]); CHKERRQ(ierr);
    }

    // generate index sets
    IS P_rowIndices;
    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, P_rowIndexValues, PETSC_COPY_VALUES, &P_rowIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    //LOG("Creating transitionProbabilities submatrix");
    ierr = MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, NULL, MAT_INITIAL_MATRIX, &transitionProbabilities); CHKERRQ(ierr);

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


PetscErrorCode MDP::iterativePolicyEvaluation(const Mat &jacobian, const Vec &stageCosts, Vec &V, KSPContext &ctx) {
    PetscErrorCode ierr;
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY); // overlap communication with KSP setup

    // KSP setup
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, jacobian, jacobian); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-40, ctx.threshold, PETSC_DEFAULT, ctx.maxIter); CHKERRQ(ierr); // use L2 norm of residual to compare. This works since ||x||_2 >= ||x||_inf. Much faster than infinity norm.
    //ierr = KSPSetConvergenceTest(ksp, &MDP::cvgTest, &ctx, NULL); CHKERRQ(ierr); // custom convergence test using infinity norm
    
    ierr = MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = KSPSolve(ksp, stageCosts, V); CHKERRQ(ierr);

    // output
    ierr = KSPGetIterationNumber(ksp, &ctx.kspIterations); CHKERRQ(ierr);
    KSPType type;
    KSPGetType(ksp, &type);
    jsonWriter_->add_data("KSPType", type);

    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    return ierr;
}

// defines the matrix-vector product for the jacobian shell
void MDP::jacobianMultiplication(Mat mat, Vec x, Vec y) {
    JacobianContext *ctx;
    MatShellGetContext(mat, (void **) &ctx);
    // (I - gamma * P_pi) * x == -gamma * P_pi * x + x
    MatMult(ctx->P_pi, x, y);
    VecScale(y, -ctx->discountFactor);
    VecAXPY(y, 1.0, x);
}

void MDP::jacobianMultiplicationTranspose(Mat mat, Vec x, Vec y) {
    JacobianContext *ctx;
    MatShellGetContext(mat, (void **) &ctx);
    // (I - gamma * P_pi)^T * x == -gamma * P_pi^T * x + x; since transposition distributes over subtraction
    MatMultTranspose(ctx->P_pi, x, y);
    VecScale(y, -ctx->discountFactor);
    VecAXPY(y, 1.0, x);
}


// creates MPIAIJ matrix and computes jacobian = I - gamma * P_pi
PetscErrorCode MDP::createJacobian(Mat &jacobian, const Mat &transitionProbabilities, JacobianContext &ctx) {
    PetscErrorCode ierr;
    ierr = MatCreateShell(PETSC_COMM_WORLD, localNumStates_, localNumStates_, numStates_, numStates_, &ctx, &jacobian); CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobian, MATOP_MULT, (void (*)(void)) jacobianMultiplication); CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobian, MATOP_MULT_TRANSPOSE, (void (*)(void)) jacobianMultiplicationTranspose); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode MDP::inexactPolicyIteration(const Vec &V0, IS &policy, Vec &optimalCost) {
    PetscErrorCode ierr;
    if(rank_ == 0) LOG("Entering inexactPolicyIteration");
    jsonWriter_->add_solver_run();

    // allocation for extractGreedyPolicy
    ierr = MatCreate(PETSC_COMM_WORLD, &costMatrix_); CHKERRQ(ierr);
    ierr = MatSetType(costMatrix_, MATDENSE); CHKERRQ(ierr);
    ierr = MatSetSizes(costMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_); CHKERRQ(ierr);
    ierr = MatSetUp(costMatrix_); CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD, localNumStates_*numActions_, numStates_*numActions_, &costVector_); CHKERRQ(ierr);

    Vec V;
    VecDuplicate(V0, &V);
    VecCopy(V0, V);

    Mat transitionProbabilities, jacobian;
    Vec stageCosts;
    PetscInt *policyValues;
    PetscMalloc1(localNumStates_, &policyValues);
    PetscReal residualNorm;
    PetscLogDouble startTime, endTime;

    PetscInt PI_iteration= 0;
    for(; PI_iteration < maxIter_PI_; ++PI_iteration) {
        PetscTime(&startTime);

        extractGreedyPolicy(V, policyValues, residualNorm);
        // debug
        if(rank_ == 0) LOG("DEBUG Iteration " + std::to_string(PI_iteration) + " Bellman residual norm: " + std::to_string(residualNorm));
        // check policy: every element must be in [0, numK_-1]
        for(PetscInt i = 0; i < localNumStates_; ++i) {
            if(policyValues[i] < 0 || policyValues[i] >= numActions_) {
                LOG("Error: policyValues[" + std::to_string(i) + "] = " + std::to_string(policyValues[i]) + " is not in [0, " + std::to_string(numActions_) + ")");
                return 1;
            }
        }

        if(residualNorm < atol_PI_) {
            PetscTime(&endTime);
            jsonWriter_->add_iteration_data(PI_iteration, 0, (endTime - startTime) * 1000, residualNorm);
            if(rank_ == 0) LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
            break;
        }
        constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
        JacobianContext ctxJac = {transitionProbabilities, discountFactor_};
        createJacobian(jacobian, transitionProbabilities, ctxJac);

        PetscReal vnorm;
        VecNorm(V, NORM_INFINITY, &vnorm);
        if(rank_ == 0) LOG("[before KSP] V norm: " + std::to_string(vnorm));
        // solve linear system
        KSPContext ctx = {maxIter_KSP_, residualNorm * rtol_KSP_, -1};
        iterativePolicyEvaluation(jacobian, stageCosts, V, ctx);
        // DEBUG: print V norm
        VecNorm(V, NORM_INFINITY, &vnorm);
        if(rank_ == 0) LOG("[after KSP] V norm: " + std::to_string(vnorm));


        MatDestroy(&transitionProbabilities);
        MatDestroy(&jacobian); // avoid memory leak
        VecDestroy(&stageCosts);

        PetscTime(&endTime);
        jsonWriter_->add_iteration_data(PI_iteration, ctx.kspIterations, (endTime - startTime) * 1000, residualNorm);
        //if(rank_ == 0) LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm)); // todo uncomment
    }

    if(PI_iteration >= maxIter_PI_) {
        LOG("Warning: maximum number of PI iterations reached. Solution might not be optimal.");
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

PetscErrorCode MDP::benchmarkIPI(const Vec &V0, IS &policy, Vec &optimalCost) {

    for(PetscInt i = 0; i < numPIRuns_; ++i) {
        inexactPolicyIteration(V0, policy, optimalCost);
    }

    return 0;
}
