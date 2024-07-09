//
// Created by robin on 27.04.23.
//

#include "MDP.h"
#include <algorithm> // std::min_element, std::max_element
#include <iostream>  // TODO: replace with some logger
// #include "../utils/Logger.h"

// find $\argmin_{\pi} \{ g^\pi + \gamma P^\pi V \}$
// PRE: policy is a array of size localNumStates_ and must be allocated. Function will write into it but not allocate it.
// idea: Mult gamma * P*V, reshape, add g, use MatGetRowMin
void MDP::extractGreedyPolicy(const Vec& V, PetscInt* policy, PetscReal& residualNorm)
{
    // costVector_ = discountFactor * P * V
    PetscCallThrow(MatMult(transitionProbabilityTensor_, V, costVector_));
    PetscCallThrow(VecScale(costVector_, discountFactor_)); // costVector_ = gamma * P*V

    // reshape costVector_ into costMatrix_ (fill matrix with values)
    const PetscReal* costVectorValues;
    PetscCallThrow(VecGetArrayRead(costVector_, &costVectorValues));
    IS rows, cols;
    PetscCallThrow(ISCreateStride(comm_, localNumStates_, g_start_, 1, &rows));
    PetscCallThrow(ISCreateStride(comm_, numActions_, 0, 1, &cols));
    const PetscInt *rowIndices, *colIndices;
    PetscCallThrow(ISGetIndices(rows, &rowIndices));
    PetscCallThrow(ISGetIndices(cols, &colIndices));
    // ierr = MatSetValuesIS(costMatrix_, rows, cols, costVectorValues, INSERT_VALUES); CHKERRQ(ierr); // for higher versions of PETSc
    PetscCallThrow(MatSetValues(costMatrix_, localNumStates_, rowIndices, numActions_, colIndices, costVectorValues, INSERT_VALUES));
    PetscCallThrow(MatAssemblyBegin(costMatrix_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(VecRestoreArrayRead(costVector_, &costVectorValues));
    PetscCallThrow(ISRestoreIndices(rows, &rowIndices));
    PetscCallThrow(ISRestoreIndices(cols, &colIndices));
    PetscCallThrow(MatAssemblyEnd(costMatrix_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(ISDestroy(&rows));
    PetscCallThrow(ISDestroy(&cols));

    // add g to costMatrix_
    PetscCallThrow(MatAXPY(costMatrix_, 1.0, stageCostMatrix_, SAME_NONZERO_PATTERN));

    // find minimum for each row and compute Bellman residual norm
    Vec residual;
    PetscCallThrow(VecCreateMPI(comm_, localNumStates_, numStates_, &residual));

    PetscInt         nnz;
    const PetscInt*  colIdx;
    const PetscReal* vals;

    if (mode_ == mode::MINCOST) {
        for (PetscInt rowInd = 0; rowInd < localNumStates_; ++rowInd) {
            PetscCallThrow(MatGetRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals));
            const PetscReal* min = std::min_element(vals, vals + nnz);
            policy[rowInd]       = colIdx[min - vals];
            PetscCallThrow(VecSetValue(residual, rowInd + g_start_, *min, INSERT_VALUES));
            PetscCallThrow(MatRestoreRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals));
        }
    } else if (mode_ == mode::MAXREWARD) {
        for (PetscInt rowInd = 0; rowInd < localNumStates_; ++rowInd) {
            PetscCallThrow(MatGetRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals));
            const PetscReal* max = std::max_element(vals, vals + nnz);
            policy[rowInd]       = colIdx[max - vals];
            PetscCallThrow(VecSetValue(residual, rowInd + g_start_, *max, INSERT_VALUES));
            PetscCallThrow(MatRestoreRow(costMatrix_, rowInd + g_start_, &nnz, &colIdx, &vals));
        }
    }

    PetscCallThrow(VecAssemblyBegin(residual));
    PetscCallThrow(VecAssemblyEnd(residual));

    PetscCallThrow(VecAXPY(residual, -1.0, V));
    PetscCallThrow(VecNorm(residual, NORM_INFINITY, &residualNorm));
    PetscCallThrow(VecDestroy(&residual));
}

// createse P_pi and g_pi from policy pi. P_pi and g_pi are allocated by this function but must be destroyed by the user.
void MDP::constructFromPolicy(const PetscInt* policy, Mat& transitionProbabilities, Vec& stageCosts)
{
    // LOG("Entering constructFromPolicy [policy]");
    //  compute where local ownership of new P_pi matrix starts
    PetscInt P_pi_start; // start of ownership of new matrix (to be created)
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &P_pi_start, NULL));
    P_pi_start /= numActions_;

    // allocate memory for values
    PetscInt* P_rowIndexValues;
    PetscCallThrow(PetscMalloc1(localNumStates_, &P_rowIndexValues));
    PetscReal* g_pi_values;
    PetscCallThrow(PetscMalloc1(localNumStates_, &g_pi_values));

    // compute global row indices for P and get values for g_pi
    PetscInt g_srcRow, actionInd;
    for (PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        actionInd = policy[localStateInd];
        // compute values for row index set
        P_rowIndexValues[localStateInd] = p_start_ + localStateInd * numActions_ + actionInd;
        // get values for stageCosts
        g_srcRow = g_start_ + localStateInd;
        PetscCallThrow(MatGetValue(stageCostMatrix_, g_srcRow, actionInd, &g_pi_values[localStateInd]));
    }

    // generate index sets
    IS P_rowIndices;
    PetscCallThrow(ISCreateGeneral(comm_, localNumStates_, P_rowIndexValues, PETSC_COPY_VALUES, &P_rowIndices));
    IS g_pi_rowIndices;
    PetscCallThrow(ISCreateStride(comm_, localNumStates_, g_start_, 1, &g_pi_rowIndices));

    // LOG("Creating transitionProbabilities submatrix");
    PetscCallThrow(MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, NULL, MAT_INITIAL_MATRIX, &transitionProbabilities));

    // LOG("Creating stageCosts vector");
    PetscCallThrow(VecCreateMPI(comm_, localNumStates_, numStates_, &stageCosts));
    const PetscInt* g_pi_rowIndexValues; // global indices
    PetscCallThrow(ISGetIndices(g_pi_rowIndices, &g_pi_rowIndexValues));
    PetscCallThrow(VecSetValues(stageCosts, localNumStates_, g_pi_rowIndexValues, g_pi_values, INSERT_VALUES));
    PetscCallThrow(ISRestoreIndices(g_pi_rowIndices, &g_pi_rowIndexValues));

    // LOG("Assembling transitionProbabilities and stageCosts");
    PetscCallThrow(MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(VecAssemblyBegin(stageCosts));
    PetscCallThrow(MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(VecAssemblyEnd(stageCosts));

    PetscCallThrow(ISDestroy(&P_rowIndices));
    PetscCallThrow(ISDestroy(&g_pi_rowIndices));
    PetscCallThrow(PetscFree(P_rowIndexValues));
    PetscCallThrow(PetscFree(g_pi_values));
}

void MDP::iterativePolicyEvaluation(const Mat& jacobian, const Vec& stageCosts, Vec& V, KSPContext& ctx)
{
    PetscCallThrow(MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY)); // overlap communication with KSP setup

    // KSP setup
    KSP ksp;
    PetscCallThrow(KSPCreate(comm_, &ksp));
    PetscCallThrow(KSPSetOperators(ksp, jacobian, jacobian));

    // PC setup
    PC pc;
    PetscCallThrow(KSPGetPC(ksp, &pc));
    PetscCallThrow(PCSetFromOptions(pc));

    PetscCallThrow(KSPSetFromOptions(ksp));
    PetscCallThrow(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCallThrow(KSPSetTolerances(ksp, 1e-40, ctx.threshold, PETSC_DEFAULT,
        ctx.maxIter)); // use L2 norm of residual to compare. This works since ||x||_2 >= ||x||_inf. Much faster than infinity norm.
    // ierr = KSPSetConvergenceTest(ksp, &MDP::cvgTest, &ctx, NULL); CHKERRQ(ierr); // custom convergence test using infinity norm

    PetscCallThrow(MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY));

    PetscCallThrow(KSPSolve(ksp, stageCosts, V));

    // output
    PetscCallThrow(KSPGetIterationNumber(ksp, &ctx.kspIterations));
    KSPType ksptype;
    PetscCallThrow(KSPGetType(ksp, &ksptype));
    jsonWriter_->add_data("KSPType",
        ksptype); // must be written here since KSPType is only available after KSPSolve. Other data should be written in MDP::writeJSONmetadata
    PCType pctype;
    PetscCallThrow(PCGetType(pc, &pctype));
    jsonWriter_->add_data("PCType", pctype);

    PetscCallThrow(KSPDestroy(&ksp));
}

// defines the matrix-vector product for the jacobian shell
void MDP::jacobianMultiplication(Mat mat, Vec x, Vec y)
{
    JacobianContext* ctx;
    PetscCallThrow(MatShellGetContext(mat, (void**)&ctx));
    // (I - gamma * P_pi) * x == -gamma * P_pi * x + x
    PetscCallThrow(MatMult(ctx->P_pi, x, y));
    PetscCallThrow(VecScale(y, -ctx->discountFactor));
    PetscCallThrow(VecAXPY(y, 1.0, x));
}

void MDP::jacobianMultiplicationTranspose(Mat mat, Vec x, Vec y)
{
    JacobianContext* ctx;
    PetscCallThrow(MatShellGetContext(mat, (void**)&ctx));
    // (I - gamma * P_pi)^T * x == -gamma * P_pi^T * x + x; since transposition distributes over subtraction
    PetscCallThrow(MatMultTranspose(ctx->P_pi, x, y));
    PetscCallThrow(VecScale(y, -ctx->discountFactor));
    PetscCallThrow(VecAXPY(y, 1.0, x));
}

// creates MPIAIJ matrix and computes jacobian = I - gamma * P_pi
void MDP::createJacobian(Mat& jacobian, const Mat& transitionProbabilities, JacobianContext& ctx)
{
    PetscCallThrow(MatCreateShell(comm_, localNumStates_, localNumStates_, numStates_, numStates_, &ctx, &jacobian));
    PetscCallThrow(MatShellSetOperation(jacobian, MATOP_MULT, (void (*)(void))jacobianMultiplication));
    PetscCallThrow(MatShellSetOperation(jacobian, MATOP_MULT_TRANSPOSE, (void (*)(void))jacobianMultiplicationTranspose));
}

void MDP::solve()
{
    // make sure MDP is set up
    setUp();

    // if(rank_ == 0) LOG("Entering solve");
    jsonWriter_->add_solver_run();
    writeJSONmetadata();

    // init guess V0
    Vec V0;
    PetscCallThrow(VecCreateMPI(comm_, localNumStates_, numStates_, &V0));
    PetscCallThrow(VecSet(V0, 1.0));

    // allocation for extractGreedyPolicy
    PetscCallThrow(MatCreate(comm_, &costMatrix_));
    PetscCallThrow(MatSetType(costMatrix_, MATDENSE));
    PetscCallThrow(MatSetSizes(costMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetUp(costMatrix_));
    PetscCallThrow(VecCreateMPI(comm_, localNumStates_ * numActions_, numStates_ * numActions_, &costVector_));

    Vec V;
    PetscCallThrow(VecDuplicate(V0, &V));
    PetscCallThrow(VecCopy(V0, V));

    Mat       transitionProbabilities, jacobian;
    Vec       stageCosts;
    PetscInt* policyValues;
    PetscCallThrow(PetscMalloc1(localNumStates_, &policyValues));
    PetscReal      residualNorm;
    PetscLogDouble startTime, endTime, startiPI, endiPI;

    PetscCallThrow(PetscTime(&startiPI));
    PetscInt PI_iteration = 0;
    for (; PI_iteration < maxIter_PI_; ++PI_iteration) { // outer loop
        PetscCallThrow(PetscTime(&startTime));

        extractGreedyPolicy(V, policyValues, residualNorm);

        if (residualNorm < atol_PI_) {
            PetscCallThrow(PetscTime(&endTime));
            jsonWriter_->add_iteration_data(PI_iteration, 0, (endTime - startTime) * 1000, residualNorm);
            // if(rank_ == 0) LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
            break;
        }
        constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
        JacobianContext ctxJac = { transitionProbabilities, discountFactor_ };
        createJacobian(jacobian, transitionProbabilities, ctxJac);

        // solve linear system
        KSPContext ctx = { maxIter_KSP_, residualNorm * alpha_, -1 };
        iterativePolicyEvaluation(jacobian, stageCosts, V, ctx); // inner loop

        PetscCallThrow(MatDestroy(&transitionProbabilities));
        PetscCallThrow(MatDestroy(&jacobian)); // avoid memory leak
        PetscCallThrow(VecDestroy(&stageCosts));

        PetscCallThrow(PetscTime(&endTime));
        jsonWriter_->add_iteration_data(PI_iteration, ctx.kspIterations, (endTime - startTime) * 1000, residualNorm);
        // if(rank_ == 0) LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
    }
    PetscCallThrow(PetscTime(&endiPI));
    // LOG("Inexact Policy Iteration took: " + std::to_string((endiPI - startiPI) * 1000) + " ms");

    if (PI_iteration >= maxIter_PI_) {
        // LOG("Warning: maximum number of PI iterations reached. Solution might not be optimal.");
        std::cout << "Warning: maximum number of PI iterations reached. Solution might not be optimal." << std::endl;
    }

    jsonWriter_->write_to_file(file_stats_);

    PetscCallThrow(MatDestroy(&transitionProbabilities));
    PetscCallThrow(MatDestroy(&jacobian));
    PetscCallThrow(VecDestroy(&stageCosts));
    PetscCallThrow(VecDestroy(&V0));

    // output results
    PetscCallThrow(PetscTime(&startTime));
    IS optimalPolicy;
    PetscCallThrow(ISCreateGeneral(comm_, localNumStates_, policyValues, PETSC_COPY_VALUES, &optimalPolicy));

    writeVec(V, file_cost_);
    writeIS(optimalPolicy, file_policy_);

    PetscCallThrow(ISDestroy(&optimalPolicy));
    PetscCallThrow(VecDestroy(&V));
    PetscCallThrow(PetscFree(policyValues));

    PetscCallThrow(PetscTime(&endTime));
    PetscLogDouble duration = (endTime - startTime) * 1000;
    if (rank_ == 0) {
        // LOG("Saving results took: " + std::to_string(duration) + " ms");
    }
}

void MDP::cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason* reason, void* ctx)
{
    PetscReal threshold = static_cast<KSPContext*>(ctx)->threshold;
    PetscReal norm;

    Vec res;
    // ierr = VecDuplicate(ksp->vec_rhs, &res); CHKERRQ(ierr);
    PetscCallThrow(KSPBuildResidual(ksp, NULL, NULL, &res));
    PetscCallThrow(VecNorm(res, NORM_INFINITY, &norm));
    PetscCallThrow(VecDestroy(&res));

    // PetscPrintf(comm_, "it = %d: residual norm = %f\n", it, norm);

    if (it == 0)
        *reason = KSP_CONVERGED_ITERATING;
    else if (norm < threshold)
        *reason = KSP_CONVERGED_RTOL;
    else if (it >= static_cast<KSPContext*>(ctx)->maxIter)
        *reason = KSP_DIVERGED_ITS;
    else
        *reason = KSP_CONVERGED_ITERATING;
}

/*
void MDP::benchmarkIPI(const Vec &V0, IS &policy, Vec &optimalCost) {

    for(PetscInt i = 0; i < numPIRuns_; ++i) {
        solve(V0, policy, optimalCost);
    }
}
*/
