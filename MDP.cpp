//
// Created by robin on 27.04.23.
//

#include "MDP.h"
#include <numeric>
#include <petscksp.h>
#include <mpi.h>
#include <iostream>
#include "utils/Logger.h"
#include <cassert>
#include <algorithm>

// declare function for convergence test
PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx);


MDP::MDP(const PetscInt numStates, const PetscInt numActions, const PetscReal discountFactor) : numStates_(numStates),
                                                                                                numActions_(numActions),
                                                                                                discountFactor_(discountFactor) {
    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_ = nullptr;
    nnz_ = nullptr;

    // MPI parallelization initialization
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);
    MPI_Comm_size(PETSC_COMM_WORLD, &size_);
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
    LOG("owns " + std::to_string(localNumStates_) + " states.");
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
    VecDestroy(&nnz_);
}

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

// user must destroy P and g by himself. Function will create them.
PetscErrorCode MDP::constructFromPolicy(const PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts) {
    LOG("Entering constructFromPolicy [actionInd]");

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
    IS P_pi_colIndices; // TODO: remove
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, P_pi_start, 1, &P_pi_colIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    //LOG("Creating transitionProbabilities submatrix");
    MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, P_pi_colIndices, MAT_INITIAL_MATRIX, &transitionProbabilities); // todo set colIndices to NULL, then all columns are selected

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

    // output dimensions (DEBUG)
    PetscInt m, n;
    MatGetSize(transitionProbabilities, &m, &n);
    //LOG("transitionProbabilities: " + std::to_string(m) + "x" + std::to_string(n));

    ISDestroy(&P_rowIndices);
    ISDestroy(&P_pi_colIndices);
    ISDestroy(&g_pi_rowIndices);
    PetscFree(P_rowIndexValues);
    PetscFree(g_pi_values);
    return 0;
}

// user must destroy P and g by himself. Function will create them.
PetscErrorCode MDP::constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    LOG("Entering constructFromPolicy [policy]");

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
    IS P_pi_colIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, P_pi_start, 1, &P_pi_colIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    //LOG("Creating transitionProbabilities submatrix");
    MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, P_pi_colIndices, MAT_INITIAL_MATRIX, &transitionProbabilities); // todo set colIndices to NULL, then all columns are selected

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

    // output dimensions (DEBUG)
    PetscInt m, n;
    MatGetSize(transitionProbabilities, &m, &n);
    //LOG("transitionProbabilities: " + std::to_string(m) + "x" + std::to_string(n));

    ISDestroy(&P_rowIndices);
    ISDestroy(&P_pi_colIndices);
    ISDestroy(&g_pi_rowIndices);
    PetscFree(P_rowIndexValues);
    PetscFree(g_pi_values);
    return 0;
}


PetscErrorCode MDP::iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, PetscReal threshold) {
    PetscErrorCode ierr;
    //const PetscReal rtol = 1e-15;

    // ksp solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr); // todo: destroy ksp
    ierr = KSPSetOperators(ksp, jacobian, jacobian); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    //ierr = KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetConvergenceTest(ksp, &cvgTest, &threshold, NULL); CHKERRQ(ierr); // todo add max_it as parameter
    ierr = KSPSolve(ksp, stageCosts, V); CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode MDP::createJacobian(Mat &jacobian, const Mat &transitionProbabilities) {
    PetscErrorCode ierr;
    MatCreate(PETSC_COMM_WORLD, &jacobian);
    MatSetType(jacobian, MATMPIAIJ);
    MatSetSizes(jacobian, localNumStates_, PETSC_DECIDE, PETSC_DETERMINE, numStates_);
    MatMPIAIJSetPreallocation(jacobian, 1, NULL, 0, NULL);
    //IS indices;
    //PetscInt rowStart = P_start_ / numActions_;
    //ISCreateStride(PETSC_COMM_WORLD, localNumStates_, rowStart, 1, &indices);
    ierr = MatZeroEntries(jacobian);
    CHKERRQ(ierr);
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
    ierr = MatShift(jacobian, 1.0);
    CHKERRQ(ierr);
    ierr = MatAXPY(jacobian, -discountFactor_, transitionProbabilities, DIFFERENT_NONZERO_PATTERN);
    CHKERRQ(ierr);
    return 0;
}


PetscErrorCode MDP::inexactPolicyIteration(Vec &V0, const PetscInt maxIter, PetscReal alpha, IS &policy, Vec &optimalCost) {
    LOG("Entering inexactPolicyIteration");
    // todo: change jacobian to ShellMat
    // todo: parallelize
    PetscReal atol = 1e-10; // for outer loop todo: as parameter
    PetscErrorCode ierr;

    Vec V;
    VecDuplicate(V0, &V);
    VecCopy(V0, V);

    Mat transitionProbabilities, jacobian;
    Vec stageCosts;
    PetscInt *policyValues;
    PetscMalloc1(localNumStates_, &policyValues);

    PetscReal r0_norm;

    // initialize policy iteration
    extractGreedyPolicy(V, policyValues);
    constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
/*
    ierr = MatCreateConstantDiagonal(PETSC_COMM_WORLD, localNumStates_, PETSC_DECIDE, PETSC_DETERMINE, numStates_, 1, &jacobian);
    CHKERRQ(ierr);
    ierr = MatAXPY(jacobian, -discountFactor_, transitionProbabilities, DIFFERENT_NONZERO_PATTERN);
    CHKERRQ(ierr);
*/
    createJacobian(jacobian, transitionProbabilities);
    // compute residual norm used for KSP stopping criterion
    computeResidualNorm(jacobian, V, stageCosts, &r0_norm);

    for(PetscInt i = 1; i <= maxIter; ++i) {
        //PetscPrintf(PETSC_COMM_WORLD, "Iteration %d\n", i);
        LOG("Iteration " + std::to_string(i) + " residual norm: " + std::to_string(r0_norm));

        // solve linear system
        iterativePolicyEvaluation(jacobian, stageCosts, V, r0_norm*alpha);
        MatDestroy(&transitionProbabilities);
        MatDestroy(&jacobian); // avoid memory leak
        VecDestroy(&stageCosts);

        // compute jacobian wrt new policy
        extractGreedyPolicy(V, policyValues);
        constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
        /*
        ierr = MatCreateConstantDiagonal(PETSC_COMM_WORLD, localNumStates_, PETSC_DECIDE, PETSC_DETERMINE, numStates_, 1, &jacobian);
        CHKERRQ(ierr);
        ierr = MatAXPY(jacobian, -discountFactor_, transitionProbabilities, DIFFERENT_NONZERO_PATTERN);
        CHKERRQ(ierr);
         */
        createJacobian(jacobian, transitionProbabilities);
        computeResidualNorm(jacobian, V, stageCosts, &r0_norm); // used for outer loop stopping criterion + RHS of KSP stopping criterion

        if(r0_norm < atol) {
            break;
        }
    }

    MatDestroy(&transitionProbabilities);
    MatDestroy(&jacobian);
    VecDestroy(&stageCosts);

    VecDuplicate(V, &optimalCost);
    VecCopy(V, optimalCost);

    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, policyValues, PETSC_COPY_VALUES, &policy);
    PetscFree(policyValues);

    VecDestroy(&V);

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


PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx) {
    PetscErrorCode ierr;
    PetscReal threshold = *static_cast<PetscReal*>(ctx);
    PetscReal norm;

    Vec res;
    //ierr = VecDuplicate(ksp->vec_rhs, &res); CHKERRQ(ierr);
    ierr = KSPBuildResidual(ksp, NULL, NULL, &res); CHKERRQ(ierr);
    ierr = VecNorm(res, NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = VecDestroy(&res); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "it = %d: residual norm = %f\n", it, norm);

    if(it == 0) *reason = KSP_CONVERGED_ITERATING;
    else if(norm < threshold) *reason = KSP_CONVERGED_RTOL;
    //else if(it >= ksp->max_it) *reason = KSP_DIVERGED_ITS;
    else *reason = KSP_CONVERGED_ITERATING;

    return 0;
}

PetscErrorCode MDP::loadFromBinaryFile(std::string filename_P, std::string filename_g, std::string filename_nnz) {
    PetscErrorCode ierr = 0;
    PetscViewer viewer;

    // load transition probability tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetFromOptions(transitionProbabilityTensor_);
    ierr = MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, PETSC_DECIDE, PETSC_DECIDE, numStates_); CHKERRQ(ierr);
    MatSetUp(transitionProbabilityTensor_);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_P.c_str());
    MatLoad(transitionProbabilityTensor_, viewer);
    PetscViewerDestroy(&viewer);

    // load stage cost matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetFromOptions(stageCostMatrix_);
    ierr = MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, PETSC_DECIDE, numActions_); CHKERRQ(ierr);
    MatSetUp(stageCostMatrix_);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_g.c_str());
    MatLoad(stageCostMatrix_, viewer);
    PetscViewerDestroy(&viewer);

    // load nnz
    VecCreate(PETSC_COMM_WORLD, &nnz_);
    VecSetFromOptions(nnz_);
    ierr = VecSetSizes(nnz_, localNumStates_*numActions_, PETSC_DECIDE); CHKERRQ(ierr);
    VecSetUp(nnz_);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_nnz.c_str());
    VecLoad(nnz_, viewer);
    PetscViewerDestroy(&viewer);

    // Information about distribution on processes
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
    LOG("owns rows " + std::to_string(P_start_) + " to " + std::to_string(P_end_) + " of P.");
    LOG("owns rows " + std::to_string(g_start_) + " to " + std::to_string(g_end_) + " of g.");

    return ierr;
}
