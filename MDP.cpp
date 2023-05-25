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
    Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all rank should output to the same file
    LOG("owns " + std::to_string(localNumStates_) + " states.");
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
    VecDestroy(&nnz_);
}

// find $\argmin_{\pi} \{ g^\pi + \gamma P^\pi V \}$
PetscErrorCode MDP::extractGreedyPolicy(Vec &V, PetscInt *policy, GreedyPolicyType type) {

    if(type == GreedyPolicyType::V1) {

        PetscErrorCode ierr;
        PetscInt *indices; // stores indices [actionInd*numStates, (actionInd+1)*numStates-1] for extracting values from P
        PetscMalloc1(numStates_, &indices);
        PetscInt *vecIndices; // contains indices [0, 1, 2, ..., numStates_-1] for setting values in tmp
        PetscMalloc1(numStates_, &vecIndices);
        std::iota(vecIndices, vecIndices + numStates_, 0); // initialize vecIndices to 0, 1, 2, ..., numStates_-1
        PetscReal *tmpValues; // contains values to be stored in tmp
        PetscMalloc1(numStates_, &tmpValues);

        Vec tmp; // contains block from P
        ierr = VecCreate(PETSC_COMM_WORLD, &tmp); CHKERRQ(ierr);
        ierr = VecSetType(tmp, VECSEQ); CHKERRQ(ierr); // todo: to be changed for parallel version, also maybe bad for performance since vector is dense
        ierr = VecSetSizes(tmp, PETSC_DECIDE, numStates_); CHKERRQ(ierr);
        ierr = VecSetUp(tmp); CHKERRQ(ierr);

        PetscReal minCost = std::numeric_limits<PetscReal>::max();

        for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) {
            for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
                // extract transition probabilities
                std::iota(indices, indices + numStates_, actionInd*numStates_); // initialize indices to [actionInd*numStates, (actionInd+1)*numStates-1]
                ierr = MatGetValues(transitionProbabilityTensor_, 1, &stateInd, numStates_, indices, tmpValues); CHKERRQ(ierr);
                ierr = VecSetValues(tmp, numStates_, vecIndices, tmpValues, INSERT_VALUES); CHKERRQ(ierr);
                // extract stage costs
                PetscReal stageCost;
                ierr = MatGetValue(stageCostMatrix_, stateInd, actionInd, &stageCost);
                CHKERRQ(ierr);
                // compute g + gamma PV
                PetscReal cost = 0;
                ierr = VecDot(tmp, V, &cost); CHKERRQ(ierr);
                cost = stageCost + discountFactor_ * cost;
                if(cost < minCost) {
                    minCost = cost;
                    policy[stateInd] = actionInd;
                }
            }
            minCost = std::numeric_limits<PetscReal>::max();
        }

        ierr = PetscFree(indices); CHKERRQ(ierr);
        ierr = PetscFree(vecIndices); CHKERRQ(ierr);
        ierr = PetscFree(tmpValues); CHKERRQ(ierr);
        ierr = VecDestroy(&tmp); CHKERRQ(ierr);
    }

    if (type == GreedyPolicyType::V2) {
        PetscErrorCode ierr;

        PetscInt *tmppolicy; // stores temporary policy (filled with actionInd)
        PetscMalloc1(localNumStates_, &tmppolicy);
        const PetscReal *costValues; // stores cost (= g + gamma PV) values for each state
        PetscReal *minCostValues; // stores minimum cost values for each state
        PetscMalloc1(localNumStates_, &minCostValues);
        std::fill(minCostValues, minCostValues + localNumStates_,
                  std::numeric_limits<PetscReal>::max());


        Vec g;
        VecCreate(PETSC_COMM_WORLD, &g);
        VecSetType(g, VECMPI);
        VecSetSizes(g, localNumStates_, PETSC_DECIDE);
        VecSetUp(g);
        Vec cost;
        VecCreate(PETSC_COMM_WORLD, &cost);
        VecSetType(cost, VECMPI);
        VecSetSizes(cost, localNumStates_, PETSC_DECIDE);
        VecSetUp(cost);

        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            Mat P;
            //MatCreate(PETSC_COMM_WORLD, &P);
            //MatSetType(P, MATSEQAIJ);
            //MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, numStates_, numStates_);
            //MatSetUp(P);

            //std::fill(tmppolicy, tmppolicy + numStates_, actionInd);
            constructFromPolicy(tmppolicy, P, g); // creates P => need to destroy P ourselves
            ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
            CHKERRQ(ierr);
            ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
            CHKERRQ(ierr);
            ierr = MatScale(P, discountFactor_);
            CHKERRQ(ierr);
            ierr = MatMultAdd(P, V, g, g);
            CHKERRQ(ierr);
            ierr = VecGetArrayRead(g, &costValues);
            CHKERRQ(ierr);

            for (int stateInd = 0; stateInd < numStates_; ++stateInd) {
                if (costValues[stateInd] < minCostValues[stateInd]) {
                    minCostValues[stateInd] = costValues[stateInd];
                    policy[stateInd] = actionInd;
                }
            }
            ierr = VecRestoreArrayRead(g, &costValues);
            CHKERRQ(ierr);
            ierr = MatDestroy(&P); CHKERRQ(ierr);
        }

        ierr = PetscFree(tmppolicy);
        CHKERRQ(ierr);
        ierr = PetscFree(costValues);
        CHKERRQ(ierr);
        //ierr = MatDestroy(&P);
        CHKERRQ(ierr);
        ierr = VecDestroy(&g);
        CHKERRQ(ierr);
        ierr = VecDestroy(&cost);
        CHKERRQ(ierr);
    }

    /*if(type == V3) {
        // multiply tall tensor with V, find n times minimum of m values
        PetscErrorCode ierr;
        Vec cost;
        VecCreate(PETSC_COMM_WORLD, &cost);
        VecSetType(cost, VECSEQ);
        VecSetSizes(cost, PETSC_DECIDE, numStates_*numActions_);

        MatMult
    }*/



    return 0;
}

// fills a provided and previously allocated array with the number of nonzeros per row of a matrix
PetscErrorCode getNNZPerRow(const Mat M, PetscInt* nnzPerRow, PetscInt rows) {
    for(PetscInt row = 0; row < rows; ++row) {
        PetscInt nnz;
        MatGetRow(M, row, &nnz, NULL, NULL);
        nnzPerRow[row] = nnz;
        MatRestoreRow(M, row, &nnz, NULL, NULL);
    }
    /* alternative approach according to ChatGPT, maybe faster, but don't know if it even works
    for (i = 0; i < rows; i++) {
        nnz = a->i[i+1] - a->i[i];  // The i array stores the row offsets in the matrix data
        nnzPerRow[i] = nnz;
    }
    */
    return 0;
}

#if 0
PetscErrorCode MDP::constructFromPolicy(const PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts) {
    // same as below, but same action for all states (used in greedy PolicyType V2)

    /* Procedure
     * Prealloc: on each rank obtain local off and on diagonal matrix (MatMPIAIJGetSeqAIJ)
     * get nnz for diag and offdiag using MatGetInfo
     * set prealloc using MatMPIAIJSetPreallocation
     * get values from MatGetRow and set values using MatSetValues, then restore row
     */
    LOG("Entering constructFromPolicy");
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilities);
    MatSetType(transitionProbabilities, MATMPIAIJ);
    MatSetSizes(transitionProbabilities, localNumStates_, PETSC_DECIDE, PETSC_DECIDE, numStates_);

    LOG("Preallocating transitionProbabilities matrix");
    // Preallocation
    Mat Diag, Offdiag;
    MatMPIAIJGetSeqAIJ(transitionProbabilityTensor_, &Diag, &Offdiag, NULL);
    MatInfo DiagInfo, OffdiagInfo;
    MatGetInfo(Diag, MAT_LOCAL, &DiagInfo);
    MatGetInfo(Offdiag, MAT_LOCAL, &OffdiagInfo);
    PetscInt *diagNNZPerRow, *offdiagNNZPerRow;
    PetscMalloc1(localNumStates_, &diagNNZPerRow);
    PetscMalloc1(localNumStates_, &offdiagNNZPerRow);
    getNNZPerRow(Diag, diagNNZPerRow, localNumStates_);
    getNNZPerRow(Offdiag, offdiagNNZPerRow, localNumStates_);
    MatMPIAIJSetPreallocation(transitionProbabilities, 0, diagNNZPerRow, 0, offdiagNNZPerRow); // 0 is ignored
    LOG("Finished preallocating transitionProbabilities matrix");
    MatSetOption(transitionProbabilities, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); // TODO remove

    PetscInt P_pi_start;
    MatGetOwnershipRange(transitionProbabilities, &P_pi_start, NULL);

    LOG("Creating stageCosts vector");

    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_, numStates_, &stageCosts);
    PetscReal *stageCostValues;
    PetscMalloc1(localNumStates_, &stageCostValues);
    PetscInt *stageCostIndices; // global indices
    PetscMalloc1(localNumStates_, &stageCostIndices);
    std::iota(stageCostIndices, stageCostIndices + localNumStates_, g_start_); // fill with global indices with [g_start_, g_start_ + localNumStates_)
    LOG("Finished creating stageCosts vector");

    // set values (row-wise)
    PetscInt P_srcRow, P_pi_destRow, g_srcRow;
    for(PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        LOG("Currently at local state " + std::to_string(localStateInd) + "/" + std::to_string(localNumStates_));
        // compute global indices
        P_srcRow  = P_start_ + localStateInd * numActions_ + actionInd;
        //P_destRow = P_start_ + localStateInd;
        P_pi_destRow = P_pi_start + localStateInd;
        g_srcRow  = g_start_ + localStateInd;

        // DEBUG
        PetscReal nnzPy; // data from python
        VecGetValues(nnz_, 1, &P_srcRow, &nnzPy);

        PetscInt nnz; // data from PETSc MatGetRow()
        const PetscInt *cols;
        const PetscReal *vals;

        MatGetRow(transitionProbabilityTensor_, P_srcRow, &nnz, &cols, &vals);

        LOG("nnzPy: " + std::to_string(static_cast<PetscInt>(nnzPy)) + ", nnz: " + std::to_string(nnz) + ", diagNNZ: " + std::to_string(diagNNZPerRow[localStateInd]) + ", offdiagNNZ: " + std::to_string(offdiagNNZPerRow[localStateInd]));
        PetscInt maxNNZ = std::max({nnz, diagNNZPerRow[localStateInd] + offdiagNNZPerRow[localStateInd], static_cast<PetscInt>(nnzPy)});

        MatSetValues(transitionProbabilities, 1, &P_pi_destRow, maxNNZ, cols, vals, INSERT_VALUES);
        MatRestoreRow(transitionProbabilityTensor_, P_srcRow, &nnz, &cols, &vals);

        // get stage cost
        MatGetValue(stageCostMatrix_, g_srcRow, actionInd, &stageCostValues[localStateInd]);
    }
    LOG("Finished setting matrix values");
    VecSetValues(stageCosts, localNumStates_, stageCostIndices, stageCostValues, INSERT_VALUES);


    LOG("Assembling transitionProbabilities and stageCosts");
    MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(stageCosts);
    MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(stageCosts);



    PetscFree(diagNNZPerRow);
    PetscFree(offdiagNNZPerRow);
    PetscFree(stageCostValues);
    PetscFree(stageCostIndices);
    return 0;
}
#endif

PetscErrorCode MDP::constructFromPolicy(const PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts) {
    LOG("Entering constructFromPolicy");

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
    IS P_pi_colIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, P_pi_start, 1, &P_pi_colIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    LOG("Creating transitionProbabilities submatrix");
    MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, P_pi_colIndices, MAT_INITIAL_MATRIX, &transitionProbabilities);

    LOG("Creating stageCosts vector");
    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_, numStates_, &stageCosts);
    const PetscInt *g_pi_rowIndexValues; // global indices
    ISGetIndices(g_pi_rowIndices, &g_pi_rowIndexValues);
    VecSetValues(stageCosts, localNumStates_, g_pi_rowIndexValues, g_pi_values, INSERT_VALUES);
    ISRestoreIndices(g_pi_rowIndices, &g_pi_rowIndexValues);

    LOG("Assembling transitionProbabilities and stageCosts");
    MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(stageCosts);
    MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(stageCosts);

    // output dimensions (DEBUG)
    PetscInt m, n;
    MatGetSize(transitionProbabilities, &m, &n);
    LOG("transitionProbabilities: " + std::to_string(m) + "x" + std::to_string(n));

    ISDestroy(&P_rowIndices);
    ISDestroy(&P_pi_colIndices);
    ISDestroy(&g_pi_rowIndices);
    PetscFree(P_rowIndexValues);
    PetscFree(g_pi_values);
    return 0;
}


PetscErrorCode MDP::constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    // extract row i*numStates_+policy[i] from transitionProbabilityTensor_ for every i (row)
    // extract stageCostMatrix_[i, policy[i]] for every i (row)

    PetscErrorCode ierr;
    PetscScalar *nnz_scalar;
    PetscMalloc1(numStates_, &nnz_scalar);
    PetscInt *nnz;
    PetscMalloc1(numStates_, &nnz);
    auto *it_scalar = nnz_scalar;
    auto *it = nnz;
    PetscInt *nnzidx;
    PetscMalloc1(numStates_, &nnzidx);
    PetscReal *stageCostVal;
    PetscMalloc1(numStates_, &stageCostVal);

    for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) {
        nnzidx[stateInd] = stateInd*numActions_ + policy[stateInd];
    }
    ierr = VecGetValues(nnz_, numStates_, nnzidx, nnz_scalar); CHKERRQ(ierr); // IMPORTANT: can only get values from part of vector that is owned by this process
    for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) { // convert to int
        *it++ = static_cast<PetscInt>(*it_scalar++);
    }

    // preallocate transitionProbabilities
    ierr = MatCreate(PETSC_COMM_WORLD, &transitionProbabilities); CHKERRQ(ierr);
    ierr = MatSetType(transitionProbabilities, MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(transitionProbabilities, PETSC_DECIDE, PETSC_DECIDE, numStates_, numStates_); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(transitionProbabilities, 0, nnz); CHKERRQ(ierr);

    // extract row and set in new matrix
    for(int stateInd = 0; stateInd < numStates_; ++stateInd) {
        const PetscInt *cols;
        const PetscReal *values;
        PetscInt row = stateInd * numActions_ + policy[stateInd];
        MatGetRow(transitionProbabilityTensor_, row, NULL, &cols, &values);
        MatSetValues(transitionProbabilities, 1, &stateInd, nnz[stateInd], cols, values, INSERT_VALUES);
        MatRestoreRow(transitionProbabilityTensor_, row, NULL, &cols, &values);
        // stage cost
        MatGetValue(stageCostMatrix_, stateInd, policy[stateInd], &stageCostVal[stateInd]);
    }
    PetscInt *indices;
    PetscMalloc1(numStates_, &indices);
    std::iota(indices, indices + numStates_, 0);
    ierr = VecSetValues(stageCosts, numStates_, indices, stageCostVal, INSERT_VALUES); CHKERRQ(ierr);


    // todo free everything
    PetscFree(nnz);
    PetscFree(nnz_scalar);
    PetscFree(nnzidx);
    PetscFree(stageCostVal);
    PetscFree(indices);
    return 0;
}

PetscErrorCode MDP::iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, PetscReal threshold) {
    PetscErrorCode ierr;
    const PetscReal rtol = 1e-15;

    // ksp solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, jacobian, jacobian); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    //ierr = KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr); // todo: custom stopping criterion
    ierr = KSPSetConvergenceTest(ksp, &cvgTest, &threshold, NULL); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, stageCosts, V); CHKERRQ(ierr);

    return ierr;
}

std::vector<PetscInt> MDP::inexactPolicyIteration(Vec &V0, const PetscInt maxIter, PetscReal alpha) {
// #define VI
    std::vector<PetscInt> policy(numStates_); // result

    // todo: change jacobian to ShellMat

    Vec V, V_old;
    VecDuplicate(V0, &V);
    VecCopy(V0, V);
    VecDuplicate(V, &V_old);

    Vec stageCosts;
    VecCreate(PETSC_COMM_WORLD, &stageCosts);
    VecSetType(stageCosts, VECSEQ);
    VecSetSizes(stageCosts, PETSC_DECIDE, numStates_);

    for(PetscInt i = 0; i < maxIter; ++i) {
        PetscPrintf(PETSC_COMM_WORLD, "Iteration %d\n", i);

        extractGreedyPolicy(V, policy.data(), GreedyPolicyType::V2);
#ifdef VI
        constructFromPolicy(policy.data(), jacobian, stageCosts);
        MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
        MatScale(jacobian, discountFactor_); // needs assembled matrix
        MatMultAdd(jacobian, V, stageCosts, stageCosts);
        VecCopy(stageCosts, V);
#endif
#ifndef VI
        Mat jacobian;
        //MatCreate(PETSC_COMM_WORLD, &jacobian);
        //MatSetSizes(jacobian, PETSC_DECIDE, PETSC_DECIDE, numStates_, numStates_);
        //MatSetType(jacobian, MATSEQAIJ);
        //MatSetUp(jacobian);
        // construct Jacobian
        constructFromPolicy(policy.data(), jacobian, stageCosts); // creates jacobian => needs to be destroyed
        MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
        MatScale(jacobian, -discountFactor_); // needs assembled matrix
        MatShift(jacobian, 1);

        VecCopy(V, V_old);

        // compute residual norm used for stopping criterion
        PetscReal r0_norm;
        computeResidualNorm(jacobian, V, stageCosts, &r0_norm);

        iterativePolicyEvaluation(jacobian, stageCosts, V, r0_norm*alpha);
        MatDestroy(&jacobian);
#endif
    }
    // todo: stopping condition for loop as discussed in meeting
    // todo: save time and optimality gap for each iteration
    // print V
    /*
    PetscPrintf(PETSC_COMM_WORLD, "V = \n");
    VecView(V, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "\n\n");
    MatDestroy(&jacobian);
     */
    VecDestroy(&V);
    VecDestroy(&V_old);
    VecDestroy(&stageCosts);
    return policy;
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

    /*
    // cols
    PetscInt start_P_col, end_P_col, start_g_col, end_g_col;
    MatGetOwnershipRangeColumn(transitionProbabilityTensor_, &start_P_col, &end_P_col);
    MatGetOwnershipRangeColumn(stageCostMatrix_, &start_g_col, &end_g_col);
    std::cout << "Rank: " << rank_ << " owns cols " << start_P_col << " to " << end_P_col << " of P." << std::endl;
    std::cout << "Rank: " << rank_ << " owns cols " << start_g_col << " to " << end_g_col << " of g." << std::endl;
    */

    return ierr;
}
