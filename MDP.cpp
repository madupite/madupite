//
// Created by robin on 27.04.23.
//

#include "MDP.h"
#include <numeric>
#include <petscksp.h>
#include <mpi.h>
#include <iostream>

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
    std::cout << "[ctor]: Rank " << rank_ << " has " << localNumStates_ << " states." << std::endl;
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
    VecDestroy(&nnz_);
}
#if 0
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
        PetscMalloc1(numStates_, &tmppolicy);
        const PetscReal *costValues; // stores cost (= g + gamma PV) values for each state
        PetscReal *minCostValues; // stores minimum cost values for each state
        PetscMalloc1(numStates_, &minCostValues);
        std::fill(minCostValues, minCostValues + numStates_,
                  std::numeric_limits<PetscReal>::max());


        Vec g;
        VecCreate(PETSC_COMM_WORLD, &g);
        VecSetType(g, VECSEQ);
        VecSetSizes(g, PETSC_DECIDE, numStates_);
        VecSetUp(g);
        Vec cost;
        VecCreate(PETSC_COMM_WORLD, &cost);
        VecSetType(cost, VECSEQ);
        VecSetSizes(cost, PETSC_DECIDE, numStates_);
        VecSetUp(cost);

        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            Mat P;
            //MatCreate(PETSC_COMM_WORLD, &P);
            //MatSetType(P, MATSEQAIJ);
            //MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, numStates_, numStates_);
            //MatSetUp(P);

            std::fill(tmppolicy, tmppolicy + numStates_, actionInd);
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
    // todo: stopping condition for loop
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
#endif
PetscErrorCode MDP::loadFromBinaryFile(std::string filename_P, std::string filename_g, std::string filename_nnz) {
    PetscErrorCode ierr = 0;
    PetscViewer viewer;

    // load transition probability tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetFromOptions(transitionProbabilityTensor_);
    ierr = MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, PETSC_DECIDE, PETSC_DECIDE, numStates_);
    CHKERRQ(ierr);
    MatSetUp(transitionProbabilityTensor_);
    std::cout << "P: " << localNumStates_*numActions_ << " " << numStates_ << " " << numStates_*numActions_ << " " << numStates_ << std::endl;
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_P.c_str());
    MatLoad(transitionProbabilityTensor_, viewer);
    PetscViewerDestroy(&viewer);

    // load stage cost matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetFromOptions(stageCostMatrix_);
    ierr = MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, PETSC_DECIDE, numActions_);
    CHKERRQ(ierr);
    std::cout << "g: " << localNumStates_ << " " << numActions_ << " " << numStates_ << " " << numActions_ << std::endl;
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
    ierr = VecSetSizes(nnz_, localNumStates_*numActions_, PETSC_DECIDE);
    CHKERRQ(ierr);
    std::cout << "nnz: " << localNumStates_*numActions_ << " " << numStates_*numActions_ << std::endl;
    VecSetUp(nnz_);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_nnz.c_str());
    VecLoad(nnz_, viewer);
    PetscViewerDestroy(&viewer);

    // Information about distribution on processes
    std::cout << "Rank: " << rank_ << " owns " << localNumStates_ << " states." << std::endl;
    PetscInt start_P, end_P, start_g, end_g, start_nnz, end_nnz;
    MatGetOwnershipRange(transitionProbabilityTensor_, &start_P, &end_P);
    MatGetOwnershipRange(stageCostMatrix_, &start_g, &end_g);
    VecGetOwnershipRange(nnz_, &start_nnz, &end_nnz);
    std::cout << "Rank: " << rank_ << " owns rows " << start_P << " to " << end_P << " of P." << std::endl;
    std::cout << "Rank: " << rank_ << " owns rows " << start_g << " to " << end_g << " of g." << std::endl;
    std::cout << "Rank: " << rank_ << " owns rows " << start_nnz << " to " << end_nnz << " of nnz." << std::endl;

    // cols
    PetscInt start_P_col, end_P_col, start_g_col, end_g_col;
    MatGetOwnershipRangeColumn(transitionProbabilityTensor_, &start_P_col, &end_P_col);
    MatGetOwnershipRangeColumn(stageCostMatrix_, &start_g_col, &end_g_col);
    std::cout << "Rank: " << rank_ << " owns cols " << start_P_col << " to " << end_P_col << " of P." << std::endl;
    std::cout << "Rank: " << rank_ << " owns cols " << start_g_col << " to " << end_g_col << " of g." << std::endl;

    return ierr;
}

#if 0
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
#endif