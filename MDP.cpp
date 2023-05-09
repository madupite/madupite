//
// Created by robin on 27.04.23.
//

#include "MDP.h"
#include <numeric>
#include <petscksp.h>

MDP::MDP(const PetscInt numStates, const PetscInt numActions, const PetscReal discountFactor) : numStates_(numStates),
                                                                                                numActions_(numActions),
                                                                                                discountFactor_(discountFactor) {
    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_ = nullptr;
    nnz_ = nullptr;
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
    MatDestroy(&nnz_);
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
            MatCreate(PETSC_COMM_WORLD, &P);
            MatSetType(P, MATSEQAIJ);
            MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, numStates_, numStates_);
            MatSetUp(P);

            std::fill(tmppolicy, tmppolicy + numStates_, actionInd);
            constructFromPolicy(tmppolicy, P, g);
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

    return 0;
}

PetscErrorCode MDP::constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    // extract cols [ policy[i]*numStates, (policy[i]+1)*numStates ) from transitionProbabilities for every i (row)
    // extract stageCostMatrix_[i, policy[i]] for every i (row)
    PetscErrorCode ierr;
    PetscInt *indices; // stores indices [actionInd*numStates, (actionInd+1)*numStates-1] for extracting values from P
    PetscMalloc1(numStates_, &indices);
    PetscInt *baseIndices; // stores indices [0, 1, 2, ..., numStates_-1] for inserting values into P^\pi
    PetscMalloc1(numStates_, &baseIndices);
    std::iota(baseIndices, baseIndices + numStates_, 0); // initialize baseIndices to 0, 1, 2, ..., numStates_-1
    PetscReal *P_values; // contains values to be stored in P^\pi
    PetscMalloc1(numStates_, &P_values);
    PetscReal *g_values; // contains values to be stored in g^\pi
    PetscMalloc1(numStates_, &g_values);

    PetscInt *nnz;
    PetscMalloc1(numStates_, &nnz); // stores number of nonzeros in each row of P^\pi
    // construct nnz array
    auto *it = nnz;
    for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) {
        double tmp;
        MatGetValue(nnz_, stateInd, policy[stateInd], &tmp);
        *it = static_cast<PetscInt>(tmp);
        ++it;
    }
    ierr = MatSeqAIJSetPreallocation(transitionProbabilities, 0, nnz); CHKERRQ(ierr);

    for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) {
        // extract stage cost
        ierr = MatGetValue(stageCostMatrix_, stateInd, policy[stateInd], &g_values[stateInd]); CHKERRQ(ierr);
        // extract transition probabilities
        std::iota(indices, indices + numStates_, policy[stateInd]*numStates_); // initialize indices to [policy[stateInd]*numStates, (policy[stateInd]+1)*numStates-1]
        ierr = MatGetValues(transitionProbabilityTensor_, 1, &stateInd, numStates_, indices, P_values); CHKERRQ(ierr);
        ierr = MatSetValues(transitionProbabilities, 1, &stateInd, numStates_, baseIndices, P_values, INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecSetValues(stageCosts, numStates_, baseIndices, g_values, INSERT_VALUES); CHKERRQ(ierr);

    PetscFree(indices);
    PetscFree(baseIndices);
    PetscFree(P_values);
    PetscFree(g_values);
    delete[] nnz;
    return ierr;
}

PetscErrorCode MDP::iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, PetscReal alpha) {
    PetscErrorCode ierr;
    const PetscReal rtol = 1e-15;

    // ksp solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, jacobian, jacobian); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr); // todo: custom stopping criterion

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

        extractGreedyPolicy(V, policy.data(), GreedyPolicyType::V1);
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
        MatCreate(PETSC_COMM_WORLD, &jacobian);
        MatSetSizes(jacobian, PETSC_DECIDE, PETSC_DECIDE, numStates_, numStates_);
        MatSetType(jacobian, MATSEQAIJ);
        MatSetUp(jacobian);
        constructFromPolicy(policy.data(), jacobian, stageCosts); // returns g
        MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY);
        MatScale(jacobian, -discountFactor_); // needs assembled matrix
        MatShift(jacobian, 1);
        VecCopy(V, V_old);
        iterativePolicyEvaluation(jacobian, stageCosts, V, alpha);
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

PetscErrorCode MDP::loadFromBinaryFile(std::string filename_P, std::string filename_g, std::string filename_nnz) {
    PetscErrorCode ierr = 0;

    // load transition probability tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename_P.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(transitionProbabilityTensor_, viewer);
    PetscViewerDestroy(&viewer);

    // load stage cost matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename_g.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(stageCostMatrix_, viewer);
    PetscViewerDestroy(&viewer);

    // load nnz
    if(filename_nnz.empty()) return 0;
    MatCreate(PETSC_COMM_WORLD, &nnz_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename_nnz.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(nnz_, viewer);
    PetscViewerDestroy(&viewer);

    return ierr;
}
