//
// Created by robin on 27.04.23.
//

#include "MDP.h"
#include "TransitionMatrixGenerator.h"
#include <petscksp.h>

MDP::MDP(const PetscInt numStates, const PetscInt numActions, const PetscReal discountFactor) : numStates_(numStates),
                                                                                                numActions_(numActions),
                                                                                                discountFactor_(discountFactor) {
    //MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor);
    //MatCreate(PETSC_COMM_WORLD, &stageCostMatrix);
    // TODO initialize matrices
    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_ = nullptr;
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
}

// find min and argmin of  
PetscErrorCode MDP::extractGreedyPolicy(Vec &V, PetscInt *policy) {
    PetscErrorCode ierr;
    PetscInt *indices; // stores indices [actionInd*numStates, (actionInd+1)*numStates-1] for extracting values from P
    PetscMalloc1(numStates_, &indices);
    //PetscInt *greedyPolicy; // result
    //PetscMalloc1(numStates_, &greedyPolicy);

    Vec tmp; // contains block from P
    ierr = VecCreate(PETSC_COMM_WORLD, &tmp);
    ierr = VecSetType(tmp, VECSEQ); // todo: to be changed for parallel version
    CHKERRQ(ierr);
    ierr = VecSetSizes(tmp, PETSC_DECIDE, numStates_);


    PetscReal *tmpValues; // contains values to be stored in tmp
    PetscMalloc1(numStates_, &tmpValues);
    PetscInt *vecIndices; // contains indices [0, 1, 2, ..., numStates_-1] for setting values in tmp
    PetscMalloc1(numStates_, &vecIndices);
    std::iota(vecIndices, vecIndices + numStates_, 0); // initialize vecIndices to 0, 1, 2, ..., numStates_-1

    PetscReal minCost = std::numeric_limits<PetscReal>::max();
    PetscInt minAction = 0;

    for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            // extract transition probabilities
            std::iota(indices, indices + numStates_, 0); // initialize indices to 0, 1, 2, ..., numStates_-1
            ierr = MatGetValues(transitionProbabilityTensor_, 1, &stateInd, numStates_, indices, tmpValues);
            CHKERRQ(ierr);
            ierr = VecSetValues(tmp, numStates_, vecIndices, tmpValues, INSERT_VALUES);
            CHKERRQ(ierr);
            // extract stage costs
            PetscReal stageCost;
            ierr = MatGetValues(stageCostMatrix_, 1, &stateInd, 1, &actionInd, &stageCost);
            CHKERRQ(ierr);
            // compute g + gamma PV
            PetscReal cost = 0;
            ierr = VecDot(tmp, V, &cost);
            cost = stageCost + discountFactor_ * cost;
            if(cost < minCost) {
                minCost = cost;
                minAction = actionInd;
            }
        }
        //greedyPolicy[stateInd] = minAction;
        policy[stateInd] = minAction;
    }
    //std::swap(policy, greedyPolicy); // s.t. greedyPolicy is returned
    ierr = PetscFree(indices);
    CHKERRQ(ierr);
    ierr = PetscFree(tmpValues);
    CHKERRQ(ierr);
    ierr = PetscFree(vecIndices);
    CHKERRQ(ierr);
    //ierr = PetscFree(policy);
    //CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);
    CHKERRQ(ierr);
    return ierr;
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
    PetscReal *tmpValues; // contains values to be stored in tmp
    PetscMalloc1(numStates_, &tmpValues);

    PetscReal stageCost;

    for(PetscInt stateInd = 0; stateInd < numStates_; ++stateInd) {
        // extract stage cost
        ierr = MatGetValues(stageCostMatrix_, 1, &stateInd, 1, &policy[stateInd], &stageCost);
        ierr = VecSetValue(stageCosts, stateInd, stageCost, INSERT_VALUES);
        // extract transition probabilities
        std::iota(indices, indices + numStates_, policy[stateInd]*numStates_); // initialize indices to [policy[stateInd]*numStates, (policy[stateInd]+1)*numStates-1]
        ierr = MatGetValues(transitionProbabilityTensor_, 1, &stateInd, numStates_, indices, tmpValues);
        ierr = MatSetValues(transitionProbabilities, 1, &stateInd, numStates_, baseIndices, tmpValues, INSERT_VALUES);
    }

    PetscFree(indices);
    PetscFree(baseIndices);
    PetscFree(tmpValues);
    return ierr;
}

PetscErrorCode MDP::iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, PetscReal alpha) {
    PetscErrorCode ierr;
    const PetscReal rtol = 1e-14;

    // ksp solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    ierr = KSPSetOperators(ksp, jacobian, jacobian);
    ierr = KSPSetType(ksp, KSPGMRES);
    ierr = KSPSetFromOptions(ksp);
    ierr = KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); // todo: custom stopping criterion

    ierr = KSPSolve(ksp, stageCosts, V);
    CHKERRQ(ierr);

    return ierr;
}

std::vector<PetscInt> MDP::inexactPolicyIteration(Vec &V0, const PetscInt maxIter, PetscReal alpha) {

    std::vector<PetscInt> policy(numStates_); // result

    // todo: change jacobian to ShellMat
    Mat jacobian;

    Vec V, V_old;
    VecDuplicate(V0, &V);
    VecCopy(V0, V);
    VecDuplicate(V, &V_old);

    Vec stageCosts;
    VecCreate(PETSC_COMM_WORLD, &stageCosts);
    VecSetSizes(stageCosts, PETSC_DECIDE, numStates_);

    for(PetscInt i = 0; i < maxIter; ++i) {
        extractGreedyPolicy(V, policy.data());
        constructFromPolicy(policy.data(), jacobian, stageCosts);
        VecCopy(V, V_old);
        iterativePolicyEvaluation(jacobian, stageCosts, V, alpha);
    }

    return {};
}

PetscErrorCode MDP::loadFromBinaryFile(std::string filename_P, std::string filename_g) {
    PetscErrorCode ierr = 0;

    if(filename_P.empty()) {
        filename_P = "../data/P_" + std::to_string(numStates_) + "_" + std::to_string(numActions_) + ".bin";
    }
    if(filename_g.empty()) {
        filename_g = "../data/g_" + std::to_string(numStates_) + "_" + std::to_string(numActions_) + ".bin";
    }

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

    return ierr;
}
