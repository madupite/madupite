#include "MDP.h"
#include <petscksp.h>
#include <mpi.h>
#include <iostream>
#include "../utils/Logger.h"
#include <chrono>
#include <algorithm>


std::pair<int, int> MDP::getStateOwnershipRange(){
    return {P_start_ / numActions_, P_end_ / numActions_};
}

std::pair<int, int> MDP::getMDPSize(){
    return {numStates_, numActions_};
}

std::pair<int, int> MDP::request_states(int nstates, int mactions, int matrix, int preallocate){
    // TODO: after request_states the corresponding matrix should be setup to be filled
    // i.e. preallocate memory for the matrix (sizes might change during the process)
    std::pair<int, int> states;
    if (numStates_!=nstates){
        numStates_ = nstates;
        localNumStates_ = PETSC_DECIDE;
        PetscSplitOwnership(PETSC_COMM_WORLD, &localNumStates_, &numStates_);
    }
    numActions_ = mactions;
    if (matrix == 0) {
        if (transitionProbabilityTensor_ != nullptr) {
            MatDestroy(&transitionProbabilityTensor_);
        }
        MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
        MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_);
        MatSetFromOptions(transitionProbabilityTensor_);
        MatSetUp(transitionProbabilityTensor_);
        MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
        states.first = P_start_;
        states.second = P_end_;
        if (matrix==0) {
            states.first /= numActions_;
            states.second /= numActions_;
        }
    } else if (matrix == 1) {
        if (stageCostMatrix_ != nullptr) {
            MatDestroy(&stageCostMatrix_);
        }
        MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
        MatSetSizes(stageCostMatrix_, PETSC_DECIDE, PETSC_DECIDE, numStates_, numActions_);
        MatSetFromOptions(stageCostMatrix_);
        MatSetUp(stageCostMatrix_);
        MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
        states.first = g_start_;
        states.second = g_end_;
    } else { 
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }    
    return states;
}

void MDP::fillRow(std::vector<int> &idxs, std::vector<double> &vals, int i, int matrix){
    Mat *mat_ptr;
    if (matrix == 0) {
        mat_ptr = &transitionProbabilityTensor_;
    }
    else if (matrix == 1) {
        mat_ptr = &stageCostMatrix_;        
    }
    else {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1); // todo: exit needed? other way to handle?
    }
    MatSetValues(*mat_ptr, 1, &i, idxs.size(), idxs.data(), vals.data(), INSERT_VALUES);
}

void MDP::assembleMatrix(int matrix){ // assembles the matrix; blocking call
    if (matrix == 0) { // transitionProbabilityTensor
        MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    } else if (matrix == 1) { // stageCostMatrix
        MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    } else {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }
}

PetscErrorCode MDP::createCostMatrix(){
    if (stageCostMatrix_ != nullptr) {
        MatDestroy(&stageCostMatrix_);
    }
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_);
    MatSetType(stageCostMatrix_, MATDENSE);
    // MatSetFromOptions(costMatrix_);
    MatSetUp(stageCostMatrix_);
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
    return 0;
}

PetscErrorCode MDP::createTransitionProbabilityTensor(PetscInt d_nz, const std::vector<int> &d_nnz, PetscInt o_nz, const std::vector<int> &o_nnz){
    if (transitionProbabilityTensor_ != nullptr) {
        MatDestroy(&transitionProbabilityTensor_);
    }
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_);
    MatSetFromOptions(transitionProbabilityTensor_);
    MatMPIAIJSetPreallocation(transitionProbabilityTensor_, d_nz, d_nnz.data(), o_nz, o_nnz.data());
    MatSetUp(transitionProbabilityTensor_);
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    return 0;
}

PetscErrorCode MDP::createTransitionProbabilityTensor(){
    if (transitionProbabilityTensor_ != nullptr) {
        MatDestroy(&transitionProbabilityTensor_);
    }
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_);
    MatSetFromOptions(transitionProbabilityTensor_);
    MatSetUp(transitionProbabilityTensor_);
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    return 0;
}

