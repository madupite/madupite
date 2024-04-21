#include "MDP.h"
#include <petscksp.h>
// #include <mpi.h>
#include <iostream>
// #include "../utils/Logger.h"
#include <algorithm>
#include <chrono>

std::pair<int, int> MDP::getStateOwnershipRange() { return {p_start_ / numActions_, p_end_ / numActions_ }; }

std::pair<int, int> MDP::getMDPSize() { return { numStates_, numActions_ }; }

std::pair<int, int> MDP::request_states(int nstates, int mactions, int matrix, int preallocate)
{
    // TODO: after request_states the corresponding matrix should be setup to be filled
    // i.e. preallocate memory for the matrix (sizes might change during the process)
    std::pair<int, int> states;
    if (numStates_ != nstates) {
        numStates_      = nstates;
        localNumStates_ = PETSC_DECIDE;
        PetscCallThrow(PetscSplitOwnership(comm_, &localNumStates_, &numStates_));
    }
    numActions_ = mactions;
    if (matrix == 0) {
        if (transitionProbabilityTensor_ != nullptr) {
            PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
        }
        PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
        PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_));
        PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
        PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
        PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));
        states.first  = p_start_;
        states.second = p_end_;
        if (matrix == 0) {
            states.first /= numActions_;
            states.second /= numActions_;
        }
    } else if (matrix == 1) {
        if (stageCostMatrix_ != nullptr) {
            PetscCallThrow(MatDestroy(&stageCostMatrix_));
        }
        PetscCallThrow(MatCreate(comm_, &stageCostMatrix_));
        PetscCallThrow(MatSetSizes(stageCostMatrix_, PETSC_DECIDE, PETSC_DECIDE, numStates_, numActions_));
        PetscCallThrow(MatSetFromOptions(stageCostMatrix_));
        PetscCallThrow(MatSetUp(stageCostMatrix_));
        PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));
        states.first  = g_start_;
        states.second = g_end_;
    } else {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }
    return states;
}

void MDP::fillRow(std::vector<int>& idxs, std::vector<double>& vals, int i, int matrix)
{
    Mat* mat_ptr;
    if (matrix == 0) {
        mat_ptr = &transitionProbabilityTensor_;
    } else if (matrix == 1) {
        mat_ptr = &stageCostMatrix_;
    } else {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1); // todo: exit needed? other way to handle?
    }
    PetscCallThrow(MatSetValues(*mat_ptr, 1, &i, idxs.size(), idxs.data(), vals.data(), INSERT_VALUES));
}

void MDP::assembleMatrix(int matrix)
{                      // assembles the matrix; blocking call
    if (matrix == 0) { // transitionProbabilityTensor
        PetscCallThrow(MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    } else if (matrix == 1) { // stageCostMatrix
        PetscCallThrow(MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
    } else {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }
}

// Pre: localNumStates_, numStates_, numActions_ are set
void MDP::createStageCostMatrix()
{
    if (stageCostMatrix_ != nullptr) {
        PetscCallThrow(MatDestroy(&stageCostMatrix_));
    }
    PetscCallThrow(MatCreate(comm_, &stageCostMatrix_));
    PetscCallThrow(MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetType(stageCostMatrix_, MATDENSE));
    // MatSetFromOptions(stageCostMatrix_);
    PetscCallThrow(MatSetUp(stageCostMatrix_));
    PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));

    // fill
    double value;
    for(PetscInt stateInd = g_start_; stateInd < g_end_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            value = g_func_(stateInd, actionInd);
            PetscCallThrow(MatSetValue(stageCostMatrix_, stateInd, actionInd, value, INSERT_VALUES));
        }
    }
    assembleMatrix(1);
}

void MDP::createTransitionProbabilityTensorPrealloc()
{
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    auto [d_nz, d_nnz, o_nz, o_nnz] = p_nnz_;
    const PetscInt *d_nnz_ptr = d_nnz.data(), *o_nnz_ptr = o_nnz.data();
    if (d_nnz.empty())
        d_nnz_ptr = nullptr;
    if (o_nnz.empty())
        o_nnz_ptr = nullptr;
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatMPIAIJSetPreallocation(transitionProbabilityTensor_, d_nz, d_nnz_ptr, o_nz, o_nnz_ptr));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));

    // fill
    for(PetscInt stateInd = p_start_ / numActions_; stateInd < p_end_ / numActions_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = p_func_(stateInd, actionInd);
            PetscInt rowInd = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    assembleMatrix(0);
}

void MDP::createTransitionProbabilityTensor()
{
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));

    // fill
    for(PetscInt stateInd = p_start_ / numActions_; stateInd < p_end_ / numActions_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = p_func_(stateInd, actionInd);
            PetscInt rowInd = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    assembleMatrix(0);
}
