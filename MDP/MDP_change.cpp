#include "MDP.h"
#include <petscksp.h>
#include <mpi.h>
#include <iostream>
#include "../utils/Logger.h"
#include <chrono>
#include <algorithm>


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
    if (matrix == 0){
        if (transitionProbabilityTensor_ != nullptr)
        {
            MatDestroy(&transitionProbabilityTensor_);
        }
        MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
        MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_);
        MatSetFromOptions(transitionProbabilityTensor_);
        MatSetUp(transitionProbabilityTensor_);
        MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
        MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
        states.first = P_start_;
        states.second = P_end_;
        if (matrix==0){
            states.first /= numActions_;
            states.second /= numActions_;
        }
    } else if (matrix == 1){
        if (stageCostMatrix_ != nullptr)
        {
            MatDestroy(&stageCostMatrix_);
        }
        MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
        MatSetSizes(stageCostMatrix_, PETSC_DECIDE, PETSC_DECIDE, numStates_, numActions_);
        MatSetFromOptions(stageCostMatrix_);
        MatSetUp(stageCostMatrix_);
        MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
        MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
        states.first = g_start_;
        states.second = g_end_;
    } else{
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }    
    return states;
}

void MDP::fill_row(std::vector<int> &idxs, std::vector<double> &vals, int i, int matrix){
    Mat mat_ptr;
    if (matrix == 0)
    {
        mat_ptr = transitionProbabilityTensor_;
    }
    else if (matrix == 1)
    {
        mat_ptr = stageCostMatrix_;        
    }
    else
    {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }
    MatSetValues(mat_ptr, 1, &i, idxs.size(), idxs.data(), vals.data(), INSERT_VALUES);
}

void MDP::mat_asssembly_end(int matrix){
    if (matrix == 0){
        MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    } else if (matrix == 1){
        MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    } else{
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }
}