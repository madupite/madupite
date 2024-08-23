#pragma once

#include "madupite_matrix.h"

// matrices constructed specifically for MDP

// just sparse
template <typename comm_t>
Matrix createTransitionProbabilityTensor(
    comm_t comm_arg, const std::string& name, PetscInt numStates, PetscInt numActions, Probfunc func, const MatrixPreallocation& preallocation = {});

template <typename comm_t>
Matrix createStageCostMatrix(comm_t comm_arg, const std::string& name, PetscInt numStates, PetscInt numActions, Costfunc func);
