#pragma once

#include "madupite_matrix.h"
#include <memory>

// matrices constructed specifically for MDP

// just sparse
Matrix createTransitionProbabilityTensor(
    MPI_Comm comm, const std::string& name, PetscInt numStates, PetscInt numActions, Probfunc func, const MatrixPreallocation& preallocation = {});

Matrix createStageCostMatrix(MPI_Comm comm, const std::string& name, PetscInt numStates, PetscInt numActions, Costfunc func);
