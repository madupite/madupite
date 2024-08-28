#include "mdp_matrix.h"
#include "madupite_matrix.h"
#include "mdp.h"
#include "utils.h"
#include <petscmat.h>

template <typename comm_t>
Matrix createTransitionProbabilityTensor(
    comm_t comm_arg, const std::string& name, PetscInt numStates, PetscInt numActions, Probfunc func, const MatrixPreallocation& pa)
{
    MPI_Comm comm        = convertComm(comm_arg);
    PetscInt localStates = PETSC_DECIDE;
    PetscSplitOwnership(Madupite::getCommWorld(), &localStates, &numStates);
    Layout rowLayout(Madupite::getCommWorld(), localStates * numActions, true);
    Layout colLayout(Madupite::getCommWorld(), localStates, true);
    Matrix A(Madupite::getCommWorld(), name, MatrixType::Sparse, rowLayout, colLayout, pa);
    auto   lo = rowLayout.start();
    auto   hi = rowLayout.end();

    // fill
    for (PetscInt stateInd = lo / numActions; stateInd < hi / numActions; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions; ++actionInd) {
            auto [values, indices] = func(stateInd, actionInd);
            auto rowInd            = stateInd * numActions + actionInd;
            // TODO wrapping function/operator? MatGetRow()/MatRestoreRow()?
            PetscCallThrow(MatSetValues(A.petsc(), 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    A.assemble();
    return A;
}
template Matrix createTransitionProbabilityTensor<int>(
    int comm_arg, const std::string& name, PetscInt numStates, PetscInt numActions, Probfunc func, const MatrixPreallocation& prealloc);

template <typename comm_t>
Matrix createStageCostMatrix(comm_t comm_arg, const std::string& name, PetscInt numStates, PetscInt numActions, Costfunc func)
{
    MPI_Comm comm        = convertComm(comm_arg);
    PetscInt localStates = PETSC_DECIDE;
    PetscSplitOwnership(Madupite::getCommWorld(), &localStates, &numStates);
    Layout   rowLayout(Madupite::getCommWorld(), localStates, true);
    PetscInt localActions = PETSC_DECIDE;
    PetscSplitOwnership(Madupite::getCommWorld(), &localActions, &numActions);
    Layout colLayout(Madupite::getCommWorld(), localActions, true);
    Matrix A(Madupite::getCommWorld(), name, MatrixType::Dense, rowLayout, colLayout);
    auto   g_start_ = rowLayout.start();
    auto   g_end_   = rowLayout.end();

    // fill
    for (PetscInt stateInd = g_start_; stateInd < g_end_; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions; ++actionInd) {
            auto value = func(stateInd, actionInd);
            // TODO wrapping function/operator? MatGetRow()/MatRestoreRow()?
            PetscCallThrow(MatSetValue(A.petsc(), stateInd, actionInd, value, INSERT_VALUES));
        }
    }
    A.assemble();
    return A;
}
template Matrix createStageCostMatrix<int>(int comm_arg, const std::string& name, PetscInt numStates, PetscInt numActions, Costfunc func);
