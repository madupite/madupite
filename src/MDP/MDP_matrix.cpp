#include "MDP_matrix.h"
#include "MDP.h"

Matrix createTransitionProbabilityTensor(
    MPI_Comm comm, const std::string& name, PetscInt numStates, PetscInt numActions, Probfunc func, const MatrixPreallocation& pa)
{
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

Matrix createStageCostMatrix(MPI_Comm comm, const std::string& name, PetscInt numStates, PetscInt numActions, Costfunc func)
{
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
