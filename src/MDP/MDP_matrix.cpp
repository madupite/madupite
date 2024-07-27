#include "MDP_matrix.h"
#include "madupite_matrix.h"

Matrix createTransitionProbabilityTensor(
    MPI_Comm comm, const std::string& name, PetscInt numStates, PetscInt numActions, Probfunc func, const MatrixPreallocation& pa)
{
    PetscInt localStates = PETSC_DECIDE;
    PetscSplitOwnership(comm, &localStates, &numStates);
    Layout rowLayout(comm, localStates * numActions, true);
    Layout colLayout(comm, localStates, true);
    Matrix A(comm, name, MatrixType::Sparse, rowLayout, colLayout, pa);
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
    PetscSplitOwnership(comm, &localStates, &numStates);
    Layout   rowLayout(comm, localStates, true);
    PetscInt localActions = PETSC_DECIDE;
    PetscSplitOwnership(comm, &localActions, &numActions);
    Layout colLayout(comm, localActions, true);
    Matrix A(comm, name, MatrixType::Dense, rowLayout, colLayout);
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
