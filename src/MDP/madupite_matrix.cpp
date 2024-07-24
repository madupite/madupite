#include "madupite_matrix.h"
#include "madupite_errors.h"
#include "petscmat.h"

std::string Matrix::typeToString(MatrixType type)
{
    switch (type) {
    case MatrixType::Dense:
        return MATDENSE;
    case MatrixType::Sparse:
        return MATAIJ;
    default:
        throw MadupiteException("Unsupported matrix type");
    }
}

Matrix::Matrix(MPI_Comm comm, const std::string& name, MatrixType type)
{
    const auto typeStr = Matrix::typeToString(type);

    PetscCallThrow(MatCreate(comm, &_mat));
    PetscCallThrow(PetscObjectSetName((PetscObject)_mat, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_mat, (name + '_').c_str()));
    PetscCallThrow(MatSetType(_mat, typeStr.c_str()));
    PetscCallThrow(MatSetFromOptions(_mat));
    // This private constructor does not set the layouts; they should always be set in the public constructors
}

Matrix::Matrix(
    MPI_Comm comm, const std::string& name, MatrixType type, const Layout& rowLayout, const Layout& colLayout, const MatrixPreallocation& pa)
    : Matrix(comm, name, type)
{
    _rowLayout = rowLayout;
    _colLayout = colLayout;
    PetscCallThrow(MatSetLayouts(_mat, rowLayout.petsc(), colLayout.petsc()));
    auto d_nnz_ptr = pa.d_nnz.empty() ? nullptr : pa.d_nnz.data();
    auto o_nnz_ptr = pa.d_nnz.empty() ? nullptr : pa.d_nnz.data();
    // for any matrix type, only one of the following applies and the rest is no-op
    PetscCallThrow(MatSeqAIJSetPreallocation(_mat, pa.d_nz, d_nnz_ptr));
    PetscCallThrow(MatMPIAIJSetPreallocation(_mat, pa.d_nz, d_nnz_ptr, pa.o_nz, o_nnz_ptr));
    PetscCallThrow(MatSeqDenseSetPreallocation(_mat, nullptr));
    PetscCallThrow(MatMPIDenseSetPreallocation(_mat, nullptr));
    PetscCallThrow(MatSetUp(_mat));
}

// TODO PETSc feature proposal: it would be nice if the PETSc loader infers the correct type from the file
Matrix Matrix::fromFile(MPI_Comm comm, const std::string& name, const std::string& filename, MatrixCategory category, MatrixType type)
{
    Matrix      A(comm, name, type);
    PetscViewer viewer;

    PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
    PetscInt sizes[4];
    PetscInt dummy;
    // must read the following first such that we can specify the distribution of rows (local rows) by ourselves
    // since we split by the granularity of states and not the rows as equally as possible
    // to ensure that all data associated with a state is on the same process
    // Read ClassId, Rows, Cols, NNZ
    PetscCallThrow(PetscViewerBinaryRead(viewer, sizes, 4, &dummy, PETSC_INT));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    switch (category) {
    case MatrixCategory::Dynamics: {
        PetscInt numStates      = sizes[2];
        PetscInt numActions     = sizes[1] / sizes[2];
        PetscInt localNumStates = PETSC_DECIDE;
        PetscCallThrow(PetscSplitOwnership(PETSC_COMM_WORLD, &localNumStates, &numStates));
        PetscCallThrow(MatSetSizes(A._mat, localNumStates * numActions, PETSC_DECIDE, PETSC_DECIDE, sizes[2]));
        break;
    }
    case MatrixCategory::Cost: {
        PetscInt numStates      = sizes[1];
        PetscInt numActions     = sizes[2];
        PetscInt localNumStates = PETSC_DECIDE;
        PetscCallThrow(PetscSplitOwnership(PETSC_COMM_WORLD, &localNumStates, &numStates));
        PetscCallThrow(MatSetSizes(A._mat, localNumStates, PETSC_DECIDE, PETSC_DECIDE, sizes[2]));
        break;
    }
    default:
        throw MadupiteException("Unsupported matrix category");
    }

    PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(A._mat, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));

    PetscLayout pRowLayout, pColLayout;
    PetscCallThrow(MatGetLayouts(A._mat, &pRowLayout, &pColLayout));
    A._rowLayout = Layout(pRowLayout);
    A._colLayout = Layout(pColLayout);
    return A;
}

// Get row in AIJ format
std::vector<PetscScalar> Matrix::getRow(PetscInt row) const
{
    PetscInt           n;
    const PetscScalar* data;

    PetscCallThrow(MatGetRow(_mat, row, &n, nullptr, &data));
    auto result = std::vector<PetscScalar>(data, data + n);
    PetscCallThrow(MatRestoreRow(_mat, row, &n, nullptr, &data));
    return result;
}
