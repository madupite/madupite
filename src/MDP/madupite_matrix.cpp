#include "madupite_matrix.h"
#include "madupite_errors.h"
#include "petscmat.h"

// TODO PETSc feature proposal: it would be nice if the PETSc loader infers the correct type from the file
Matrix::Matrix(MPI_Comm comm, const std::string& name, const std::string& filename, MatrixType type)
{
    const char* matType;

    switch (type) {
    case MatrixType::Dense:
        matType = MATDENSE;
        break;
    case MatrixType::Sparse:
        matType = MATAIJ;
        break;
    default:
        throw MadupiteException("Unsupported matrix type");
    }

    PetscCallThrow(MatCreate(comm, &_mat));
    PetscCallThrow(PetscObjectSetName((PetscObject)_mat, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_mat, (name + '_').c_str()));
    PetscCallThrow(MatSetType(_mat, matType));
    PetscCallThrow(MatSetFromOptions(_mat));
    {
        PetscViewer viewer;
        PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
        PetscCallThrow(MatLoad(_mat, viewer));
        PetscCallThrow(PetscViewerDestroy(&viewer));
    }
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
