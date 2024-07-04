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
}

// TODO PETSc feature proposal: it would be nice if the PETSc loader infers the correct type from the file
Matrix Matrix::loadFromFile(MPI_Comm comm, const std::string& name, const std::string& filename, MatrixType type)
{
    Matrix      A(comm, name, type);
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(A._mat, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
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
