#pragma once

#include <ranges>
#include <string>
#include <vector>

#include <petscmat.h>

#include "madupite_errors.h"
#include "madupite_vector.h"

// using std::ranges::range;

enum MatrixType {
    Dense,
    Sparse,
};

class Matrix {
    Mat _mat;

public:
    ////////
    // Constructors, destructors and assignment
    ////////

    // Matrix(PetscInt rows, PetscInt cols);
    // Matrix(PetscInt rows, PetscInt cols, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz);

    Matrix(MPI_Comm comm, const std::string& name, const std::string& filename, MatrixType type = MatrixType::Sparse);

    ~Matrix() { MatDestroy(&_mat); }

    // Forbid copy and move for now
    Matrix(const Matrix&)            = delete;
    Matrix(Matrix&&)                 = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix& operator=(Matrix&&)      = delete;

    ////////
    // Operators
    ////////

    // single-value getter
    PetscScalar operator()(PetscInt row, PetscInt col) const
    {
        PetscScalar value;

        PetscCallThrow(MatGetValue(_mat, row, col, &value));
        return value;
    }

    // single-value setter
    void operator()(PetscInt row, PetscInt col, PetscScalar value) { PetscCallThrow(MatSetValue(_mat, row, col, value, INSERT_VALUES)); }

    // multi-value setter
    std::vector<PetscScalar> operator()(const std::vector<PetscInt>& rows, const std::vector<PetscInt>& cols)
    {
        PetscInt m = rows.size();
        PetscInt n = cols.size();

        PetscScalar* data = new PetscScalar[m * n];
        PetscCallThrow(MatGetValues(_mat, m, rows.data(), n, cols.data(), data));
        return std::vector<PetscScalar>(data, data + m * n);
    }

    ////////
    // Inline methods
    ////////

    // Get the MPI communicator
    MPI_Comm comm() const { return PetscObjectComm((PetscObject)_mat); }

    // Get the local size
    std::pair<PetscInt, PetscInt> localSize() const
    {
        PetscInt m, n;

        PetscCallThrow(MatGetLocalSize(_mat, &m, &n));
        return std::make_pair(m, n);
    }

    // Get the global size
    std::pair<PetscInt, PetscInt> size() const
    {
        PetscInt M, N;

        PetscCallThrow(MatGetSize(_mat, &M, &N));
        return std::make_pair(M, N);
    }

    // We currently don't need to have separate assemblyBegin and assemblyEnd
    void assemble()
    {
        PetscCallThrow(MatAssemblyBegin(_mat, MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(_mat, MAT_FINAL_ASSEMBLY));
    }

    // Matrix-vector multiplication
    void mult(const Vector& x, Vector& y) const { PetscCallThrow(MatMult(_mat, x.petscVec(), y.petscVec())); }

    // Transposed matrix-vector multiplication
    void multT(const Vector& x, Vector& y) const { PetscCallThrow(MatMultTranspose(_mat, x.petscVec(), y.petscVec())); }

    // Add another matrix
    void add(const Matrix& other, MatStructure structure = SAME_NONZERO_PATTERN) { MatAXPY(_mat, 1.0, other._mat, structure); }

    // Get the inner PETSc matrix
    Mat petscMat() const { return _mat; }

    ////////
    // Out-of-line method declarations
    ////////

    // Get row in AIJ format
    std::vector<PetscScalar> getRow(PetscInt row) const;
};
