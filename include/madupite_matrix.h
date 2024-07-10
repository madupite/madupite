#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <petscmat.h>

#include "madupite_errors.h"
#include "madupite_vector.h"

// using std::ranges::range;

enum class MatrixType {
    Dense,
    Sparse,
};

using Costfunc = std::function<double(int, int)>;
using Probfunc = std::function<std::pair<std::vector<double>, std::vector<int>>(int, int)>;

struct MatrixPreallocation {
    PetscInt         d_nz = PETSC_DECIDE;
    std::vector<int> d_nnz;
    PetscInt         o_nz = PETSC_DECIDE;
    std::vector<int> o_nnz;
};

// one of one of (m, M) and (n, N) must be set
class MatrixLayout {
    PetscLayout rowLayout;
    PetscLayout colLayout;

    // Deleted constructors and assignment operators to prevent copying/moving
    MatrixLayout()                                     = delete;
    MatrixLayout(const MatrixLayout&)                  = delete;
    MatrixLayout(MatrixLayout&&)                       = delete;
    const MatrixLayout& operator=(const MatrixLayout&) = delete;
    const MatrixLayout& operator=(MatrixLayout&&)      = delete;

public:
    MatrixLayout(MPI_Comm comm, PetscInt m = PETSC_DECIDE, PetscInt n = PETSC_DECIDE, PetscInt M = PETSC_DECIDE, PetscInt N = PETSC_DECIDE)
    {
        // Ensure at least one of (m, M) and (n, N) is set
        if ((m == PETSC_DECIDE && M == PETSC_DECIDE) || (n == PETSC_DECIDE && N == PETSC_DECIDE)) {
            throw std::invalid_argument("At least one of (m, M) and one of (n, N) must be set explicitly.");
        }
        PetscCallThrow(PetscLayoutCreate(comm, &rowLayout));
        PetscCallThrow(PetscLayoutCreate(comm, &colLayout));
        PetscCallThrow(PetscLayoutSetLocalSize(rowLayout, m));
        PetscCallThrow(PetscLayoutSetLocalSize(colLayout, n));
        PetscCallThrow(PetscLayoutSetSize(rowLayout, M));
        PetscCallThrow(PetscLayoutSetSize(colLayout, N));
        PetscCallThrow(PetscLayoutSetUp(rowLayout));
        PetscCallThrow(PetscLayoutSetUp(colLayout));
    }

    const PetscLayout row() const { return rowLayout; }
    const PetscLayout col() const { return colLayout; }

    ~MatrixLayout()
    {
        PetscCallNoThrow(PetscLayoutDestroy(&rowLayout));
        PetscCallNoThrow(PetscLayoutDestroy(&colLayout));
    }
};

class Matrix {
    Mat _mat;

    // Private constructor setting communicator, name and type (but no size)
    Matrix(MPI_Comm comm, const std::string& name, MatrixType type);

public:
    ////////
    // Constructors, destructors and assignment
    ////////

    // Create a matrix with given communicator, name, type and size with or without preallocation
    Matrix(MPI_Comm comm, const std::string& name, MatrixType type, const MatrixLayout& layout, const MatrixPreallocation& preallocation = {});

    // Destructor
    ~Matrix()
    {
        PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
    }

    // Forbid copy for now
    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Move constructor
    Matrix(Matrix&& other) noexcept
    {
        PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
        _mat       = other._mat;
        other._mat = nullptr;
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept
    {
        if (this != &other) {
            PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
            _mat       = other._mat;
            other._mat = nullptr;
        }
        return *this;
    }

    ////////
    // Static methods
    ////////
    static std::string typeToString(MatrixType type);

    static Matrix fromFile(MPI_Comm comm, const std::string& name, const std::string& filename, MatrixType type = MatrixType::Sparse);

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

    // multi-value getter
    std::vector<PetscScalar> operator()(const std::vector<PetscInt>& rows, const std::vector<PetscInt>& cols) const
    {
        PetscInt m = rows.size();
        PetscInt n = cols.size();

        auto data = std::make_unique<PetscScalar[]>(m * n);
        PetscCallThrow(MatGetValues(_mat, m, rows.data(), n, cols.data(), data.get()));
        return std::vector<PetscScalar>(&data[0], &data[m * n]);
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

    // Get the ownership range
    std::pair<PetscInt, PetscInt> getOwnershipRange() const
    {
        PetscInt lo, hi;
        PetscCallThrow(MatGetOwnershipRange(_mat, &lo, &hi));
        return std::make_pair(lo, hi);
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
    void add(const Matrix& other, MatStructure structure = SAME_NONZERO_PATTERN) { PetscCallThrow(MatAXPY(_mat, 1.0, other._mat, structure)); }

    // Get the inner PETSc matrix
    Mat petscMat() const { return _mat; }

    ////////
    // Out-of-line method declarations
    ////////

    // Get row in AIJ format
    std::vector<PetscScalar> getRow(PetscInt row) const;
};
