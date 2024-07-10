#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <petscmat.h>

#include "madupite_errors.h"
#include "madupite_layout.h"
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

class Matrix {
    Layout _rowLayout;
    Layout _colLayout;
    Mat    _mat;

    // Private constructor setting communicator, name and type (but no size)
    Matrix(MPI_Comm comm, const std::string& name, MatrixType type);

public:
    ////////
    // Constructors, destructors and assignment
    ////////

    // Create a matrix with given communicator, name, type and size with or without preallocation
    Matrix(MPI_Comm comm, const std::string& name, MatrixType type, const Layout& rowLayout, const Layout& colLayout,
        const MatrixPreallocation& pa = {});

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
        : _rowLayout(std::move(other._rowLayout))
        , _colLayout(std::move(other._colLayout))
    {
        PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
        _mat = std::exchange(other._mat, nullptr);
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept
    {
        if (this != &other) {
            _rowLayout = std::move(other._rowLayout);
            _colLayout = std::move(other._colLayout);

            PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
            _mat = std::exchange(other._mat, nullptr);
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

    // We currently don't need to have separate assemblyBegin and assemblyEnd
    void assemble()
    {
        PetscCallThrow(MatAssemblyBegin(_mat, MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(_mat, MAT_FINAL_ASSEMBLY));
    }

    // Matrix-vector multiplication
    void mult(const Vector& x, Vector& y) const { PetscCallThrow(MatMult(_mat, x.petsc(), y.petsc())); }

    // Transposed matrix-vector multiplication
    void multT(const Vector& x, Vector& y) const { PetscCallThrow(MatMultTranspose(_mat, x.petsc(), y.petsc())); }

    // Add another matrix
    void add(const Matrix& other, MatStructure structure = SAME_NONZERO_PATTERN) { PetscCallThrow(MatAXPY(_mat, 1.0, other._mat, structure)); }

    // Get the inner PETSc matrix
    Mat petsc() { return _mat; }

    ////////
    // Out-of-line method declarations
    ////////

    // Get row in AIJ format
    std::vector<PetscScalar> getRow(PetscInt row) const;
};
