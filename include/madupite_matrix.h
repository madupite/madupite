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

using ConstMat = const _p_Mat*;

// using std::ranges::range;

enum class MatrixType {
    Dense,
    Sparse,
};

enum class MatrixCategory {
    Dynamics,
    Cost,
};

using Costfunc = std::function<PetscScalar(PetscInt, PetscInt)>;
using Probfunc = std::function<std::pair<std::vector<PetscScalar>, std::vector<PetscInt>>(PetscInt, PetscInt)>;

struct MatrixPreallocation {
    PetscInt         d_nz = PETSC_DECIDE;
    std::vector<int> d_nnz;
    PetscInt         o_nz = PETSC_DECIDE;
    std::vector<int> o_nnz;
};

class Matrix {
    Layout _rowLayout;
    Layout _colLayout;

    // Inner PETSc matrix; should be accessed directly only in constructors and via petsc() otherwise
    Mat _mat = nullptr;

    // Private constructor setting communicator, name and type (but no size)
    Matrix(MPI_Comm comm, const std::string& name, MatrixType type);

public:
    ////////
    // Basic getters
    ////////

    // Get the inner PETSc matrix
    Mat petsc()
    {
        if (_mat == nullptr) {
            throw MadupiteException("Matrix is uninitialized");
        }
        return _mat;
    }

    // Get the inner PETSc matrix in the const form.
    // We currently pass
    //   const_cast<Mat>(petsc())
    // to PETSc functions as they currently don't take ConstMat arguments (const _p_Mat *).
    // This could be a good proposal for PETSc.
    ConstMat petsc() const
    {
        // reuse the non-const implementation
        return const_cast<ConstMat>(const_cast<Matrix*>(this)->petsc());
    }

    // Get the MPI communicator
    MPI_Comm comm() const { return PetscObjectComm((PetscObject)petsc()); }

    // Get the row layout
    const Layout& rowLayout() const { return _rowLayout; }

    // Get the column layout
    const Layout& colLayout() const { return _colLayout; }

    ////////
    // Constructors, destructors and assignment
    ////////

    // default constructor creates a 'null' matrix
    Matrix() = default;

    // create a matrix with given communicator, name, type and size with or without preallocation
    Matrix(MPI_Comm comm, const std::string& name, MatrixType type, const Layout& rowLayout, const Layout& colLayout,
        const MatrixPreallocation& pa = {});

    // destructor
    ~Matrix()
    {
        PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
    }

    // copy constructor (shallow)
    Matrix(const Matrix& other)
        : _rowLayout(other._rowLayout)
        , _colLayout(other._colLayout)
    {
        PetscCallThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
        _mat = other._mat;
        PetscCallThrow(PetscObjectReference((PetscObject)other._mat));
    }

    // copy assignment (shallow)
    Matrix& operator=(const Matrix& other)
    {
        if (this != &other) {
            _rowLayout = other._rowLayout;
            _colLayout = other._colLayout;

            PetscCallThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
            _mat = other._mat;
            PetscCallThrow(PetscObjectReference((PetscObject)other._mat));
        }
        return *this;
    }

    // move constructor
    Matrix(Matrix&& other) noexcept
        : _rowLayout(std::move(other._rowLayout))
        , _colLayout(std::move(other._colLayout))
    {
        PetscCallNoThrow(MatDestroy(&_mat)); // No-op if _mat is nullptr
        _mat = std::exchange(other._mat, nullptr);
    }

    // move assignment
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

    static Matrix fromFile(
        MPI_Comm comm, const std::string& name, const std::string& filename, MatrixCategory category, MatrixType type = MatrixType::Sparse);

    ////////
    // Operators
    ////////

    // boolean conversion
    explicit operator bool() const { return _mat != nullptr; }

    // single-value getter
    PetscScalar operator()(PetscInt row, PetscInt col) const
    {
        PetscScalar value;
        PetscCallThrow(MatGetValue(const_cast<Mat>(petsc()), row, col, &value));
        return value;
    }

    // single-value setter
    void operator()(PetscInt row, PetscInt col, PetscScalar value) { PetscCallThrow(MatSetValue(petsc(), row, col, value, INSERT_VALUES)); }

    // multi-value getter
    std::vector<PetscScalar> operator()(const std::vector<PetscInt>& rows, const std::vector<PetscInt>& cols) const
    {
        PetscInt m = rows.size();
        PetscInt n = cols.size();

        auto data = std::make_unique<PetscScalar[]>(m * n);
        PetscCallThrow(MatGetValues(const_cast<Mat>(petsc()), m, rows.data(), n, cols.data(), data.get()));
        return std::vector<PetscScalar>(&data[0], &data[m * n]);
    }

    ////////
    // Inline methods
    ////////

    // We currently don't need to have separate assemblyBegin and assemblyEnd
    void assemble()
    {
        PetscCallThrow(MatAssemblyBegin(petsc(), MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(petsc(), MAT_FINAL_ASSEMBLY));
    }

    // Matrix-vector multiplication
    void mult(const Vector& x, Vector& y) const { PetscCallThrow(MatMult(const_cast<Mat>(petsc()), const_cast<Vec>(x.petsc()), y.petsc())); }

    // Transposed matrix-vector multiplication
    void multT(const Vector& x, Vector& y) const
    {
        PetscCallThrow(MatMultTranspose(const_cast<Mat>(petsc()), const_cast<Vec>(x.petsc()), y.petsc()));
    }

    // Add another matrix
    void add(const Matrix& other, MatStructure structure = SAME_NONZERO_PATTERN)
    {
        PetscCallThrow(MatAXPY(petsc(), 1.0, const_cast<Mat>(other.petsc()), structure));
    }

    ////////
    // Out-of-line method declarations
    ////////

    // Get row in AIJ format
    std::vector<PetscScalar> getRow(PetscInt row) const;

    // write matrix to file
    void writeToFile(const std::string& filename, MatrixType type) const;
};
