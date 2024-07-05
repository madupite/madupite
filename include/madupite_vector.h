#pragma once

#include <string>
#include <vector>

#include <petscvec.h>

#include "madupite_errors.h"

class Vector {
    Vec _vec;

public:
    ////////
    // Constructors, destructors and assignment
    ////////

    // Create a zero vector
    Vector(MPI_Comm comm, const std::string& name, PetscInt size);

    // Load a vector from file
    Vector(MPI_Comm comm, const std::string& name, const std::string& filename);

    // Create a vector from an std::vector
    Vector(MPI_Comm comm, const std::string& name, const std::vector<PetscScalar>& data);

    // Destructor
    ~Vector() { VecDestroy(&_vec); }

    // Forbid copy and move for now
    Vector(const Vector&)            = delete;
    Vector(Vector&&)                 = delete;
    Vector& operator=(const Vector&) = delete;
    Vector& operator=(Vector&&)      = delete;

    ////////
    // Operators
    ////////

    // single-value getter for floating point types
    PetscScalar operator()(PetscInt index) const
    {
        PetscScalar value;

        PetscCallThrow(VecGetValues(_vec, 1, &index, &value));
        return value;
    }

    // single-value setter for floating point types
    void operator()(PetscInt index, PetscScalar value) { PetscCallThrow(VecSetValues(_vec, 1, &index, &value, INSERT_VALUES)); }

    ////////
    // Inline methods
    ////////

    // Get the MPI communicator
    MPI_Comm comm() const { return PetscObjectComm((PetscObject)_vec); }

    // Get the inner PETSc vector
    Vec petscVec() const { return _vec; }

    // Get the local size
    PetscInt localSize() const
    {
        PetscInt m;

        PetscCallThrow(VecGetLocalSize(_vec, &m));
        return m;
    }

    // Get the global size
    PetscInt size() const
    {
        PetscInt M;

        PetscCallThrow(VecGetSize(_vec, &M));
        return M;
    }

    // Assemble the vector
    void assemble()
    {
        PetscCallThrow(VecAssemblyBegin(_vec));
        PetscCallThrow(VecAssemblyEnd(_vec));
    }

    std::pair<PetscInt, PetscInt> getOwnershipRange() const
    {
        PetscInt lo, hi;
        PetscCallThrow(VecGetOwnershipRange(_vec, &lo, &hi));
        return std::make_pair(lo, hi);
    }

    ////////
    // Out-of-line method declarations
    ////////

    void write(const std::string& filename) const;
};
