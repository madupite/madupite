#pragma once

#include <string>
#include <utility>
#include <vector>

#include <petscvec.h>

#include "madupite_errors.h"
#include "madupite_layout.h"

class Vector {
    Vec    _vec;
    Layout _layout;

    Vector(MPI_Comm comm, const std::string& name);

public:
    ////////
    // Basic getters
    ////////

    // Get the MPI communicator
    MPI_Comm comm() const { return PetscObjectComm((PetscObject)_vec); }

    // Get the name
    std::string name() const
    {
        const char* name;
        PetscCallThrow(PetscObjectGetName((PetscObject)_vec, &name));
        return std::string(name);
    }

    // Get the inner PETSc vector
    Vec petscVec() const { return _vec; }

    // Get the layout
    const Layout& layout() const { return _layout; }

    ////////
    // Constructors, destructors and assignment
    ////////

    // Create a zero vector
    Vector(MPI_Comm comm, const std::string& name, const Layout& layout)
        : Vector(comm, name)
    {
        _layout = layout;
        PetscCallThrow(VecSetLayout(_vec, _layout.petsc()));
        PetscCallThrow(VecZeroEntries(_vec));
    }

    // Create a vector from an std::vector
    Vector(MPI_Comm comm, const std::string& name, const std::vector<PetscScalar>& data);

    // Destructor
    ~Vector() { PetscCallNoThrow(VecDestroy(&_vec)); }

    // Copy constructor
    Vector(const Vector& other)
        : Vector(other.comm(), other.name(), other.layout())
    {
        PetscCallThrow(VecCopy(other._vec, _vec));
    }

    // Copy assignment
    Vector& operator=(const Vector& other)
    {
        if (this != &other) {
            if (other._layout != _layout) {
                throw MadupiteException("Vector layout does not match");
            }
            PetscCallThrow(VecCopy(other._vec, _vec));
        }
        return *this;
    }

    // Move constructor
    Vector(Vector&& other) noexcept
        : _layout(std::move(other._layout))
    {
        PetscCallNoThrow(VecDestroy(&_vec)); // No-op if _vec is nullptr
        _vec = std::exchange(other._vec, nullptr);
    }

    // Move assignment
    Vector& operator=(Vector&& other) noexcept
    {
        if (this != &other) {
            _layout = std::move(other._layout);

            PetscCallNoThrow(VecDestroy(&_vec)); // No-op if _vec is nullptr
            _vec = std::exchange(other._vec, nullptr);
        }
        return *this;
    }

    ////////
    // Static methods
    ////////
    static Vector load(MPI_Comm comm, const std::string& name, const std::string& filename);

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
