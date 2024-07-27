#pragma once

#include <string>
#include <utility>
#include <vector>

#include <petscvec.h>

#include "madupite_errors.h"
#include "madupite_layout.h"

using ConstVec = const _p_Vec*;

class Vector {
    Layout _layout;

    // Inner PETSc vector; should be accessed directly only in constructors and via petsc() otherwise
    Vec _vec = nullptr;

    Vector(MPI_Comm comm, const std::string& name);

public:
    ////////
    // Basic getters
    ////////

    // Get the MPI communicator
    MPI_Comm comm() const { return PetscObjectComm((PetscObject)petsc()); }

    // Get the name
    std::string name() const
    {
        const char* name;
        PetscCallThrow(PetscObjectGetName((PetscObject)petsc(), &name));
        return std::string(name);
    }

    // Get the inner PETSc vector
    Vec petsc()
    {
        if (_vec == nullptr) {
            throw MadupiteException("Vector is uninitialized");
        }
        return _vec;
    }

    // Get the inner PETSc vector in the const form.
    // We currently pass
    //   const_cast<Vec>(petsc())
    // to PETSc functions as they currently don't take ConstVec arguments (const _p_Vec *).
    // This could be a good proposal for PETSc.
    ConstVec petsc() const
    {
        // reuse the non-const implementation
        return const_cast<ConstVec>(const_cast<Vector*>(this)->petsc());
    }

    // Get the layout
    const Layout& layout() const { return _layout; }

    ////////
    // Constructors, destructors and assignment
    ////////

    // default constructor creates a 'null' vector
    Vector() = default;

    // Create a zero vector
    Vector(MPI_Comm comm, const std::string& name, const Layout& layout)
        : Vector(comm, name)
    {
        _layout = layout;
        PetscCallThrow(VecSetLayout(_vec, _layout.petsc()));
        PetscCallThrow(VecSetFromOptions(_vec));
        PetscCallThrow(VecZeroEntries(_vec));
    }

    // Create a vector from an std::vector
    Vector(MPI_Comm comm, const std::string& name, const std::vector<PetscScalar>& data);

    // Destructor
    ~Vector() { PetscCallNoThrow(VecDestroy(&_vec)); }

    // copy constructor (shallow)
    Vector(const Vector& other)
        : _layout(other._layout)
    {
        PetscCallThrow(VecDestroy(&_vec)); // No-op if _vec is nullptr
        _vec = other._vec;
        PetscCallThrow(PetscObjectReference((PetscObject)other._vec));
    }

    // copy assignment (shallow)
    Vector& operator=(const Vector& other)
    {
        if (this != &other) {
            _layout = other._layout;

            PetscCallThrow(VecDestroy(&_vec)); // No-op if _vec is nullptr
            _vec = other._vec;
            PetscCallThrow(PetscObjectReference((PetscObject)other._vec));
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

    // boolean conversion
    explicit operator bool() const { return _vec != nullptr; }

    // single-value getter for floating point types
    PetscScalar operator()(PetscInt index) const
    {
        PetscScalar value;
        PetscCallThrow(VecGetValues(const_cast<Vec>(petsc()), 1, &index, &value));
        return value;
    }

    // single-value setter for floating point types
    void operator()(PetscInt index, PetscScalar value) { PetscCallThrow(VecSetValues(petsc(), 1, &index, &value, INSERT_VALUES)); }

    ////////
    // Inline methods
    ////////

    // Assemble the vector
    void assemble()
    {
        PetscCallThrow(VecAssemblyBegin(petsc()));
        PetscCallThrow(VecAssemblyEnd(petsc()));
    }

    // copy this vector to a new vector
    Vector deepCopy() const
    {
        Vector copy;
        copy._layout = this->_layout;
        PetscCallThrow(VecDuplicate(const_cast<Vec>(petsc()), &copy._vec));
        PetscCallThrow(VecCopy(const_cast<Vec>(petsc()), copy._vec));
        return copy;
    }

    // copy into this vector from another vector without allocating a new PETSc vector; layouts must match
    void deepCopyFrom(const Vector& other)
    {
        if (other._layout != _layout) {
            throw MadupiteException("Vector layout does not match");
        }
        PetscCallThrow(VecCopy(const_cast<Vec>(other.petsc()), petsc()));
    }

    ////////
    // Out-of-line method declarations
    ////////

    void write(const std::string& filename) const;
};
