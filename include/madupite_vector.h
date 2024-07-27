#pragma once

#include <string>
#include <utility>
#include <vector>

#include <petscvec.h>

#include "madupite_errors.h"
#include "madupite_layout.h"

class Vector {
    Layout _layout;
    Vec    _vec = nullptr;

    Vector(MPI_Comm comm, const std::string& name);

public:
    // A helper class to manage the read lock (RAII principle)
    class VecReadOnly {
        // Inner PETSc vector
        const Vec& _vec;

    public:
        // Constructor - acquire the read lock
        VecReadOnly(const Vec& vec)
            : _vec(vec)
        {
            PetscCallThrow(VecLockReadPush(_vec));
        }

        // Destructor - release the read lock
        ~VecReadOnly() { PetscCallNoThrow(VecLockReadPop(_vec)); }

        // Disable copy semantics and move assignment
        VecReadOnly(const VecReadOnly&)             = delete;
        VecReadOnly& operator=(const VecReadOnly&)  = delete;
        VecReadOnly& operator=(VecReadOnly&& other) = delete;

        // Move constructor
        VecReadOnly(VecReadOnly&& other) noexcept
            : _vec(other._vec)
        {
        }

        // Accessor to get the const Vec&
        const Vec& get() const { return _vec; }

        // Implicit cast to const Vec&
        operator const Vec&() const { return _vec; }
    };

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
    Vec petsc() { return _vec; }

    // Get the read-only view of the inner PETSc vector.
    // Can be implicitly cast to `const Vec&` thanks to the `operator const Vec&() const`
    VecReadOnly petsc() const { return VecReadOnly(_vec); }

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
        PetscCallThrow(VecGetValues(_vec, 1, &index, &value));
        return value;
    }

    // single-value setter for floating point types
    void operator()(PetscInt index, PetscScalar value) { PetscCallThrow(VecSetValues(_vec, 1, &index, &value, INSERT_VALUES)); }

    ////////
    // Inline methods
    ////////

    // Assemble the vector
    void assemble()
    {
        PetscCallThrow(VecAssemblyBegin(_vec));
        PetscCallThrow(VecAssemblyEnd(_vec));
    }

    // copy this vector to a new vector
    Vector deepCopy() const
    {
        Vector copy;
        copy._layout = this->_layout;
        PetscCallThrow(VecDuplicate(this->_vec, &copy._vec));
        PetscCallThrow(VecCopy(this->_vec, copy._vec));
        return copy;
    }

    // copy into this vector from another vector without allocating a new PETSc vector; layouts must match
    void deepCopyFrom(const Vector& other)
    {
        if (other._layout != _layout) {
            throw MadupiteException("Vector layout does not match");
        }
        PetscCallThrow(VecCopy(other._vec, _vec));
    }

    ////////
    // Out-of-line method declarations
    ////////

    void write(const std::string& filename) const;
};
