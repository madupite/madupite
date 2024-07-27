#pragma once

#include "petscis.h"

#include "madupite_errors.h"

class Layout {
    PetscLayout layout = nullptr;

public:
    ////////
    // Constructors, destructors and assignment
    ////////

    // default constructor creates a 'null' layout
    Layout() = default;

    Layout(MPI_Comm comm, PetscInt N, bool local = false)
    {
        PetscCallThrow(PetscLayoutCreate(comm, &layout));
        if (local) {
            PetscCallThrow(PetscLayoutSetLocalSize(layout, N));
        } else {
            PetscCallThrow(PetscLayoutSetSize(layout, N));
        }
    }

    // create from an existing PetscLayout
    // We might consider removing explicit once the usage is fully clear
    explicit Layout(PetscLayout petscLayout) { PetscCallThrow(PetscLayoutReference(petscLayout, &layout)); }

    // copy constructor (shallow)
    Layout(const Layout& other)
    {
        // If layout already exists it is destroyed first
        PetscCallThrow(PetscLayoutReference(other.layout, &layout));
    }

    // copy assignment (shallow)
    Layout& operator=(const Layout& other)
    {
        if (this != &other) {
            PetscCallThrow(PetscLayoutReference(other.layout, &layout));
        }
        return *this;
    }

    // move constructor
    Layout(Layout&& other) noexcept
    {
        PetscCallNoThrow(PetscLayoutDestroy(&layout));
        layout       = other.layout;
        other.layout = nullptr;
    }

    // move assignment
    Layout& operator=(Layout&& other) noexcept
    {
        if (this != &other) {
            PetscCallNoThrow(PetscLayoutDestroy(&layout));
            layout       = other.layout;
            other.layout = nullptr;
        }
        return *this;
    }

    // destructor
    ~Layout()
    {
        // no-op if layout is nullptr
        PetscCallNoThrow(PetscLayoutDestroy(&layout));
    }

    ////////
    // Basic getters
    ////////

    // PetscLayout is locked from changes once PetscLayoutSetUp has been called on it
    const PetscLayout petsc() const
    {
        PetscCallThrow(PetscLayoutSetUp(layout));
        return layout;
    }

    // Get the MPI communicator
    MPI_Comm comm() const { return layout->comm; }

    PetscInt localSize() const { return petsc()->n; }

    PetscInt size() const { return petsc()->N; }

    PetscInt start() const { return petsc()->rstart; }

    PetscInt end() const { return petsc()->rend; }

    ////////
    // Operators
    ////////

    // comparison operator
    bool operator==(const Layout& other) const
    {
        PetscBool result;
        PetscCallThrow(PetscLayoutCompare(layout, other.layout, &result));
        return result;
    }

    // boolean conversion
    explicit operator bool() const { return layout != nullptr; }
};
