#pragma once

#include "petscis.h"

#include "madupite_errors.h"

class Layout {
    MPI_Comm    comm   = MPI_COMM_NULL;
    PetscInt    N      = PETSC_DECIDE;
    bool        local  = false;
    PetscLayout layout = nullptr;

public:
    ////////
    // Constructors, destructors and assignment
    ////////

    // use default constructor only to represent a 'null' layout
    Layout() = default;

    Layout(MPI_Comm comm, PetscInt N, bool local = false)
        : comm(comm)
        , N(N)
        , local(local)
    {
        PetscCallThrow(PetscLayoutCreate(comm, &layout));
        if (local) {
            PetscCallThrow(PetscLayoutSetLocalSize(layout, N));
        } else {
            PetscCallThrow(PetscLayoutSetSize(layout, N));
        }
    }

    // create from an existing PetscLayout
    Layout(PetscLayout petscLayout)
        : comm(petscLayout->comm)
        , N(petscLayout->N)
    {
        PetscCallThrow(PetscLayoutReference(petscLayout, &layout));
    }

    // copy constructor (shallow)
    Layout(const Layout& other)
        : comm(other.comm)
        , N(other.N)
        , local(other.local)
    {
        // If layout already exists it is destroyed first
        PetscCallThrow(PetscLayoutReference(other.layout, &layout));
    }

    // copy assignment (shallow)
    Layout& operator=(const Layout& other)
    {
        if (this != &other) {
            comm  = other.comm;
            N     = other.N;
            local = other.local;
            PetscCallThrow(PetscLayoutReference(other.layout, &layout));
        }
        return *this;
    }

    // move constructor
    Layout(Layout&& other) noexcept
        : comm(other.comm)
        , N(other.N)
        , local(other.local)
    {
        PetscCallNoThrow(PetscLayoutDestroy(&layout));
        layout       = other.layout;
        other.layout = nullptr;
    }

    // move assignment
    Layout& operator=(Layout&& other) noexcept
    {
        if (this != &other) {
            comm  = other.comm;
            N     = other.N;
            local = other.local;

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
};
