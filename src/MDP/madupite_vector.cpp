#include "madupite_vector.h"
#include "madupite_errors.h"
#include "petscvec.h"

Vector::Vector(MPI_Comm comm, const std::string& name)
{
    PetscCallThrow(VecCreate(comm, &_vec));
    PetscCallThrow(PetscObjectSetName((PetscObject)_vec, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_vec, (name + '_').c_str()));
    PetscCallThrow(VecSetFromOptions(_vec));
}

Vector::Vector(MPI_Comm comm, const std::string& name, PetscInt size, bool local)
    : Vector::Vector(comm, name)
{
    if (local) {
        PetscCallThrow(VecSetSizes(_vec, size, PETSC_DECIDE));
    } else {
        PetscCallThrow(VecSetSizes(_vec, PETSC_DECIDE, size));
    }
    PetscCallThrow(VecZeroEntries(_vec));
}

Vector::Vector(MPI_Comm comm, const std::string& name, const std::vector<PetscScalar>& data)
    : Vector::Vector(comm, name)
{
    PetscCallThrow(VecSetSizes(_vec, data.size(), PETSC_DECIDE));
    PetscCallThrow(VecSetValues(_vec, data.size(), nullptr, data.data(), INSERT_VALUES));
    assemble();
    PetscCallThrow(VecSetOptionsPrefix(_vec, name.c_str()));
}

Vector Vector::load(MPI_Comm comm, const std::string& name, const std::string& filename)
{
    Vector      x(comm, name);
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(VecLoad(x._vec, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    return x;
}

void Vector::write(const std::string& filename) const
{
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm(), filename.c_str(), FILE_MODE_WRITE, &viewer));
    PetscCallThrow(VecView(_vec, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}
