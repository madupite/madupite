#include "madupite_vector.h"
#include "madupite_errors.h"

Vector::Vector(MPI_Comm comm, const std::string& name, PetscInt size)
{
    PetscCallThrow(VecCreate(comm, &_vec));
    PetscCallThrow(PetscObjectSetName((PetscObject)_vec, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_vec, (name + '_').c_str()));
    PetscCallThrow(VecSetFromOptions(_vec));
    PetscCallThrow(VecZeroEntries(_vec));
}

Vector::Vector(MPI_Comm comm, const std::string& name, const std::string& filename)
{
    PetscCallThrow(VecCreate(comm, &_vec));
    PetscCallThrow(PetscObjectSetName((PetscObject)_vec, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_vec, (name + '_').c_str()));
    PetscCallThrow(VecSetFromOptions(_vec));
    {
        PetscViewer viewer;
        PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
        PetscCallThrow(VecLoad(_vec, viewer));
        PetscCallThrow(PetscViewerDestroy(&viewer));
    }
}

Vector::Vector(MPI_Comm comm, const std::string& name, const std::vector<PetscScalar>& data)
{
    PetscCallThrow(VecCreate(comm, &_vec));
    PetscCallThrow(VecSetFromOptions(_vec));
    PetscCallThrow(VecSetSizes(_vec, data.size(), PETSC_DECIDE));
    PetscCallThrow(VecSetValues(_vec, data.size(), nullptr, data.data(), INSERT_VALUES));
    assemble();
    PetscCallThrow(VecSetOptionsPrefix(_vec, name.c_str()));
}

void Vector::write(const std::string& filename) const
{
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm(), filename.c_str(), FILE_MODE_WRITE, &viewer));
    PetscCallThrow(VecView(_vec, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}
