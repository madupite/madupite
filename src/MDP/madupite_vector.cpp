#include "madupite_vector.h"
#include "madupite_errors.h"
#include "petscvec.h"
#include <numeric>

Vector::Vector(MPI_Comm comm, const std::string& name)
{
    PetscCallThrow(VecCreate(comm, &_vec));
    PetscCallThrow(PetscObjectSetName((PetscObject)_vec, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_vec, (name + '_').c_str()));
    PetscCallThrow(VecSetFromOptions(_vec));
}

Vector::Vector(MPI_Comm comm, const std::string& name, const std::vector<PetscScalar>& data)
    : Vector::Vector(comm, name)
{
    _layout = Layout(comm, data.size(), true);
    PetscCallThrow(VecSetLayout(Vec(_vec), _layout.petsc()));
    PetscCallThrow(VecSetFromOptions(_vec));
    std::vector<PetscInt> idx(data.size());
    std::iota(idx.begin(), idx.end(), _layout.start());
    PetscCallThrow(VecSetValues(_vec, data.size(), idx.data(), data.data(), INSERT_VALUES));
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

    PetscLayout pLayout;
    PetscCallThrow(VecGetLayout(x._vec, &pLayout));
    x._layout = Layout(pLayout);
    return x;
}

void Vector::write(const std::string& filename) const
{
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm(), filename.c_str(), FILE_MODE_WRITE, &viewer));
    PetscCallThrow(VecView(_vec, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}
