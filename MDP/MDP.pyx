# distutils: language = c++
# cython: language_level = 3

# Import necessary Cython and Python libraries
import numpy as np
cimport numpy as cnp
from libcpp.string cimport string

def pystr2cppstr(pystr):
    cdef string cppstr = string(bytes(pystr, "utf-8"))
    return cppstr

# Include the necessary C++ headers for PETSc
cdef extern from "<petscvec.h>":
    pass  # Include necessary PETSc declarations here

cdef extern from "<petscmat.h>":
    pass  # Include necessary PETSc declarations here

cdef extern from "<petscksp.h>":
    pass  # Include necessary PETSc declarations here

# Declaration of the C++ structs and classes from MDP.h
cdef extern from "MDP.h":
    cdef cppclass MDP:
        MDP() except +
        int setValuesFromOptions() except +
        int setOption(const char *option, const char *value) except +
        int inexactPolicyIteration() except +

# Cython wrapper for the MDP C++ class
cdef class PyMDP:
    cdef MDP c_mdp
    _all_instances = []

    def __cinit__(self):
        self.c_mdp = MDP()
        self._all_instances.append(self)

    # def __dealloc__(self):
    #     if self.c_mdp is not NULL:
    #         del self.c_mdp

    # Wrapper method for setValuesFromOptions
    def setValuesFromOptions(self):
        cdef int result
        result = self.c_mdp.setValuesFromOptions()
        if result != 0:
            raise RuntimeError("setValuesFromOptions failed with error code %d" % result)
        return result

    # Wrapper method for inexactPolicyIteration
    def inexactPolicyIteration(self):
        cdef int result
        result = self.c_mdp.inexactPolicyIteration()
        if result != 0:
            raise RuntimeError("inexactPolicyIteration failed with error code %d" % result)
        return result

    def setOption(self, option, value):
        cdef int result
        cdef string cpp_option = pystr2cppstr(option)
        cdef string cpp_value = pystr2cppstr(value)
        result = self.c_mdp.setOption(cpp_option.c_str(), cpp_value.c_str())
        if result != 0:
            raise RuntimeError("setOption failed with error code %d" % result)
        return result

    @classmethod
    def _get_all_instances(cls):
        return cls._all_instances

cdef extern from "petsc.h":
    cdef int PetscInitialize(int *argc, char ***args, const char *file, const char *help) 

def _initialize_petsc():
    cdef int argc = 0
    cdef char **argv = NULL
    cdef int ierr = PetscInitialize(&argc, &argv, NULL, NULL)
    if ierr != 0:
        raise RuntimeError("PetscInitialize failed with error code %d" % ierr)

cdef extern from "petsc.h":
    cdef int PetscFinalize()

def _finalize_petsc():
    PetscFinalize()


class PETScContextManager:
    def __enter__(self):
        _initialize_petsc()

    def __exit__(self, exc_type, exc_value, traceback):
        for obj in PyMDP._get_all_instances():
            del obj

        _finalize_petsc()


cdef extern from "mpi.h":
    cdef int MPI_Comm_rank(int comm, int *rank)
    cdef int MPI_Comm_size(int comm, int *size)

# def _get_mpi_rank():
#     cdef int rank
#     MPI_Comm_rank(, &rank)
#     return rank

