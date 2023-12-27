# distutils: language = c++
# cython: language_level = 3

# Import necessary Cython and Python libraries
import numpy as np
cimport numpy as cnp

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
        int inexactPolicyIteration() except +

# Cython wrapper for the MDP C++ class
cdef class PyMDP:
    cdef MDP *c_mdp

    def __cinit__(self):
        self.c_mdp = new MDP()

    def __dealloc__(self):
        if self.c_mdp is not NULL:
            del self.c_mdp

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

# Additional utility functions or classes can be added here as needed
