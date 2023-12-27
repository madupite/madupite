# distutils: language = c++
# distutils: sources = MDP_algorithm.cpp, MDP_setup.cpp, MDP.h

cdef extern from "MDP.h":
    cdef cppclass MDP:
        MDP() except +
        int setValuesFromOptions() except +
        int inexactPolicyIteration() except +