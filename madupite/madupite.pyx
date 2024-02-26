# distutils: language=c++
# cython: language_level=3

# Import necessary Cython and Python libraries
import numpy as np
cimport numpy as cnp
import inspect
from libcpp.string cimport string

def pystr2cppstr(pystr):
    cdef string cppstr = string(bytes(pystr, "utf-8"))
    return cppstr

from cpython.list cimport PyList_Check, PyList_Size, PyList_GetItem
from cpython.float cimport PyFloat_AsDouble
from cpython.long cimport PyLong_AsLong

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair


cdef py2cppstr (pystr):
    cdef string cppstr = string(bytes(pystr, "utf-8"))
    return cppstr

# Declaration of the C++ structs and classes from MDP.h
cdef extern from "MDP.h":
    cdef cppclass MDP:
        MDP() except +
        int setValuesFromOptions() except +
        int setOption(const char *option, const char *value, bint setValues) except +
        int inexactPolicyIteration() except +
        void loadFromBinaryFile()
        pair[int, int] getStateOwnershipRange()
        pair[int, int] getMDPSize()
        void fillRow(vector[int] &idxs, vector[double] &vals, int i, int matrix)
        void assembleMatrix(int matrix)
        int createCostMatrix()
        int createTransitionProbabilityTensor(int d_nz, const vector[int] &d_nnz, int o_nz, const vector[int] &o_nnz)
        int createTransitionProbabilityTensor()

# Cython wrapper for the MDP C++ class
cdef class PyMDP:
    cdef MDP *c_mdp
    _all_instances = []

    cdef fill_matrix_helper(self, list idxs, list vals, int row, int matrix):
        cdef vector[int] cidxs
        cdef vector[double] cvals
        cdef Py_ssize_t i

        if not PyList_Check(idxs) or not PyList_Check(vals):
            raise ValueError("idxs and vals must be lists")

        for i in range(len(idxs)):
            cidxs.push_back(PyLong_AsLong(idxs[i]))
        for i in range(len(vals)):
            cvals.push_back(PyFloat_AsDouble(vals[i]))

        self.c_mdp.fillRow(cidxs, cvals, row, matrix)
        # TODO: do we need to free memory here?


    def __cinit__(self):
        self.c_mdp = new MDP()
        self._all_instances.append(self)

    # def __dealloc__(self):
    #     if self.c_mdp is not NULL:
    #         del self.c_mdp

    # [] operator
    def __setitem__(self, key, value):
        cdef int result = self.c_mdp.setOption(py2cppstr(key), py2cppstr(value), True)
        if result:
            raise RuntimeError("setOption failed with error code %d" % result)

    def setValuesFromFile(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                key, value = line.split()
                self.c_mdp.setOption(py2cppstr(key), py2cppstr(value), False)
        self.c_mdp.setValuesFromOptions()

    def setValuesFromOptions(self):
        cdef int result
        result = self.c_mdp.setValuesFromOptions()
        if result != 0:
            raise RuntimeError("setValuesFromOptions failed with error code %d" % result)
        return result

    def setOption(self, option, value):
        cdef string cpp_option = py2cppstr(option)
        cdef string cpp_value = py2cppstr(value)
        cdef int result = self.c_mdp.setOption(cpp_option.c_str(), cpp_value.c_str(), False)
        if result:
            raise RuntimeError("setOption failed with error code %d" % result)
        return result

    def inexactPolicyIteration(self):
        cdef int result
        result = self.c_mdp.inexactPolicyIteration()
        if result != 0:
            raise RuntimeError("inexactPolicyIteration failed with error code %d" % result)
        return result

    def loadFromBinaryFile(self):
        self.c_mdp.loadFromBinaryFile()


    def createTP0(self, pre_alloc):
        cdef vector[int] emptyVec = vector[int]()
        self.c_mdp.createTransitionProbabilityTensor(pre_alloc[0], emptyVec, pre_alloc[1], emptyVec)

    def createTP1(self, pre_alloc):
        cdef vector[int] o_nnz = pre_alloc[1]
        cdef vector[int] emptyVec = vector[int]()
        self.c_mdp.createTransitionProbabilityTensor(pre_alloc[0], emptyVec, 0, o_nnz)

    def createTP2(self, pre_alloc):
        cdef vector[int] d_nnz = pre_alloc[0]
        cdef vector[int] emptyVec = vector[int]()
        self.c_mdp.createTransitionProbabilityTensor(0, d_nnz, pre_alloc[1], emptyVec)

    def createTP3(self, pre_alloc):
        cdef vector[int] d_nnz = pre_alloc[0]
        cdef vector[int] o_nnz = pre_alloc[1]
        self.c_mdp.createTransitionProbabilityTensor(0, d_nnz, 0, o_nnz)


    def createTransitionProbabilities(self, func, pre_alloc=None):
        print("CreateTransitionProbabilities start")
        # check function signature
        # params = inspect.signature(func).parameters
        # if len(params) != 2:
        #     raise ValueError("func should have exactly two parameters: state and action. Function signature: func(state, action) -> (next_states, probs), where next_states contains indices of the next states and probs (correspondig transition probabilities) are lists")

        # user's choice to pre-allocate memory
        # pre_alloc = [d_nz | d_nnz, o_nz | o_nnz] 
        """
        5 different modes TODO implement
        -1 - no preallocation
        0 - use d_nz and o_nz
        1 - use d_nz and o_nnz
        2 - use d_nnz and o_nz
        3 - use d_nnz and o_nnz
        """

        """
        pre_alloc: None => mode -1
        pre_alloc[0]: int d_nz or list d_nnz => set LSB to 1 if list
        pre_alloc[1]: int o_nz or list o_nnz => set Bit nr. 2 to 1 if list
        """


        mode = 0
        if pre_alloc is None:
            print("no preallocation")
            self.c_mdp.createTransitionProbabilityTensor()

        else:
            if len(pre_alloc) != 2:
                raise ValueError("pre_alloc should be a tuple [d_nz | [d_nnz], o_nz | [o_nnz]] where d_nnz and o_nnz are lists or None if no preallocation is desired (not recommended)")
            # create bit pair set according to pre_alloc
            if isinstance(pre_alloc[0], list):
                mode |= 1 # set LSB to 1
            if isinstance(pre_alloc[1], list):
                mode |= 2 # set Bit nr. 2 to 1


        if pre_alloc is not None and mode == 0:
            self.createTP0(pre_alloc)
        elif pre_alloc is not None and mode == 1:
            self.createTP1(pre_alloc)
        elif pre_alloc is not None and mode == 2:
            self.createTP2(pre_alloc)
        elif pre_alloc is not None and mode == 3:
            self.createTP3(pre_alloc)
        else:
            self.c_mdp.createTransitionProbabilityTensor()
        print(f"{mode=}")

        cdef pair[int, int] indices = self.c_mdp.getStateOwnershipRange()
        print(f"finished getStateOwnershipRange, {indices.first=}, {indices.second=}")
        cdef pair[int, int] size = self.c_mdp.getMDPSize() # (numStates, numActions)
        print(f"finished getMDPSize, {size.first=}, {size.second=}")
        print(f"start filling values - {indices.first=}, {indices.second=}")

        for i in range(indices.first  * size.second, indices.second * size.second):
            idxs, vals = func(i // size.second, i % size.second)
            self.fill_matrix_helper(idxs, vals, i, 0)
        print("finished filling values")
        self.c_mdp.assembleMatrix(0)
        print("CreateTransitionProbabilities end")


    def createStageCosts(self, func):
        print("CreateStageCosts start")
        # TODO: check if func has the correct signature
        params = inspect.signature(func).parameters
        if len(params) != 2:
            raise ValueError("func should have exactly two parameters: state and action. Function signature: func(state, action) -> cost (double)")
        
        self.c_mdp.createCostMatrix()
        print("created cost matrix")   

        cdef pair[int, int] indices = self.c_mdp.getStateOwnershipRange()
        cdef pair[int, int] size = self.c_mdp.getMDPSize() # (numStates, numActions)
        print(f"{size.first=}, {size.second=}")
        print(f"{indices.first=}, {indices.second=}")
        

        actionIndices = np.arange(size.second, dtype=np.int32)
        for stateInd in range(indices.first, indices.second):
            vals = np.array([func(stateInd, action) for action in actionIndices])
            self.fill_matrix_helper(actionIndices.tolist(), vals.tolist(), stateInd, 1)

        print("filled cost matrix")
        self.c_mdp.assembleMatrix(1)
        print("CreateStageCosts end")

    @classmethod
    def _get_all_instances(cls):
        """helper function for garbage collection in PETScContextManager. This function returns all PyMdp instances, s.t. the context manager can explicitly delete them before calling PetscFinalize.

        Returns
        -------
        list
            list of all PyMdp instances
        """
        return cls._all_instances



# Additional utility functions or classes can be added here as needed

### new 
cdef extern from "petsc.h":
    cdef int PetscInitialize(int *argc, char ***argv, const char *file, const char *help)


def _initialize_petsc():
    """Wrapper for calling PetscInitialize()
    """
    cdef int argc = 0
    cdef char **argv = NULL

    cdef int ierr = PetscInitialize(&argc, &argv, NULL, NULL)

    if ierr != 0:
        raise RuntimeError("PetscInitialize failed with error code: {}".format(ierr))


cdef extern from "petsc.h":
    cdef int PetscFinalize()


def _finalize_petsc():
    """Wrapper for calling PetscFinalize()
    """
    PetscFinalize()


class PETScContextManager:
    """Use this context manager to handle initializing and finalizing the PETSc/MPI execution environment.

    .. code-block:: python

        with madupite.PETScContextManager():
            madupite.PyMdp()
            ...

    """
    def __enter__(self):
        _initialize_petsc()

    def __exit__(self, exc_type, exc_value, exc_tb):
        # explicitly delete all PETSc objects before MPI is finalized
        for obj in PyMDP._get_all_instances():
            del obj

        _finalize_petsc()

###
cdef extern from "../src/utils.cpp":
    cdef int rankPETSCWORLD()


def MPI_master():
    """This helper function allows calling functions only by the master rank, e.g. if you want to write to file by only one process. This function must be called within the PETScContextManager.

    Returns
    -------
    bool
        True if this is rank 0 on PETSC_COMM_WORLD
    """
    return rankPETSCWORLD() == 0


def generateArtificialMDP(nstates, mactions, transition_rate, seed=None):
    """Generate an artificial MDP, whose nonzero transition probabilities are sampled from a uniform distribution. Returns the transition probability in the correct format for PyMdp.loadP().

    Parameters
    ----------
    nstates : int
        Number of states.
    mactions : int
        Number of actions
    transition_rate : float in (0,1]
        average rate of transitions to other states. 0.3 means that on average a state has a nonzero probability of transitioning to 30% of the other states.
    seed : int, optional
        random seed, by default None
    Returns
    -------
        Tuple
            (transition probability matrix, stage cost matrix)
    """
    import numpy as np
    from scipy.sparse import csr_array

    rng = np.random.default_rng(seed=seed)
    if transition_rate == 1:
        transprobmat = rng.random((nstates * mactions, nstates))
        norms = transprobmat.sum(axis=1)
        transprobmat /= norms[:, np.newaxis]
    else:
        rowptr = [0]
        indices = []
        data = []
        for state in range(nstates):
            for action in range(mactions):
                idx = rng.choice(range(nstates), rng.binomial(nstates, transition_rate), replace=False).tolist()
                if len(idx) == 0:
                    idx = rng.integers(low=0, high=nstates, size=1).tolist()
                else:
                    idx.sort()
                rowptr+=[rowptr[-1] + len(idx)]
                indices+=idx
                data+=rng.random(len(idx)).tolist()
        transprobmat = csr_array(
            (data, indices, rowptr), shape=(nstates * mactions, nstates)
        )
        # normalize
        norms = transprobmat.sum(axis=1)
        transprobmat /= norms[:, np.newaxis]

    # create stage cost matrix
    stagecostmat = rng.random((nstates, mactions))
    return transprobmat, stagecostmat


def writePETScBinary(matrix, filename):
    """Write numpy/scipy matrix as petsc binary sparse format to file
    https://petsc.org/release/manualpages/Mat/MatLoad/#notes

    Parameters
    ----------
    matrix : numpy/scipy matrix
        any matrix type that allows calling scipy.sparse.csr_array(matrix)
    filename : string
        output filename
    """
    import numpy as np
    from scipy.sparse import csr_array
    csr_matrix = csr_array(matrix)
    csr_matrix.sort_indices()
    with open(filename, "wb") as f:
        f.write(b"\x00\x12\x7b\x50")  # class id, sort of a magic number
        f.write(np.array(matrix.shape, dtype=">i4").tobytes())  # rows and cols
        f.write(np.array(csr_matrix.count_nonzero(), dtype=">i4").tobytes())  # nnz
        f.write(
            np.array(np.diff(csr_matrix.indptr), dtype=">i4").tobytes()
        )  # row pointer
        f.write(np.array((csr_matrix.indices), dtype=">i4").tobytes())  # column indices
        f.write(np.array(csr_matrix.data, dtype=">f8").tobytes())  # values
    with open(filename + ".info", "wb") as f:  # avoid petsc complaints
        pass


def generateMDP(nstates, mactions, probfunction, costfunction, transprobfilename, stagecostfilename, verbose=False):
    """ Generate the transition probability tensor and stage cost matrix from a probability function and a stagecost function. Output in a PetscBinary file which can be read by the solver.
    This function must be wrapped in a if madupite.MPI_master() statement.
    Note: defining probfunction in a numba compatible way provides a significant speedup.

    Parameters
    ----------
    nstates : int
        Number of states
    mactions : int
        Number of actions
    probfunction : func
        A function that returns the transition probability to every other state given a state-action pair: :math:`\\mathcal{P}(s\\vert s,a)`. Since this is often sparse probfunction(state,action) should return a tuple which consists of an array of the indices of nonzero entries and an array of the values of the nonzero entries.
    costfunction : func
        A function that returns the stage cost for a state-action pair. costfunction(state, action) returns a scalar
    transprobfilename : str
        output filename for the transition probability file
    stagecostfilename : str
        output filename for the stage cost file
    verbose : bool, optional
        print potential numba compile error, by default False

    """
    import numpy as np

    def writeSetup(rows, cols, filename):
        with open(filename, "wb") as f:
            f.write(b"\x00\x12\x7b\x50")  # class id, sort of a magic number
            f.write(np.array([rows, cols], dtype=">i4").tobytes())  # rows and cols
            f.write(np.array(0, dtype=">i4").tobytes())  # nnz dummy

    # if numba is available and the provided functions can be compiled, they
    # are compiled to speed up execution
    try:
        from numba import njit
        numba_avail = True
    except Exception as err:
        numba_avail = False
        print("numba not available")
    if numba_avail:
        k = probfunction
        a = costfunction
        try:
            probfunction=njit(probfunction)
            probfunction(0, 0)  # dummy call to trigger numba compile
        except Exception as err:
            if verbose:
                print(err)
                print("Transition probability function could not be compiled by numba. Generating MDP continues slower.")
            probfunction = k
            pass
        try:
            costfunction=njit(costfunction)
            costfunction(0, 0)  # dummy call to trigger numba compile
        except Exception as err:
            if verbose:
                print(err)
                print("Stage cost function could not be compiled by numba. Generating MDP continues slower.")
            costfunction = a
            pass

    # erase file and write first three ints
    writeSetup(nstates * mactions, nstates, transprobfilename)
    # write row pointers
    with open(transprobfilename, "r+b") as f:
        with open(transprobfilename+".temp", "wb") as ftemp:
            f.seek(16)
            nnz=0
            for i in range(nstates):
                for j in range(mactions):
                    idxs, vals = probfunction(i, j)
                    f.write(np.array(len(idxs), dtype=">i4").tobytes())
                    f.seek(16 + nstates * mactions * 4 + nnz * 4)
                    f.write(np.array(idxs, dtype=">i4").tobytes())
                    nnz += len(idxs)
                    f.seek(16+(i*mactions+j+1)*4)
                    ftemp.write(np.array(vals, dtype=">f8").tobytes())
            f.seek(12)
            f.write(np.array(nnz, dtype=">i4").tobytes())
            f.seek(16 + nstates * mactions * 4 + nnz * 4)
            ftemp.seek(0)
        with open(transprobfilename+".temp", "r+b") as ftemp:
            chunk_size=int(1e9)  # reading 1GB is reasonable
            while True:
                chunk = ftemp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        import os
        os.remove(transprobfilename+".temp")
    # erase file and write first three ints
    writeSetup(nstates, mactions, stagecostfilename)
    with open(stagecostfilename, "r+b") as f:
        f.seek(12)
        # nnz
        f.write(
            np.array(nstates * mactions, dtype=">i4").tobytes()
        )
        # rowptr
        f.write(
            np.full(nstates, mactions, dtype=">i4").tobytes()
        )

        f.write(
            np.broadcast_to(
                np.arange(mactions, dtype=">i4"), (nstates, mactions)
            ).tobytes()
        )
        for i in range(nstates):
            for j in range(mactions):
                f.write(np.array(costfunction(i, j), dtype=">f8").tobytes())
    if verbose:
        print("Generated MDP with "+str(nstates)+" states and "+str(mactions)+" actions.")
