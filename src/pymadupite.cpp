#include <iostream>
#include "mdp.h"
#include "petsc.h"
#include "utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

// Fix for OpenMPI 4
// (https://github.com/hpc4cmb/toast/issues/298)
struct ompi_communicator_t {};

NB_MODULE(_madupite_impl, m)
{
    nb::class_<Madupite>(m, "Madupite");

    m.def(
        "initialize_madupite",
        []( const std::vector<std::string>& in = {}) {
            int argc = in.size();
            if (argc > 1) {
                char** cStrings = new char*[in.size()];

                for (size_t i = 0; i < in.size(); ++i) {
                    cStrings[i] = const_cast<char*>(in[i].c_str());
                }
                PetscOptionsInsert(NULL, &argc, &cStrings, NULL);
                delete[] cStrings;
            } 
            return Madupite::initialize();
        }, "argv"_a = std::vector<std::string>{""},
        R"doc(
        Initialize the Madupite instance.

        This function initializes the Madupite library, setting up the necessary environment
        for solving Markov Decision Processes (MDPs). It returns an instance of the `Madupite` class.

        Returns
        -------
        Madupite
            An initialized instance of the `Madupite` class.
        )doc");

    m.def("getCommWorld", []() { return Madupite::pyGetCommWorld(); }, 
        R"doc(
        Get the global MPI communicator.

        This function returns the global MPI communicator.

        Returns
        -------
        int
            The global MPI communicator identifier.
        )doc");

    m.def("mpi_rank_size", &mpi_rank_size, nb::kw_only(), "comm"_a = 0,
        R"doc(
        Return the rank and size of the global MPI communicator.

        This function retrieves the rank (the unique identifier of the current process) and the size 
        (the total number of processes) within the global MPI communicator.

        Returns
        -------
        pair<int, int>
            A pair where the first element is the rank of the current process, and the second element
            is the total number of processes.
        )doc");

    //////////
    // Matrix
    //////////

    nb::enum_<MatrixType>(m, "MatrixType",
        R"doc(
        Enum class representing the type of a matrix in the Madupite framework.

        The `MatrixType` enum class defines the different types of matrices that can be used in the
        Madupite library. The matrix type determines how the matrix data is stored and accessed.

        Enum Values
        -----------
        Dense : MatrixType
            Represents a dense matrix where most of the elements are non-zero.
        Sparse : MatrixType
            Represents a sparse matrix where most of the elements are zero, allowing for more efficient
            storage and computation.
        )doc")
        .value("Dense", MatrixType::Dense)
        .value("Sparse", MatrixType::Sparse)
        .export_values();

    nb::enum_<MatrixCategory>(m, "MatrixCategory",
        R"doc(
        Enum class representing the category of a matrix in the Madupite framework.

        The `MatrixCategory` enum class defines the different categories of matrices that can be used in
        the Madupite library. Each category represents a specific type of data or role that the matrix
        plays within a Markov Decision Process (MDP).

        Enum Values
        -----------
        Dynamics : MatrixCategory
            Represents a matrix that models the dynamics of the system, i.e. transition probabilities
            in an MDP.
        Cost : MatrixCategory
            Represents a matrix that models the costs associated with different state-action pairs in an
            MDP.
        )doc")
        .value("Dynamics", MatrixCategory::Dynamics)
        .value("Cost", MatrixCategory::Cost)
        .export_values();

    nb::class_<MatrixPreallocation>(m, "MatrixPreallocation", 
        R"doc(
        Represents the preallocation for a matrix in the Madupite framework.

        The `MatrixPreallocation` class is used to define memory preallocation for matrices. Preallocating
        memory helps optimize matrix operations by reducing the need for dynamic memory allocation during
        computation. 

        The matrices are stored in blocks according to the number of ranks available. 
        E.g. a 6x6 matrix on 3 ranks is split into 2x2 blocks of size 3x3. 
        This class provides attributes to specify the number of non-zero entries in both diagonal and
        off-diagonal blocks of the matrix.

        Attributes
        ----------
        d_nz : int
            The number of non-zero entries per row on the diagonal block.
        d_nnz : list of int
            A list with the number of non-zero entries for each row on the diagonal block.
        o_nz : int
            The number of non-zero entries per row on the remaining blocks.
        o_nnz : list of int
            A list with the number of non-zero entries for each row of the remaining blocks.
        )doc")
        .def(nb::init<>())
        .def_rw("d_nz", &MatrixPreallocation::d_nz, "Number of diagonal non-zero entries")
        .def_rw("d_nnz", &MatrixPreallocation::d_nnz, "Diagonal non-zero entries")
        .def_rw("o_nz", &MatrixPreallocation::o_nz, "Number of off-diagonal non-zero entries")
        .def_rw("o_nnz", &MatrixPreallocation::o_nnz, "Off-diagonal non-zero entries");


    nb::class_<Matrix>(m, "Matrix",
        R"doc(
        Represents a matrix used within the Madupite framework.

        The `Matrix` class provides functionalities to handle matrices, including the ability to
        load a matrix from a file, convert matrix types to string, and write the matrix to a file.

        Methods
        -------
        __init__(self)
            Initializes an empty `Matrix` object.
            
        typeToString(cls, type: MatrixType) -> str
            Convert a matrix type enum to its corresponding string representation.

            Parameters
            ----------
            type : MatrixType
                The matrix type to convert to a string.

            Returns
            -------
            str
                A string representation of the matrix type.

        fromFile(cls, *, comm: int, name: str, filename: str, category: MatrixCategory, type: MatrixType) -> Matrix
            Load a matrix from a file.

            This static method creates a `Matrix` object by reading matrix data from a specified file.

            Parameters
            ----------
            comm : int, optional
                The MPI communicator identifier for parallel computation.
            name : str
                The name to associate with the loaded matrix within the Madupite environment.
            filename : str
                The path to the file from which the matrix will be loaded.
            category : MatrixCategory
                The category of the matrix (Dynamics, Cost).
            type : MatrixType
                The type of the matrix (Dense, Sparse).

            Returns
            -------
            Matrix
                A `Matrix` object containing the data loaded from the specified file.

        writeToFile(self, filename: str)
            Write the matrix data to a file.

            This method saves the current state of the matrix to a specified file, allowing for persistence
            and later retrieval.

            Parameters
            ----------
            filename : str
                The path to the file where the matrix will be saved.

        )doc")
        .def(nb::init<>())
        .def_static("typeToString", &Matrix::typeToString)
        .def_static("fromFile", &Matrix::fromFile<int>, nb::kw_only(),
            "comm"_a = 0, "name"_a = "", "filename"_a, "category"_a, "type"_a)
        .def("writeToFile", &Matrix::writeToFile, "filename"_a, "matrix_type"_a, "binary"_a = false, "overwrite"_a = false);


    m.def("createTransitionProbabilityTensor", 
        &createTransitionProbabilityTensor<int>, 
        nb::kw_only(), 
        "comm"_a = 0, 
        "name"_a = "",
        "numStates"_a, 
        "numActions"_a, 
        "func"_a, 
        "preallocation"_a = MatrixPreallocation {}, 
        R"doc(
        Creates a transition probability tensor for a Markov Decision Process (MDP).

        This function constructs a madupite.Matrix that represents the probabilities of transitioning 
        from one state to another, given a specific action.

        Parameters
        ----------
        comm : int, optional
            The communicator identifier for parallel computation.
        name : str
            The name of the transition probability tensor, used for identification within the MDP environment.
        numStates : int
            The total number of states in the Markov Decision Process.
        numActions : int
            The total number of possible actions in the Markov Decision Process.
        func : Callable[[int, int], Tuple[Sequence[float], Sequence[int]]]
            A callable function that takes in two arguments: a state index and an action index. 
            It returns a tuple containing:
            
            - A list of transition probabilities to other states.
            - A list of corresponding state indices for these probabilities.
        preallocation : madupite.MatrixPreallocation, optional
            A preallocation structure that defines how the matrix memory is allocated. 
            Defaults to an instance of MatrixPreallocation.

        Returns
        -------
        madupite.Matrix
            A matrix object representing the transition probability tensor.
        )doc");

    m.def("createStageCostMatrix", 
        &createStageCostMatrix<int>, 
        nb::kw_only(),
        "comm"_a = 0, 
        "name"_a = "", 
        "numStates"_a, 
        "numActions"_a, 
        "func"_a, 
        R"doc(
        Creates a stage cost matrix for a Markov Decision Process (MDP).

        This function constructs a matrix that represents the cost associated with each state-action pair in an MDP.

        Parameters
        ----------
        comm : int, optional
            The communicator identifier for parallel computation.
        name : str
            The name of the stage cost matrix, used for identification within the MDP environment.
        numStates : int
            The total number of states in the Markov Decision Process.
        numActions : int
            The total number of possible actions in the Markov Decision Process.
        func : Callable[[int, int], float]
            A callable function that takes in two arguments: a state index and an action index. 
            It returns a float representing the cost associated with that state-action pair.

        Returns
        -------
        madupite.madupite.Matrix
            A matrix object representing the stage cost matrix.
        )doc");

    //////////
    // MDP
    //////////

    nb::class_<MDP>(m, "MDP",
        R"doc(
        Represents a Markov Decision Process (MDP).

        This class provides methods to define and solve an MDP by setting the necessary components such as 
        the stage cost matrix and transition probability tensor.


        Methods
        -------
        __init__(self, madupite=madupite_initialize(), comm=0)
            Initialize the MDP with a Madupite instance and MPI communicator.

        clearOptions(self)
            Clear all options for the MDP.

        setOption(self, option, value=None)
            Set an option for the MDP.

            Parameters
            ----------
            option : str
                The name of the option to set.
            value : optional
                The value to assign to the option. If not provided, the option is set with its default value.

        setStageCostMatrix(self, arg)
            Set the stage cost matrix for the MDP.

            Parameters
            ----------
            arg : madupite.madupite.Matrix
                The matrix representing the stage costs for the MDP.

        setTransitionProbabilityTensor(self, arg)
            Set the transition probability tensor for the MDP.

            Parameters
            ----------
            arg : madupite.madupite.Matrix
                The tensor representing the transition probabilities for the MDP.

        setUp(self)
            Optional call to set up the MDP class internally.

        solve(self)
            Solve the MDP problem.

            This method computes the optimal policy and value function for the defined MDP.

            )doc")
        .def(nb::init<std::shared_ptr<Madupite>, int>(), "madupite"_a = Madupite::initialize(nullptr, nullptr),
            "comm"_a = 0, "Initialize MDP with Madupite instance and MPI communicator")
        .def("setOption", &MDP::setOption, "option"_a, "value"_a = nullptr, "Set options for MDP")
        .def("clearOptions", &MDP::clearOptions, "Clear all options for MDP")
        .def("setStageCostMatrix", &MDP::setStageCostMatrix, "Set the stage cost matrix")
        .def("setTransitionProbabilityTensor", &MDP::setTransitionProbabilityTensor, "Set the transition probability tensor")
        .def("setUp", &MDP::setUp, "Set up the MDP class")
        .def("solve", &MDP::solve, "Solve the MDP problem")
        .def(
            "__setitem__",
            [](MDP& self, const char* key, nb::handle value) {
                self.setOption(key, nb::str(value).c_str());
            });
}
