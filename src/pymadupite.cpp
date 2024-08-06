#include "MDP.h"
#include "MDP_matrix.h"
#include "madupite_matrix.h"
#include "madupite_vector.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(madupite, m)
{
    nb::class_<Madupite>(m, "Madupite");

    m.def("initialize_madupite", []() { return Madupite::initialize(nullptr, nullptr); }, R"pbdoc(Initialize Madupite instance)pbdoc");

    m.def("getCommWorld", []() { return Madupite::getCommWorld(); }, "Get global communicator for MPI");

    m.def("mpi_rank_size", []() {
        int rank, size;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        return std::make_pair(rank, size);
    });

    //////////
    // Matrix
    //////////

    nb::enum_<MatrixType>(m, "MatrixType").value("Dense", MatrixType::Dense).value("Sparse", MatrixType::Sparse).export_values();

    nb::enum_<MatrixCategory>(m, "MatrixCategory").value("Dynamics", MatrixCategory::Dynamics).value("Cost", MatrixCategory::Cost).export_values();

    nb::class_<MatrixPreallocation>(m, "MatrixPreallocation")
        .def(nb::init<>())
        .def_rw("d_nz", &MatrixPreallocation::d_nz, "Number of diagonal non-zero entries")
        .def_rw("d_nnz", &MatrixPreallocation::d_nnz, "Diagonal non-zero entries")
        .def_rw("o_nz", &MatrixPreallocation::o_nz, "Number of off-diagonal non-zero entries")
        .def_rw("o_nnz", &MatrixPreallocation::o_nnz, "Off-diagonal non-zero entries");

    nb::class_<Matrix>(m, "Matrix")
        .def(nb::init<>())
        .def_static("typeToString", &Matrix::typeToString, "Convert matrix type to string")
        .def_static("fromFile", &Matrix::fromFile, nb::kw_only(), "comm"_a = Madupite::getCommWorld(), "name"_a, "filename"_a, "category"_a, "type"_a,
            "Load matrix from file")
        .def("writeToFile", &Matrix::writeToFile, "Write matrix to file");

    m.def("createTransitionProbabilityTensor", &createTransitionProbabilityTensor, nb::kw_only(), "comm"_a = Madupite::getCommWorld(), "name"_a,
        "numStates"_a, "numActions"_a, "func"_a, "preallocation"_a = MatrixPreallocation {}, "Create transition probability tensor");

    m.def("createStageCostMatrix", &createStageCostMatrix, nb::kw_only(), "comm"_a = Madupite::getCommWorld(), "name"_a, "numStates"_a,
        "numActions"_a, "func"_a, "Create stage cost matrix");

    //////////
    // MDP
    //////////

    nb::class_<MDP>(m, "MDP")
        .def(nb::init<std::shared_ptr<Madupite>, MPI_Comm>(), "madupite"_a, "comm"_a = Madupite::getCommWorld(),
            "Initialize MDP with Madupite instance and MPI communicator")
        .def("setOption", &MDP::setOption, "option"_a, "value"_a = nullptr, "Set options for MDP")
        .def("clearOptions", &MDP::clearOptions, "Clear all options for MDP")
        .def("setStageCostMatrix", &MDP::setStageCostMatrix, "Set the stage cost matrix")
        .def("setTransitionProbabilityTensor", &MDP::setTransitionProbabilityTensor, "Set the transition probability tensor")
        .def("setUp", &MDP::setUp, "Set up the MDP problem")
        .def("solve", &MDP::solve, "Solve the MDP problem");
}
