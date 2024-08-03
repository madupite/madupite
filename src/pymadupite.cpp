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
    // Madupite
    nb::class_<Madupite>(m, "Madupite");
    m.def("initialize_madupite", []() { return Madupite::initialize(nullptr, nullptr); }, "Initialize Madupite instance");
    m.def("getCommWorld", []() { return Madupite::getCommWorld(); }, "Get global communicator for MPI");
    // nb::class_<MPI_Comm>(m, "MPI_Comm");
    // m.def("COMM_WORLD", []() { return PETSC_COMM_WORLD; }, "Global communicator for MPI");
    // m.def("petsc_initialize")

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
        .def_rw("d_nz", &MatrixPreallocation::d_nz)
        .def_rw("d_nnz", &MatrixPreallocation::d_nnz)
        .def_rw("o_nz", &MatrixPreallocation::o_nz)
        .def_rw("o_nnz", &MatrixPreallocation::o_nnz);

    nb::class_<Matrix>(m, "Matrix")
        .def(nb::init<>())
        .def_static("typeToString", &Matrix::typeToString)
        .def_static("fromFile", &Matrix::fromFile, nb::kw_only(), "comm"_a = Madupite::getCommWorld(), "name"_a, "filename"_a, "category"_a, "type"_a)
        .def("writeToFile", &Matrix::writeToFile);

    m.def("createTransitionProbabilityTensor", &createTransitionProbabilityTensor, nb::kw_only(), "comm"_a = Madupite::getCommWorld(), "name"_a,
        "numStates"_a, "numActions"_a, "func"_a, "preallocation"_a                                         = MatrixPreallocation {});
    m.def("createStageCostMatrix", &createStageCostMatrix, nb::kw_only(), "comm"_a = Madupite::getCommWorld(), "name"_a, "numStates"_a,
        "numActions"_a, "func"_a);

    //////////
    // MDP
    //////////
    nb::class_<MDP>(m, "MDP")
        .def(nb::init<std::shared_ptr<Madupite>, MPI_Comm>(), nb::arg("madupite"), nb::arg("comm") = Madupite::getCommWorld())
        .def("setOption", &MDP::setOption, nb::arg("option"), nb::arg("value") = nullptr)
        .def("clearOptions", &MDP::clearOptions)
        .def("setStageCostMatrix", &MDP::setStageCostMatrix)
        .def("setTransitionProbabilityTensor", &MDP::setTransitionProbabilityTensor)
        .def("setUp", &MDP::setUp)
        .def("solve", &MDP::solve);
}
