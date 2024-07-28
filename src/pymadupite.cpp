#include "MDP.h"
#include "MDP_matrix.h"
#include "madupite_matrix.h"
#include "madupite_vector.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_madupite_impl, m)
{
    // Madupite
    nb::class_<Madupite>(m, "Madupite");
    m.def("initialize_madupite", []() { return Madupite::initialize(nullptr, nullptr); }, "Initialize Madupite instance");
    // nb::class_<MPI_Comm>(m, "MPI_Comm");
    m.def("COMM_WORLD", []() { return PETSC_COMM_WORLD; }, "Global communicator for MPI");


    //////////
    // Matrix
    //////////
    nb::enum_<MatrixType>(m, "MatrixType")
        .value("Dense", MatrixType::Dense)
        .value("Sparse", MatrixType::Sparse)
        .export_values();

    nb::enum_<MatrixCategory>(m, "MatrixCategory")
        .value("Dynamics", MatrixCategory::Dynamics)
        .value("Cost", MatrixCategory::Cost)
        .export_values();

    nb::class_<MatrixPreallocation>(m, "MatrixPreallocation")
        .def(nb::init<>())
        .def_rw("d_nz", &MatrixPreallocation::d_nz)
        .def_rw("d_nnz", &MatrixPreallocation::d_nnz)
        .def_rw("o_nz", &MatrixPreallocation::o_nz)
        .def_rw("o_nnz", &MatrixPreallocation::o_nnz);

    nb::class_<Matrix>(m, "Matrix")
        .def(nb::init<>())
        .def_static("typeToString", &Matrix::typeToString)
        .def_static("fromFile", &Matrix::fromFile)
        .def("writeToFile", &Matrix::writeToFile);

    m.def("createTransitionProbabilityTensor", &createTransitionProbabilityTensor);
    m.def("createStageCostMatrix", &createStageCostMatrix);

    //////////
    // MDP 
    //////////
    nb::class_<MDP>(m, "MDP")
        .def(nb::init<std::shared_ptr<Madupite>, MPI_Comm>(), 
             nb::arg("madupite"), nb::arg("comm") = PETSC_COMM_WORLD)
        .def("setOption", &MDP::setOption,
             nb::arg("option"), nb::arg("value") = nullptr)
        .def("clearOptions", &MDP::clearOptions)
        .def("setStageCostMatrix", &MDP::setStageCostMatrix)
        .def("setTransitionProbabilityTensor", &MDP::setTransitionProbabilityTensor)
        .def("setUp", &MDP::setUp)
        .def("solve", &MDP::solve);
}
