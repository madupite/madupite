#include "MDP.h"
#include "MDP_matrix.h"
#include "madupite_matrix.h"
#include "madupite_vector.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_madupite, m)
{
    m.doc() = "Python bindings for madupite";

    // Bind the Madupite class
    nb::class_<Madupite>(m, "Madupite")
        .def_static("initialize",
            [](nb::list args) {
                std::vector<char*> cargs;
                cargs.reserve(args.size());
                for (const auto& arg : args) {
                    cargs.push_back(const_cast<char*>(nb::cast<std::string>(arg).c_str()));
                }
                int    argc = cargs.size();
                char** argv = cargs.data();
                return Madupite::initialize(&argc, &argv);
            })
        .def_static("get", &Madupite::get);

    // Bind the MDP class
    nb::class_<MDP>(m, "MDP")
        .def(nb::init<std::shared_ptr<Madupite>>())
        .def("setOption", &MDP::setOption)
        .def("clearOptions", &MDP::clearOptions)
        .def("setStageCostMatrix", &MDP::setStageCostMatrix)
        .def("setTransitionProbabilityTensor", &MDP::setTransitionProbabilityTensor)
        .def("setUp", &MDP::setUp)
        .def("solve", &MDP::solve);

    // Bind the Matrix class
    nb::class_<Matrix>(m, "Matrix").def_static("fromFile", &Matrix::fromFile).def("writeToFile", &Matrix::writeToFile);

    // Bind enum classes
    nb::enum_<MatrixType>(m, "MatrixType").value("Dense", MatrixType::Dense).value("Sparse", MatrixType::Sparse);

    nb::enum_<MatrixCategory>(m, "MatrixCategory").value("Dynamics", MatrixCategory::Dynamics).value("Cost", MatrixCategory::Cost);

    // Bind the createStageCostMatrix function
    m.def("createStageCostMatrix", &createStageCostMatrix);

    // Bind the createTransitionProbabilityTensor function
    m.def("createTransitionProbabilityTensor", &createTransitionProbabilityTensor);

    // Bind the MatrixPreallocation struct
    nb::class_<MatrixPreallocation>(m, "MatrixPreallocation")
        .def(nb::init<>())
        .def_rw("d_nz", &MatrixPreallocation::d_nz)
        .def_rw("d_nnz", &MatrixPreallocation::d_nnz)
        .def_rw("o_nz", &MatrixPreallocation::o_nz)
        .def_rw("o_nnz", &MatrixPreallocation::o_nnz);

    // Bind types from madupite_matrix.h and madupite_vector.h
    nb::class_<Layout>(m, "Layout")
        .def(nb::init<MPI_Comm, PetscInt, bool>())
        .def("localSize", &Layout::localSize)
        .def("size", &Layout::size)
        .def("start", &Layout::start)
        .def("end", &Layout::end);

    nb::class_<Vector>(m, "Vector").def(nb::init<MPI_Comm, const std::string&, const Layout&>()).def("write", &Vector::write);
}
