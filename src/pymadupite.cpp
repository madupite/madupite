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

NB_MODULE(_madupite_impl, m)
{
    m.doc() = "Python bindings for madupite";

    // nb::class_<Madupite>(m, "Madupite").def("initialize", []() { return Madupite::initialize(nullptr, nullptr); });

    // nb::class_<MatrixPreallocation>(m, "MatrixPreallocation")
    //     .def_rw("d_nz", &MatrixPreallocation::d_nz)
    //     .def_rw("d_nnz", &MatrixPreallocation::d_nnz)
    //     .def_rw("o_nz", &MatrixPreallocation::o_nz)
    //     .def_rw("o_nnz", &MatrixPreallocation::o_nnz);

    m.def("add", [](int a, int b) { return a + b; }, "Add two integers");
}
