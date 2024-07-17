#include <gtest/gtest.h>
#include "madupite_matrix.h"
#include "madupite_errors.h"
#include "MDP.h"
#include "petscmat.h"
#include <mpi.h>

// Test fixture
class TF : public ::testing::Test {
public:
    MPI_Comm comm;
    std::shared_ptr<Madupite> madupite;
protected:
    void SetUp() override {
        madupite = Madupite::initialize();
        comm     = PETSC_COMM_WORLD;
    }
};

TEST_F(TF, MatMul) {
    Layout rowLayout(comm, 6);
    Layout colLayout(comm, 3);

    Matrix matrix(comm, "testMatrix", MatrixType::Dense, rowLayout, colLayout);

    Vector x(comm, "x", colLayout);
    Vector y(comm, "y", rowLayout);

    for (size_t i = x.layout().start(); i < x.layout().end(); i++)
    {
        x(i, i + 1);
    }
    x.assemble();

    for (size_t i = matrix.rowLayout().start(); i < matrix.rowLayout().end(); i++)
    {
        for (size_t j = 0; j < matrix.colLayout().size(); j++)
        {
            matrix(i, j, i + j);
        }
    }
    matrix.assemble();

    matrix.mult(x, y);

    PetscScalar expectedVals[] = {8, 14, 20, 26, 32, 38};

    for (PetscInt i = y.layout().start(); i < y.layout().end(); ++i) {
        EXPECT_EQ(y(i), expectedVals[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
