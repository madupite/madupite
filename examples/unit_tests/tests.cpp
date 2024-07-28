#include "MDP.h"
#include <gtest/gtest.h>

// Test fixture
class TF : public ::testing::Test {
public:
    MPI_Comm comm;

protected:
    void SetUp() override { comm = PETSC_COMM_WORLD; }
};

// Test matrix multiplication
TEST_F(TF, MatMul)
{
    Layout rowLayout(comm, 6);
    Layout colLayout(comm, 3);

    Matrix matrix(comm, "mat", MatrixType::Dense, rowLayout, colLayout);

    Vector x(comm, "x", colLayout);
    Vector y(comm, "y", rowLayout);

    for (size_t i = x.layout().start(); i < x.layout().end(); i++) {
        x(i, i + 1);
    }
    x.assemble();

    for (size_t i = matrix.rowLayout().start(); i < matrix.rowLayout().end(); i++) {
        for (size_t j = 0; j < matrix.colLayout().size(); j++) {
            matrix(i, j, i + j);
        }
    }
    matrix.assemble();

    matrix.mult(x, y);

    PetscScalar expectedVals[] = { 8, 14, 20, 26, 32, 38 };

    for (PetscInt i = y.layout().start(); i < y.layout().end(); ++i) {
        EXPECT_EQ(y(i), expectedVals[i]);
    }
}

// Test matrix transposed multiplication
TEST_F(TF, MatMulTranspose)
{
    Layout rowLayout(comm, 6);
    Layout colLayout(comm, 3);

    Matrix matrix(comm, "mat", MatrixType::Dense, rowLayout, colLayout);

    Vector x(comm, "x", rowLayout);
    Vector y(comm, "y", colLayout);

    for (size_t i = x.layout().start(); i < x.layout().end(); i++) {
        x(i, i + 1);
    }
    x.assemble();

    for (size_t i = matrix.rowLayout().start(); i < matrix.rowLayout().end(); i++) {
        for (size_t j = 0; j < matrix.colLayout().size(); j++) {
            matrix(i, j, i + j);
        }
    }
    matrix.assemble();

    matrix.multT(x, y);

    PetscScalar expectedVals[] = { 70, 91, 112 };

    for (PetscInt i = y.layout().start(); i < y.layout().end(); ++i) {
        EXPECT_EQ(y(i), expectedVals[i]);
    }
}

// Test matrix addition
TEST_F(TF, MatrixAdd)
{
    Layout rowLayout(comm, 6);
    Layout colLayout(comm, 3);

    Matrix matrix1(comm, "mat1", MatrixType::Dense, rowLayout, colLayout);
    Matrix matrix2(comm, "mat2", MatrixType::Dense, rowLayout, colLayout);

    for (size_t i = matrix1.rowLayout().start(); i < matrix1.rowLayout().end(); i++) {
        for (size_t j = 0; j < matrix1.colLayout().size(); j++) {
            matrix1(i, j, i + j);
            matrix2(i, j, 2 * (i + j));
        }
    }
    matrix1.assemble();
    matrix2.assemble();

    matrix1.add(matrix2);

    for (size_t i = matrix1.rowLayout().start(); i < matrix1.rowLayout().end(); i++) {
        for (size_t j = 0; j < matrix1.colLayout().size(); j++) {
            EXPECT_EQ(matrix1(i, j), 3 * (i + j));
        }
    }
}

TEST_F(TF, InitializeWithoutData)
{
    Layout layout(comm, 10, true);
    Vector vec(comm, "test_vector", layout);

    EXPECT_EQ(vec.name(), "test_vector");

    PetscScalar value;

    for (PetscInt i = vec.layout().start(); i < vec.layout().end(); ++i) {
        value = vec(i);
        EXPECT_EQ(value, 0);
    }
}

TEST_F(TF, InitializeWithData)
{
    std::vector<PetscScalar> data = { 1.0, 2.0, 3.0 };
    Vector                   vec(comm, "test_vector_with_data", data);

    EXPECT_EQ(vec.name(), "test_vector_with_data");

    EXPECT_EQ(vec.layout().localSize(), data.size());

    for (PetscInt i = vec.layout().start(); i < vec.layout().end(); ++i) {
        EXPECT_EQ(vec(i), data[i - vec.layout().start()]);
    }
}

TEST_F(TF, CopyConstructor)
{
    std::vector<PetscScalar> data = { 1.0, 2.0, 3.0 };
    Vector                   vec1(comm, "original_vector", data);
    Vector                   vec2 = vec1;

    EXPECT_EQ(vec2.name(), "original_vector");

    for (PetscInt i = vec2.layout().start(); i < vec2.layout().end(); ++i) {
        EXPECT_EQ(vec2(i), data[i - vec2.layout().start()]);
    }
}

// Test Vector copy assignment
TEST_F(TF, CopyAssignment)
{
    std::vector<PetscScalar> data1 = { 1.0, 2.0, 3.0 };
    std::vector<PetscScalar> data2 = { 4.0, 5.0, 6.0 };
    Vector                   vec1(comm, "vector1", data1);
    Vector                   vec2(comm, "vector2", data2);
    vec2 = vec1;

    EXPECT_EQ(vec2.name(), "vector1");

    for (PetscInt i = vec2.layout().start(); i < vec2.layout().end(); ++i) {
        EXPECT_EQ(vec2(i), data1[i - vec2.layout().start()]);
    }
}

TEST_F(TF, MoveAssignment)
{
    std::vector<PetscScalar> data1 = { 1.0, 2.0, 3.0 };
    std::vector<PetscScalar> data2 = { 4.0, 5.0, 6.0 };
    Vector                   vec1(comm, "vector1", data1);
    Vector                   vec2(comm, "vector2", data2);
    vec2 = std::move(vec1);

    EXPECT_EQ(vec2.name(), "vector1");

    for (PetscInt i = vec2.layout().start(); i < vec2.layout().end(); ++i) {
        EXPECT_EQ(vec2(i), data1[i - vec2.layout().start()]);
    }
}

int main(int argc, char** argv)
{
    std::shared_ptr<Madupite> madupite = Madupite::initialize(&argc, &argv);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    int                            rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    return RUN_ALL_TESTS();
}
