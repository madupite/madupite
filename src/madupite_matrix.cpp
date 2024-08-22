#include <petscmat.h>

#include "madupite_errors.h"
#include "madupite_matrix.h"

std::string Matrix::typeToString(MatrixType type)
{
    switch (type) {
    case MatrixType::Dense:
        return MATDENSE;
    case MatrixType::Sparse:
        return MATAIJ;
    default:
        throw MadupiteException("Unsupported matrix type");
    }
}

Matrix::Matrix(MPI_Comm comm, const std::string& name, MatrixType type)
{
    const auto typeStr = Matrix::typeToString(type);

    PetscCallThrow(MatCreate(comm, &_mat));
    PetscCallThrow(PetscObjectSetName((PetscObject)_mat, name.c_str()));
    PetscCallThrow(PetscObjectSetOptionsPrefix((PetscObject)_mat, (name + '_').c_str()));
    PetscCallThrow(MatSetType(_mat, typeStr.c_str()));
    PetscCallThrow(MatSetFromOptions(_mat));
    // This private constructor does not set the layouts; they should always be set in the public constructors
}

Matrix::Matrix(
    MPI_Comm comm, const std::string& name, MatrixType type, const Layout& rowLayout, const Layout& colLayout, const MatrixPreallocation& pa)
    : Matrix(comm, name, type)
{
    _rowLayout = rowLayout;
    _colLayout = colLayout;
    PetscCallThrow(MatSetLayouts(_mat, rowLayout.petsc(), colLayout.petsc()));
    auto d_nnz_ptr = pa.d_nnz.empty() ? nullptr : pa.d_nnz.data();
    auto o_nnz_ptr = pa.d_nnz.empty() ? nullptr : pa.d_nnz.data();
    // for any matrix type, only one of the following applies and the rest is no-op
    PetscCallThrow(MatSeqAIJSetPreallocation(_mat, pa.d_nz, d_nnz_ptr));
    PetscCallThrow(MatMPIAIJSetPreallocation(_mat, pa.d_nz, d_nnz_ptr, pa.o_nz, o_nnz_ptr));
    PetscCallThrow(MatSeqDenseSetPreallocation(_mat, nullptr));
    PetscCallThrow(MatMPIDenseSetPreallocation(_mat, nullptr));
    PetscCallThrow(MatSetUp(_mat));
}

// TODO PETSc feature proposal: it would be nice if the PETSc loader infers the correct type from the file
Matrix Matrix::fromFile(MPI_Comm comm, const std::string& name, const std::string& filename, MatrixCategory category, MatrixType type)
{
    if (comm == MPI_COMM_NULL) {
        throw MadupiteException("MADUPITE: Invalid MPI communicator");
    }

    Matrix      A(comm, name, type);
    PetscViewer viewer;
    auto        mat = A.petsc();

    PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
    PetscInt sizes[4];
    PetscInt dummy;
    // must read the following first such that we can specify the distribution of rows (local rows) by ourselves
    // since we split by the granularity of states and not the rows as equally as possible
    // to ensure that all data associated with a state is on the same process
    // Read ClassId, Rows, Cols, NNZ
    PetscCallThrow(PetscViewerBinaryRead(viewer, sizes, 4, &dummy, PETSC_INT));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    switch (category) {
    case MatrixCategory::Dynamics: {
        PetscInt numStates      = sizes[2];
        PetscInt numActions     = sizes[1] / sizes[2];
        PetscInt localNumStates = PETSC_DECIDE;
        PetscCallThrow(PetscSplitOwnership(comm, &localNumStates, &numStates));
        PetscCallThrow(MatSetSizes(mat, localNumStates * numActions, PETSC_DECIDE, PETSC_DECIDE, numStates));
        break;
    }
    case MatrixCategory::Cost: {
        PetscInt numStates      = sizes[1];
        PetscInt numActions     = sizes[2];
        PetscInt localNumStates = PETSC_DECIDE;
        PetscCallThrow(PetscSplitOwnership(comm, &localNumStates, &numStates));
        PetscCallThrow(MatSetSizes(mat, localNumStates, PETSC_DECIDE, PETSC_DECIDE, numActions));
        break;
    }
    default:
        throw MadupiteException("Unsupported matrix category");
    }

    PetscCallThrow(PetscViewerBinaryOpen(comm, filename.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(mat, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));

    PetscLayout pRowLayout, pColLayout;
    PetscCallThrow(MatGetLayouts(mat, &pRowLayout, &pColLayout));
    A._rowLayout = Layout(pRowLayout);
    A._colLayout = Layout(pColLayout);
    return A;
}

// Get row in AIJ format
std::vector<PetscScalar> Matrix::getRow(PetscInt row) const
{
    PetscInt           n;
    const PetscScalar* data;
    auto               mat = const_cast<Mat>(petsc());

    PetscCallThrow(MatGetRow(mat, row, &n, nullptr, &data));
    auto result = std::vector<PetscScalar>(data, data + n);
    PetscCallThrow(MatRestoreRow(mat, row, &n, nullptr, &data));
    return result;
}

// write matrix content to file in ascii or binary format
void Matrix::writeToFile(const std::string& filename, MatrixType type, bool binary = false) const
{
    auto mat = const_cast<Mat>(petsc());
    if (binary) {
        PetscViewer viewer;
        auto        mat = const_cast<Mat>(petsc());
        PetscCallThrow(PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE, &viewer));
        PetscCallThrow(MatView(mat, viewer));
        PetscCallThrow(PetscViewerDestroy(&viewer));
    } else {
        if (type == MatrixType::Dense) {
            PetscViewer viewer;
            PetscCallThrow(PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer));
            PetscCallThrow(MatView(mat, viewer));
            PetscCallThrow(PetscViewerDestroy(&viewer));
        } else if (type == MatrixType::Sparse) {
            PetscInt    m, n, rstart, rend;
            PetscViewer viewer;
            PetscMPIInt rank, size;
            MatInfo     info;
            PetscInt    nz_global;

            PetscCallThrow(MatGetSize(mat, &m, &n));
            PetscCallThrow(MatGetOwnershipRange(mat, &rstart, &rend));
            PetscCallThrow(MPI_Comm_rank(PetscObjectComm((PetscObject)mat), &rank));
            PetscCallThrow(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));

            // Get matrix info
            PetscCallThrow(MatGetInfo(mat, MAT_GLOBAL_SUM, &info));
            nz_global = (PetscInt)info.nz_used;

            PetscCallThrow(PetscViewerCreate(PetscObjectComm((PetscObject)mat), &viewer));
            PetscCallThrow(PetscViewerSetType(viewer, PETSCVIEWERASCII));
            PetscCallThrow(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
            PetscCallThrow(PetscViewerFileSetName(viewer, filename.c_str()));

            // Write the first line with matrix dimensions and global non-zeros (global_rows, global_cols, global_nz)
            if (rank == 0) {
                PetscCallThrow(PetscViewerASCIIPrintf(viewer, "%d,%d,%d\n", m, n, nz_global));
            }

            PetscCallThrow(PetscViewerASCIIPushSynchronized(viewer));

            for (PetscInt row = rstart; row < rend; row++) {
                PetscInt           ncols;
                const PetscInt*    cols;
                const PetscScalar* vals;
                PetscCallThrow(MatGetRow(mat, row, &ncols, &cols, &vals));
                for (PetscInt j = 0; j < ncols; j++) {
                    // rowidx, colidx, value
                    PetscCallThrow(PetscViewerASCIISynchronizedPrintf(viewer, "%d,%d,%.15e\n", row, cols[j], (double)PetscRealPart(vals[j])));
                }
                PetscCallThrow(MatRestoreRow(mat, row, &ncols, &cols, &vals));
            }

            PetscCallThrow(PetscViewerFlush(viewer));
            PetscCallThrow(PetscViewerASCIIPopSynchronized(viewer));
            PetscCallThrow(PetscViewerDestroy(&viewer));
        } else {
            throw MadupiteException("Unsupported matrix type");
        }
    }
}
