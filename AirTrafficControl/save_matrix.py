from scipy.sparse import csr_matrix
import numpy as np

def writeBinarySparseMatrix(mat, filename):
    """write matrix in petsc binary sparse format

    Args:
        mat (ndarray): 2d-array to be written
        filename (string): filename

    copied from Philip
    """

    mat.sort_indices()
    with open(filename, "wb") as f:
        f.write(b"\x00\x12\x7b\x50")  # class id, sort of a magic number
        f.write(np.array(mat.shape, dtype=">i").tobytes())  # rows and cols
        f.write(np.array(mat.count_nonzero(), dtype=">i").tobytes())  # nnz
        f.write(
            np.array(np.diff(mat.indptr), dtype=">i").tobytes()
        )  # row pointer
        f.write(np.array(mat.indices, dtype=">i").tobytes())  # column indices
        f.write(np.array(mat.data, dtype=">d").tobytes())  # values
    with open(filename + ".info", "wb") as f:
        pass  # avoid petsc complaints


def writeBinaryDenseMatrix(mat, filename):
    """write matrix in petsc binary sparse format
    nnz = mat.shape[0]*mat.shape[1] because it is dense

    Args:
        mat (ndarray): 2d-array to be written
        filename (string): filename

    copied from Philip
    """
    nnz = mat.shape[0] * mat.shape[1]

    mat.sort_indices()
    with open(filename, "wb") as f:
        f.write(b"\x00\x12\x7b\x50")  # class id, sort of a magic number
        f.write(np.array(mat.shape, dtype=">i").tobytes())  # rows and cols
        f.write(np.array(nnz, dtype=">i").tobytes())  # nnz
        f.write(
            np.array(np.diff(mat.indptr), dtype=">i").tobytes()
        )  # row pointer
        f.write(np.array(mat.indices, dtype=">i").tobytes())  # column indices
        f.write(np.array(mat.data, dtype=">d").tobytes())  # values
    with open(filename + ".info", "wb") as f:
        pass  # avoid petsc complaints