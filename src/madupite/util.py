def writePETScBinary(matrix, filename, info_file=False):
    """Write numpy/scipy matrix as petsc binary sparse format to file
    https://petsc.org/release/manualpages/Mat/MatLoad/#notes

    Parameters
    ----------
    matrix : numpy/scipy matrix
        any matrix type that allows calling scipy.sparse.csr_array(matrix)
    filename : string
        output filename
    info_file : bool, optional
        write empty info file for PETSc versions that throw a warning if it does not exist, by default False
    """
    import numpy as np
    from scipy.sparse import csr_array

    csr_matrix = csr_array(matrix)
    csr_matrix.sort_indices()
    with open(filename, "wb") as f:
        f.write(b"\x00\x12\x7b\x50")  # class id, sort of a magic number
        f.write(np.array(matrix.shape, dtype=">i4").tobytes())  # rows and cols
        f.write(np.array(csr_matrix.nnz, dtype=">i4").tobytes())  # nnz
        f.write(
            np.array(np.diff(csr_matrix.indptr), dtype=">i4").tobytes()
        )  # row pointer
        f.write(np.array((csr_matrix.indices), dtype=">i4").tobytes())  # column indices
        f.write(np.array(csr_matrix.data, dtype=">f8").tobytes())  # values
    if info_file:
        with open(filename + ".info", "wb") as f:  # avoid petsc complaints
            pass
