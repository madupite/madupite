from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np
import random


def generate_row_stochastic_matrix(n, sparsity_factor, seed, stddev):
    np.random.seed(seed)

    # Compute the number of non-zero entries per row with a perturbation
    perturbation = np.random.normal(0, stddev, size=n)
    nnz_per_row = np.round(n * (sparsity_factor + perturbation)).astype(int)
    #nnz_per_row[0] = 23
    #print(nnz_per_row)

    # Ensure that nnzPerRow is in {1, ..., n-1}
    nnz_per_row = np.clip(nnz_per_row, 1, n - 1)
    print(nnz_per_row)

    # Generate a row-stochastic matrix with random values assigned to randomly chosen columns
    data = []
    row_indices = []
    col_indices = []

    for i in range(n):
        # Generate nnzPerRow uniformly distributed random values for each row
        #values = random(1, nnz_per_row[i], density=1.0, format='csr', dtype=float)
        values = np.random.rand(nnz_per_row[i])

        # Assign the values to randomly chosen columns
        rand_cols = np.random.choice(n, nnz_per_row[i], replace=False)
        data.append(values.data)
        row_indices.append(np.repeat(i, nnz_per_row[i]))
        col_indices.append(rand_cols)

    # Flatten the data, row_indices, and col_indices arrays
    data = np.concatenate(data)
    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)

    # Create the sparse matrix using the csr_matrix constructor
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # Normalize the rows of the matrix to ensure that they sum to 1
    row_sums = np.array(sparse_matrix.sum(axis=1)).flatten()
    row_indices, col_indices = sparse_matrix.nonzero()
    data = sparse_matrix.data
    data /= row_sums[row_indices]
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    # Print the sparse matrix
    #print(sparse_matrix.toarray())

    return sparse_matrix


def generateTransitionProbabilityTensor(states, actions, sparsity, seed, stddev):
    """stacks m sparse matrices horizontally of dimension n x n to a sparse matrix of dimension n x (m*n)"""
    for i in range(actions):
        if i == 0:
            P = generate_row_stochastic_matrix(states, sparsity, seed, stddev)
        else:
            P = hstack((P, generate_row_stochastic_matrix(states, sparsity, seed + i, stddev)))

    return P





def writeBinarySparse(matrix, filename):
    """write matrix in petsc binary sparse format

    Args:
        matrix (ndarray): 2d-array to be written
        filename (string): filename

    copied from Philip
    """
    mat = csr_matrix(matrix)
    mat.sort_indices()
    with open(filename, "wb") as f:
        f.write(b"\x00\x12\x7b\x50")  # class id, sort of a magic number
        f.write(np.array(matrix.shape, dtype=">i").tobytes())  # rows and cols
        f.write(np.array(mat.count_nonzero(), dtype=">i").tobytes())  # nnz
        f.write(
            np.array(np.diff(mat.indptr), dtype=">i").tobytes()
        )  # row pointer
        f.write(np.array(mat.indices, dtype=">i").tobytes())  # column indices
        f.write(np.array(mat.data, dtype=">d").tobytes())  # values
    with open(filename + ".info", "wb") as f:
        pass  # avoid petsc complaints


def main():

    states = 200
    actions = 20
    seed = 8624
    sparsity = 0.05
    stddev = 0.1 # for normal distribution for perturbation of sparsity factor

    P = generateTransitionProbabilityTensor(states, actions, sparsity, seed, stddev)
    print(P.shape)
    print("Row-stochasticity error =", np.linalg.norm(np.array(P.sum(axis=1)) - np.full((states, 1), actions), ord=np.inf))

    filename_P = f"../../data/sp_P_{states}_{actions}_{sparsity:0.6f}_{seed}.bin"
    writeBinarySparse(P, filename_P)

if __name__ == "__main__":
    main()
