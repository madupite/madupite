from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_row_stochastic_matrix(n, sparsity_factor, perturbFactor, seed, stddev):
    np.random.seed(seed)

    # Compute the number of non-zero entries per row with a perturbation
    perturbation = np.random.normal(0, stddev, size=n)
    nnz_per_row = np.round(n * (sparsity_factor + perturbation)).astype(int)
    #nnz_per_row[0] = 23
    #print(nnz_per_row)

    # Ensure that nnzPerRow is in {1, ..., n-1}
    nnz_per_row = np.clip(nnz_per_row, 1, n - 1)
    #print(nnz_per_row)

    # Generate a row-stochastic matrix with random values assigned to randomly chosen columns
    data = []
    row_indices = []
    col_indices = []

    for i in range(n):
        # Generate nnzPerRow uniformly distributed random values for each row
        #values = random(1, nnz_per_row[i], density=1.0, format='csr', dtype=float)
        values = np.random.rand(nnz_per_row[i])
        perturb_indices = np.random.choice(nnz_per_row[i], int(nnz_per_row[i] * perturbFactor), replace=False)
        values[perturb_indices] /= perturbFactor

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

    return sparse_matrix, nnz_per_row


def generateTransitionProbabilityTensor(states, actions, sparsity, perturbFactor, seed, stddev):
    """stacks m sparse matrices horizontally of dimension n x n to a sparse matrix of dimension n x (m*n)"""
    for i in range(actions):
        if i == 0:
            P, nnz = generate_row_stochastic_matrix(states, sparsity, perturbFactor, seed, stddev)
            nnz_matrix = np.array(nnz)
        else:
            matrix, nnz = generate_row_stochastic_matrix(states, sparsity, perturbFactor, seed + i, stddev)
            P = hstack((P, matrix))
            nnz_matrix = np.vstack([nnz_matrix, nnz])

    nnz_matrix = nnz_matrix.T
    print("nnz_matrix:")
    print(nnz_matrix)
    print(nnz_matrix.shape)
    # todo: store to sparse matrix bin
    return P, nnz_matrix


def plot_nnz_histogram(n, sparsity_factor, stddev, seed, path):
    """function to see nnz per rows before generating the whole matrix to see if the values are reasonable"""

    np.random.seed(seed)
    perturbation = np.random.normal(0, stddev, size=n)
    nnz_per_row = np.round(n * (sparsity_factor + perturbation)).astype(int)
    nnz_per_row = np.clip(nnz_per_row, 1, n - 1)
    print(f"Testing values for n={n}, sparsity={sparsity_factor}, n*sparsity={n*sparsity_factor}, stddev={stddev}, seed={seed}")
    print("average nnz =", np.average(nnz_per_row))
    #print(*nnz_per_row, sep=", ")

    #plot histogram of nnz per row
    plt.hist(nnz_per_row, bins=int(n/10)) # 10 values per bin
    plt.title(f"Histogram of nnz per row for n={n}, sparsity={sparsity_factor},\n n*sparsity={n*sparsity_factor}, stddev={stddev}, seed={seed}")
    plt.xlabel("nnz per row")
    plt.ylim(0, 2*n*sparsity_factor)
    # mark expected value n*sparsity with a vertical line
    plt.axvline(x=n*sparsity_factor, color='r', linestyle='dashed', linewidth=1)
    plt.savefig(f"{path}nnz_histogram_{stddev}.png", dpi=300)

    #exit(0) # exit program after plotting histogram, no matrix is generated


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


def generate_stagecost_matrix(states, actions, perturbFactor, seed):
    """generate a stage cost matrix of dimension states x actions"""
    np.random.seed(seed)
    stagecost = np.random.rand(states, actions)
    # perturb uniformity
    indices = np.random.choice(states*actions, int(perturbFactor*states*actions), replace=False)
    rows, cols = np.unravel_index(indices, (states, actions))
    # multiply values at rows and cols by 1/perturbFactor
    stagecost[rows, cols] *= 1/perturbFactor
    
    return stagecost



def matrix_from_file(filename, filename_out):
    """loads a dense matrix from file in csv, stores it as sp csr matrix and writes to binary file"""
    dense = np.loadtxt(filename, delimiter=",")
    sparse = csr_matrix(dense)
    writeBinarySparse(sparse, filename_out)
    exit(0)



def main():

    states = 5000
    actions = 40
    seed = 8624
    sparsity = 0.01
    stddev = 0.005 # for normal distribution for perturbation of sparsity factor
    perturbFactor = 0.15 # perturbFactor * n*m of stagecosts are perturbed (higher by 1/perturbFactor)

    path = "data/" + f"{states}_{actions}_{sparsity:0.6f}/"
    # create folder if it does not exist "states_actions_sparsity"
    if not os.path.exists(path):
        os.makedirs(path)
     
    #matrix_from_file("../../dataP_200_20_0.100000_8624.csv", path + "P.bin")

    plot_nnz_histogram(states, sparsity, stddev, seed, path)

    P, nnz = generateTransitionProbabilityTensor(states, actions, sparsity, perturbFactor, seed, stddev)
    row_stochasticity_error = np.linalg.norm(np.array(P.sum(axis=1)) - np.full((states, 1), actions), ord=np.inf)
    g = generate_stagecost_matrix(states, actions, perturbFactor, seed)
    
    # save used values to file
    with open(path + "info.txt", "w") as f:
        f.write(f"states={states}\n")
        f.write(f"actions={actions}\n")
        f.write(f"sparsity={sparsity}\n")
        f.write(f"stddev={stddev}\n")
        f.write(f"perturbFactor={perturbFactor}\n")
        f.write(f"seed={seed}\n")
        f.write(f"row_stochasticity_error={row_stochasticity_error}\n")

    # save matrices to file
    writeBinarySparse(P, path + "P.bin")
    writeBinarySparse(g, path + "g.bin")
    writeBinarySparse(nnz, path + "nnz.bin")


if __name__ == "__main__":
    main()
