from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np
import matplotlib.pyplot as plt
import os
import struct

def generate_row_stochastic_matrix(states, actions, sparsity_factor, perturbFactor, seed, stddev):
    """generate a (n*m) x n row-stochastic matrix with random values assigned to randomly chosen columns"""
    nrows = states * actions
    ncols = states
    np.random.seed(seed)

    # Compute the number of non-zero entries per row with a perturbation
    perturbation = np.random.normal(0, stddev, size=nrows)
    nnz_per_row = (states * (sparsity_factor + perturbation)).astype(int)
    #nnz_per_row[0] = 23
    #print(nnz_per_row)

    # Ensure that nnzPerRow is in {1, ..., n-1}
    nnz_per_row = np.clip(nnz_per_row, 1, ncols - 1)
    #print(nnz_per_row)

    # Generate a row-stochastic matrix with random values assigned to randomly chosen columns
    data = []
    row_indices = []
    col_indices = []

    for i in range(nrows):
        # Generate nnzPerRow uniformly distributed random values for each row
        values = np.random.rand(nnz_per_row[i])
        perturb_indices = np.random.choice(nnz_per_row[i], int(nnz_per_row[i] * perturbFactor), replace=False)
        values[perturb_indices] /= perturbFactor

        # Assign the values to randomly chosen columns
        rand_cols = np.random.choice(ncols, nnz_per_row[i], replace=False)
        data.append(values.data)
        row_indices.append(np.repeat(i, nnz_per_row[i]))
        col_indices.append(rand_cols)

    # Flatten the data, row_indices, and col_indices arrays
    data = np.concatenate(data)
    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)

    # Create the sparse matrix using the csr_matrix constructor
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nrows, ncols))

    # Normalize the rows of the matrix to ensure that they sum to 1
    row_sums = np.array(sparse_matrix.sum(axis=1)).flatten()
    row_indices, col_indices = sparse_matrix.nonzero()
    data = sparse_matrix.data
    data /= row_sums[row_indices]
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(nrows, ncols))

    # Print the sparse matrix
    #print(sparse_matrix.toarray())
    #print(sparse_matrix.shape)
    #print(sparse_matrix.sum(axis=1))
    #exit(0)

    return sparse_matrix, nnz_per_row


def plot_nnz_histogram(n, m, nnz_per_row, sparsity_factor, stddev, seed, path):
    """function to see nnz per rows before generating the whole matrix to see if the values are reasonable"""

    #plot histogram of nnz per row
    bins = n
    counts, _, _ = plt.hist(nnz_per_row, bins=bins)
    plt.title(f"Histogram of nnz per row for n={n}, m={m}, sparsity={sparsity_factor},\n n*sparsity={n*sparsity_factor}, stddev={stddev}, seed={seed}")
    plt.xlabel("nnz per row")
    plt.ylim(0, 2*n)
    # mark expected value n*sparsity with a vertical line
    plt.axvline(x=n*sparsity_factor, color='r', linestyle='dashed', linewidth=1)
    plt.annotate(f'#1={int(counts[0])} ({counts[0]/(n*m)*100:0.2f}%)', (1, 0), textcoords="offset points", xytext=(-10,-30), ha='center')
    plt.savefig(f"{path}nnz_histogram_{stddev}.png", dpi=300)

    #exit(0) # exit program after plotting histogram, no matrix is generated


def writeBinarySparseMatrix(matrix, filename):
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


def writeBinaryVectorSeq(vector, filename):
    with open(filename, "wb") as f:
        # Write Petsc binary file header
        # Petsc classes
        VECSEQ = 1211214
        f.write(struct.pack('>i', VECSEQ))  # class id, VecSeq in Petsc
        f.write(struct.pack('>i', len(vector)))  # number of rows

        # Write the vector data
        for val in vector:
            f.write(struct.pack('>d', val))  # values

    # Write info file to avoid Petsc complaints
    with open(filename + ".info", "wb") as f:
        pass


def generate_stagecost_matrix(states, actions, perturbFactor, seed):
    """generate a stage cost matrix of dimension states x actions"""
    np.random.seed(seed)
    stagecost = np.random.rand(states, actions) * 10 # random values between 0 and 10
    # perturb uniformity
    indices = np.random.choice(states*actions, int(perturbFactor*states*actions), replace=False)
    rows, cols = np.unravel_index(indices, (states, actions))
    # multiply values at rows and cols by 1/perturbFactor
    stagecost[rows, cols] /= perturbFactor
    
    return stagecost



def matrix_from_file(filename, filename_out):
    """loads a dense matrix from file in csv, stores it as sp csr matrix and writes to binary file"""
    dense = np.loadtxt(filename, delimiter=",")
    sparse = csr_matrix(dense)
    writeBinarySparseMatrix(sparse, filename_out)
    exit(0)


def save_as_csv(P, g, path):
    """save P and g as csv files"""
    # if P has more than 1000 rows return
    if P.shape[0] > 1000:
        print("P has more than 1000 rows, not saving as csv")
        return
    np.savetxt(f"{path}P.csv", P.toarray(), delimiter=",")
    np.savetxt(f"{path}g.csv", g, delimiter=",")




def main():

    states = 100
    actions = 10
    sparsity = 0.1
    seed = 123982
    stddev = 0.02 # for normal distribution for perturbation of sparsity factor
    perturbFactor = 0.1 # perturbFactor * n*m of stagecosts are perturbed (higher by 1/perturbFactor)

    path = "data/" + f"{states}_{actions}_{sparsity:0.6f}/"
    # create folder if it does not exist "states_actions_sparsity"
    if not os.path.exists(path):
        os.makedirs(path)


    P, nnz = generate_row_stochastic_matrix(states, actions, sparsity, perturbFactor, seed, stddev)
    g = generate_stagecost_matrix(states, actions, perturbFactor, seed)

    print(nnz.shape)
    print(*nnz)

    plot_nnz_histogram(states, actions, nnz, sparsity, stddev, seed, path)

    # save matrices to file
    writeBinarySparseMatrix(P, path + "P.bin")
    writeBinarySparseMatrix(g, path + "g.bin")
    writeBinaryVectorSeq(np.array(nnz, dtype=np.float64), path + "nnz.bin") # change to double s.t petsc can read it

    #matrix_from_file("../../dataP_200_20_0.100000_8624.csv", path + "P.bin")

    row_stochasticity_error = np.linalg.norm(np.array(P.sum(axis=1)) - 1.0, ord=np.inf)

    # save used values to file
    with open(path + "info.txt", "w") as f:
        f.write(f"states={states}\n")
        f.write(f"actions={actions}\n")
        f.write(f"sparsity={sparsity}\n")
        f.write(f"stddev={stddev}\n")
        f.write(f"perturbFactor={perturbFactor}\n")
        f.write(f"seed={seed}\n")
        f.write(f"row_stochasticity_error={row_stochasticity_error}\n")
        f.write(f"P.shape={P.shape}\n")
        f.write(f"g.shape={g.shape}\n")
        f.write(f"nnz.shape={nnz.shape}\n")

    save_as_csv(P, g, path)

if __name__ == "__main__":
    main()
