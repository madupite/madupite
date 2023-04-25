import numpy as np
import matplotlib.pyplot as plt

# generate a random matrix from a uniform distribution
n = 100
values = np.random.uniform(0, 1, (n, n))
A = np.matrix(values)

# compute the eigenvalues and eigenvectors of A
eigvals, eigvecs = np.linalg.eig(A)

# plot the eigenvalues on the complex plane
plt.scatter(np.real(eigvals), np.imag(eigvals))
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Eigenvalues of Random Matrix')
plt.show()