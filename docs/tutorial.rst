Tutorials
===============

In the following, we propose some short tutorials which we hope can help the users to better understand the deployment of some of the fundamental functionalities of ``madupite``.

Initializing and finalizing ``madupite`` 
---------------------------------------------------

``madupite`` can be imported as any other Python package. Though, since it relies on MPI for the distributed memory parallelism, it is crucial to first initialize the MPI and PETSc environment to ensure that the communication between the processes is set up correctly. 
In addition, to properly finalize all MPI jobs, we suggest that the code is contained inside a main function.

.. code-block:: python
    
    import madupite as md
    instance = md.initialize_madupite()

    def main():

        #main body of your code :)

Loading and reading data with ``madupite``
----------------------------------------------

In this tutorial, we want to show how to load and read MDP data that is stored in files. The data itself must be stored as a PETSc binary file (explained `here <https://petsc.org/release/manualpages/Mat/MatLoad/>`_). ``madupite`` provides a method to save numpy or scipy matrices to PETSc binary files (:func:`madupite.writePETScBinary`).


Assuming the stage cost matrix and transition probability tensor are stored as ``g.bin`` and ``P.bin`` we can load them with :func:`madupite.Matrix.fromFile`. We need to specify whether it is a stage cost matrix (``md.MatrixCategory.Cost``) or a transition probability tensor (``md.MatrixCategory.Dynamics``) to ensure that the number of states and actions is correctly inferred.

Furthermore, you can specify whether the matrix is sparse or dense using the ``md.MatrixType`` enum. Sparse matrices are stored in a compressed format, which can save memory and speed up computations. 

Notice that, unlike with function simulations, defining an object for matrix preallocation is not necessary when loading from files since the information about non-zero elements is already contained in the binary file.

.. code-block:: python

    import madupite as md
    import numpy as np


    def main():
        # Write matrices to file
        numStates = 10
        numActions = 3
        cost_matrix = np.random.rand(numStates, numActions)
        md.writePETScBinary(cost_matrix, "g.bin")
        prob_matrix = np.random.rand(numStates * numActions, numStates)
        # normalize rows to 1, in order to have a valid transition probability matrix
        prob_matrix /= prob_matrix.sum(axis=1)[:, None]
        md.writePETScBinary(prob_matrix, "P.bin")

        # Load matrices from file
        g = md.Matrix.fromFile(
            filename="g.bin",
            category=md.MatrixCategory.Cost,
            type=md.MatrixType.Dense,
        )
        P = md.Matrix.fromFile(
            filename="P.bin",
            category=md.MatrixCategory.Dynamics,
            type=md.MatrixType.Sparse,
        )


    if __name__ == "__main__":
        main()


.. warning::
    Note that as of ``madupite`` V1.0, the files themselves must contain the data in a sparse format because PETSc does not support reading dense matrices from binary files. By specifying the matrix type as dense, the data will be read as a sparse matrix and then converted to a dense matrix. This is recommended for stage cost matrices to benefit from data locality and speed up computations.

Generating data with ``madupite``
---------------------------------
Depending on the problem, creating the MDP data with numpy and reading them with ``madupite`` is often slower than generating them directly with ``madupite``. This is because ``madupite`` can  generate the transition probabilities in parallel and in the correct format, which avoids the need to convert the data.

In the following example, we show how to generate the stage cost matrix and transition probability tensor with ``madupite``. We define a cost function and a probability function that are used to generate the data. The cost function takes the current state and action as input and returns the cost. The probability function takes the current state and action as input and returns the transition probabilities and the next state indices.

.. code-block:: python

    import madupite as md


    def costfunc(s, a):
        return s + a


    def probfunc(s, a):
        transition_probabilities = [0.2, 0.8]
        state_indices = [s, (s + a) % 50]
        return transition_probabilities, state_indices


    def main():
        num_states = 50
        num_actions = 3
        g = md.createStageCostMatrix(
            numStates=num_states, numActions=num_actions, func=costfunc
        )
        P = md.createTransitionProbabilityTensor(
            numStates=num_states,
            numActions=num_actions,
            func=probfunc,
        )

    if __name__ == "__main__":
        main()


Matrix preallocation
-----------------------------------------
For large MDPs with sparse transition probability tensors, it is often beneficial to preallocate the matrices to avoid reallocations during the computation. This can be done by specifying the ``preallocation`` argument. The method takes an instance of the :class:`madupite.MatrixPreallocation` class, which specifies the number of non-zero elements per row in the diagonal and off-diagonal block. See the example below for more details (adapted from `PETSc <https://petsc.org/release/manualpages/Mat/MatMPIAIJSetPreallocation/>`_).

Consider the following 8x8 matrix with 34 non-zero values, that is
assembled across 3 ranks. Let's assume that rank0 owns 3 rows,
rank1 owns 3 rows, rank2 owns 2 rows. This division can be shown
as follows:

.. code-block::

             1  2  0  |  0  3  0  |  0  4
     rank0   0  5  6  |  7  0  0  |  8  0
             9  0 10  | 11  0  0  | 12  0
     -------------------------------------
            13  0 14  | 15 16 17  |  0  0
     rank1   0 18  0  | 19 20 21  |  0  0
             0  0  0  | 22 23  0  | 24  0
     -------------------------------------
     rank2  25 26 27  |  0  0 28  | 29  0
            30  0  0  | 31 32 33  |  0 34

This can be represented as a collection of submatrices as:

.. code-block::

       A B C
       D E F
       G H I

Where the submatrices A, B, C are owned by rank0, D, E, F are
owned by rank1, G, H, I are owned by rank2.

The DIAGONAL submatrices corresponding to rank0, rank1, rank2 are
submatrices [A], [E], [I] respectively. The OFF-DIAGONAL submatrices
corresponding to rank0, rank1, rank2 are [BC], [DF], [GH] respectively.

When ``d_nz``, ``o_nz`` parameters are specified, ``d_nz`` storage elements are
allocated for every row of the local diagonal submatrix, and ``o_nz``
storage locations are allocated for every row of the OFF-DIAGONAL submatrix.
Typically one chooses ``d_nz`` and ``o_nz`` as the max nonzeros per local
rows for each of the local DIAGONAL, and the OFF-DIAGONAL submatrices.
In this case, the values of ``d_nz``, ``o_nz`` are:

.. code-block::

      rank0  d_nz = 2, o_nz = 2
      rank1  d_nz = 3, o_nz = 2
      rank2  d_nz = 1, o_nz = 4

When ``d_nnz``, ``o_nnz`` parameters are specified, the storage is specified
for every row, corresponding to both DIAGONAL and OFF-DIAGONAL submatrices.
In the above case the values for ``d_nnz``, ``o_nnz`` are:

.. code-block::

      rank0 d_nnz = [2,2,2] and o_nnz = [2,2,2]
      rank1 d_nnz = [3,3,2] and o_nnz = [2,1,1]
      rank2 d_nnz = [1,1]   and o_nnz = [4,4]

.. code-block:: python

    import madupite as md
    # ...
    rank, size = md.mpi_rank_size()
    # Option 1
    pc = md.MatrixPreallocation()
    if rank == 0:
        pc.d_nz = 2
        pc.o_nz = 2
    elif rank == 1:
        pc.d_nz = 3
        pc.o_nz = 2
    else:
        pc.d_nz = 1
        pc.o_nz = 4
    # Option 2
    pc2 = md.MatrixPreallocation()
    if rank == 0:
        pc2.d_nnz = [2, 2, 2]
        pc2.o_nnz = [2, 2, 2]
    elif rank == 1:
        pc2.d_nnz = [3, 3, 2]
        pc2.o_nnz = [2, 1, 1]
    else:
        pc2.d_nnz = [1, 1]
        pc2.o_nnz = [4, 4]
    
    def probfunc(s, a):
        return [1], [0]

    P1 = md.createTransitionProbabilityTensor(
        numStates=8,
        numActions=1,
        func=probfunc,
        preallocation=pc
    )

    P2 = md.createTransitionProbabilityTensor(
        numStates=8,
        numActions=1,
        func=probfunc,
        preallocation=pc2
    )

Data format
-----------
The data format for the MDP is defined by the stage cost matrix and the transition probability tensor. The stage cost matrix is a matrix of size ``numStates x numActions``, where each element (s, a) represents the cost of taking action a in state s. The transition probabilities are usually expressed as a tensor of size ``numStates x numActions x numStates``, where each element (s, a, s') represents the probability of transitioning from state s to state s' after applying action a. For ``madupite`` the tensor is flattened to a matrix of size ``numStates*numActions x numStates``, where each row i represents the transition probabilities from state i // numStates to state s' after applying action i % numStates.

The tensor can be reshaped as follows:

:: 

    >>> import numpy as np
    >>> numStates = 3
    >>> numActions = 2
    >>> P=np.array(
    ...     [[[0.5,  0.5,  0.0 ],
    ...       [0.25, 0.33, 0.42]],
    ...   
    ...      [[0.3,  0.3,  0.4 ],
    ...       [0.4,  0.2,  0.4 ]],
    ...   
    ...      [[0.6 , 0.1,  0.3 ],
    ...       [0.7 , 0.1,  0.2 ]]])
    >>> 
    >>> P.reshape((numStates*numActions, numStates))
    array([[0.5 , 0.5 , 0.  ],
           [0.25, 0.33, 0.42],
           [0.3 , 0.3 , 0.4 ],
           [0.4 , 0.2 , 0.4 ],
           [0.6 , 0.1 , 0.3 ],
           [0.7 , 0.1 , 0.2 ]])


The MDP-class in ``madupite``
----------------------------------------------

Now that all the main ingredients are explained, we are ready to introduce the MDP-class, which is basically where all the magic of ``madupite`` happens! This class allows you to create and solve your own MDP, and it comes with a lot of options that you can customize. 
TODO