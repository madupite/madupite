Tutorial
===============
-----------
mpirun
-----------
The distributed sparse linear solvers for madupite rely on calls to MPI from PETSc. Therefore, you always need to use mpirun:

::

    export OMP_NUM_THREADS=1 # optional, see explanation below
    mpirun -n <number of ranks> python main.py

The number of MPI ranks is specified by the -n option. While PETSc's parallelism is based solely on MPI, the underlying BLAS/LAPACK routines may be parallelized with OpenMP, which can be controlled by setting the environment variable ``OMP_NUM_THREADS``. More details can be found in `the manual of PETSc <https://petsc.org/main/manual/blas-lapack/>`_ and the manual for mpirun of your MPI implementation.


---------------------------
Minimal example
---------------------------
This example demonstrates how to quickly load and solve a MDP.

.. code-block :: python

    import madupite as md
    # always initialize madupite
    md.initialize_madupite()
    g = md.Matrix.fromFile(
        name="g",
        filename="g.bin",
        category=md.MatrixCategory.Cost,
        type=md.MatrixType.Dense,
    )
    P = md.Matrix.fromFile(
        name="P",
        filename="P.bin",
        category=md.MatrixCategory.Dynamics,
        type=md.MatrixType.Sparse,
    )
    mdp.setStageCostMatrix(g)
    mdp.setTransitionProbabilityTensor(P)

    mdp.setOption("-mode", "MAXREWARD")
    mdp.setOption("-discount_factor", "0.95")
    mdp.solve()

--------------------------------------------------------------
Generate an artificial MDP and solve it with various options
--------------------------------------------------------------
Since the routines for MDP generating are sequential they need only be executed by the main rank, which is why they are contained inside the ``if md.MPI_master()`` statement.


TODO .. literalinclude :: ../examples/example1.py
   :language: python


--------------------
Input data format
--------------------
The data for the MDP (transition probability tensor and stage cost matrix) must be provided in the PETSc binary format. The installation with pip provides a generator for artificial MDPs and a converter from numpy/scipy matrices to PETSc binary.

::

    import madupite as md
    md.writePETScBinary(matrix, "path/to/matrix.bin")

If you want to create your own MDP you must match the data layout of this solver:

- stagecost matrix: :math:`G_{i,j}=` "expected stage-cost of applying input j in state i".

- The transition probability is often expressed as a 3-dimensional tensor. In order to make use of the parallel matrix layout from PETSc, the first 2-dimensions must be combined:

.. math::

    P_{i,j,k}= \text{"transition probability from state i to k given input j"}

is flattened into a matrix:

.. math::

    i&=\left\lfloor\frac{x}{\# actions}\right\rfloor \\
    a&=x\mod \# actions \\
    P'_{x,j}&= \text{"transition probability from state } i \text{ to state j given input } a.

It holds that :math:`P_{i,j,k}=P'_{i*\#actions+j,k}` and can be implemented in Python as:

.. code-block :: python

    # combine first and second dimension of a 3d numpy array in python
    1stdim, 2nddim, 3rddim = transprobtensor.shape
    transprobmat = transprobtensor.reshape(1stdim * 2nddim, 3rddim)


--------------------------------------------------------------
Generate arbitrary MDPs
--------------------------------------------------------------
The transition probabilities can be defined for a state-action pair in form of an index array and the corresponding values. 
The stage-cost function should return the cost of a state-action pair. 
The following examples show how to create the MDPs for the `forest management scenario by PyMDPToolbox <https://pymdptoolbox.readthedocs.io/en/latest/api/example.html#mdptoolbox.example.forest>`_. 

.. code-block :: python

    import madupite as md
    import numpy as np

    num_states = 10
    num_actions = 2

    def stage_cost_function(state, action):
        if action == 0 and state == nstates - 1:
            return -r1
        if action == 1 and state > 0:
            if state == nstates - 1:
                return -r2
            else:
                return -1
        return 0

    def transition_probability_function(state, action):
        if action == 0:
            idx, val = np.array(
                [0, min(state + 1, nstates - 1)], dtype=np.float64
            ), np.array([p, 1 - p])
            return idx, val
        else:
            idx, val = np.array([0], dtype=np.float64), np.array(
                [1], dtype=np.float64
            )
            return idx, val

    # madupite.Matrix object containing the stage costs
    g = madupite.createStageCostMatrix(
        name="g", numStates=num_states, numActions=num_actions, func=stage_cost_function
    )
    # madupite.Matrix object containing the transition probabilities
    P = madupite.createTransitionProbabilityTensor(
        name="P",
        numStates=num_states,
        numActions=num_actions,
        func=transition_probability_function,
    )


This will create the following MDP::

                   | p 1-p 0.......0  |
                   | .  0 1-p 0....0  |
        P[:,0,:] = | .  .  0  .       |
                   | .  .        .    |
                   | .  .         1-p |
                   | p  0  0....0 1-p |

                   | 1 0..........0 |
                   | . .          . |
        P[:,1,:] = | . .          . |
                   | . .          . |
                   | . .          . |
                   | 1 0..........0 |

                 |  0  |
                 |  .  |
        R[:,0] = |  .  |
                 |  .  |
                 |  0  |
                 |  r1 |

                 |  0  |
                 |  1  |
        R[:,1] = |  .  |
                 |  .  |
                 |  1  |
                 |  r2 |
