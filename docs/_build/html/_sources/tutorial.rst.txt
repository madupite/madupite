Tutorial
===============
-----------
mpirun
-----------
The distributed sparse linear solvers for madupite rely on calls to MPI from PETSc. Therefore, you always need to use mpirun:

::

    export OMP_NUM_THREADS=1 # optional, see explaination below
    mpirun -n <number of ranks> python main.py

The number of MPI ranks is specified by the -n option. While PETSc's parallelism is based solely on MPI, the underlying BLAS/LAPACK routines may be parallelized with OpenMP, which can be controlled by setting the environment variable ``OMP_NUM_THREADS``. More details can be found in `the manual of PETSc <https://petsc.org/main/manual/blas-lapack/>`_ and the manual for mpirun of your MPI implementation.


---------------------------
Minimal example
---------------------------
This example demonstrates how to quickly load and solve a MDP.

.. code-block :: python

    import madupite as md
    with md.PETScContextManager():
        # create an mdp instance and load the transition probabilities and stage-costs
        mdp = md.PyMdp("./transprob.bin", "./stagecost.bin")
        # generate an initial cost and solve the MDP according to the default options
        mdp.solve("./outputDirectory/")

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

------------------------
The options.json file
------------------------

Specify the details of the solver. Every call of mdp.solve(...) will solve the loaded mdp according to the details set in the solver. If you want to solve the same MDP multiple times with different settings just change the options in the json file. The repository contains a template optionstemplate.json.

.. jsonschema:: optionstemplate.json


--------------------
Metadata format
--------------------

The metadata provides insights about convergence and timings. It copies the options used during solving and adds the following.

.. jsonschema:: metadatatemplate.json

.. code-block :: python

    import json
    metadata = json.load("./metadata.json")
    # access the total execution time of the solver
    totaltime = metadata["residual log"][-1]["time"]
    # get a list of the Bellman residuals for each iPI iteration
    residuals = [d["residual"] for d in metadata["residual log"]]

--------------------------------------------------------------
Generate arbitrary MDPs
--------------------------------------------------------------
`generateMDP() <https://n.ethz.ch/~ppawlowsky/madupite/apiref.html#madupite.generateMDP>`_ allows generating arbitrary MDPs. Users have to specify a function that returns the transition probabilities for a state-action pair in form of an index array and the corresponding values. The stage-cost function should return the cost of a state-action pair. The following examples show how to create the MDPs for the `forest management scenario by PyMDPToolbox <https://pymdptoolbox.readthedocs.io/en/latest/api/example.html#mdptoolbox.example.forest>`_ and the `tiger-antelope example by AI-Toolbox <https://github.com/Svalorzen/AI-Toolbox/blob/master/examples/MDP/tiger_antelope.cpp>`_.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Forest management example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block :: python

    import madupite as md
    import numpy as np

    def probfunc(state, action):
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

    def costfunc(state, action):
        if action == 0 and state == nstates - 1:
            return -r1
        if action == 1 and state > 0:
            if state == nstates - 1:
                return -r2
            else:
                return -1
        return 0

    md.generateMDP(
        nstates,
        mactions,
        probfunc,
        costfunc,
        "./data/sparse/P" + str(nstates) + ".bin",
        "./data/sparse/G" + str(nstates) + ".bin",
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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tiger-antelope example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The tiger chases the antelope on a 2d-grid with periodic boundary conditions ("wrap-around world"). The tiger can choose to move by one field in any direction (left, right, up, down or stay). The antelope can move to the same fields, but moves randomly. It does not move to the cell where the tiger is at. The original formulation assigns a negative cost to the finnal absorbing state but an equivalent formulation could assign a positive cost to every state but the absorbing state.

TODO .. literalinclude :: ../benchmarks/aitoolbox/gen.py
   :language: python
