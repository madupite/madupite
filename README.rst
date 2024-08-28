``madupite``
============

``madupite`` is a high-performance C++ library with a Python interface designed for solving large-scale **Markov Decision Processes (MDPs)** using **Distributed Inexact Policy Iteration (iPI)**. Leveraging distributed sparse linear solvers from `PETSc <https://petsc.org/>`_, ``madupite`` efficiently handles the computational complexity associated with large-scale MDPs.

Key Features
------------
- **Scalable MDP Solver**: Efficiently solve MDPs with large state and action spaces using distributed computation.
- **Python and C++ APIs**: Access the power of ``madupite`` through both Python and C++, depending on your performance needs.
- **Distributed Computing**: Integrates with PETSc and MPI for distributed computing on multi-core and cluster environments.

Installation
------------
After cloning the repository, you can install the Python package by running:

::

    conda env create -f environment.yml
    conda activate madupiteenv
    pip install .



Examples
--------

::

    import madupite as md
    g = md.Matrix.fromFile("g.bin", category=md.MatrixCategory.Cost)
    P = md.Matrix.fromFile("P.bin", category=md.MatrixCategory.Dynamics)

### Generating Data
madupite can also generate MDP data directly, leveraging its parallel computation capabilities for faster data generation.

Example:

::

    import madupite as md

    def costfunc(s, a):
        return s + a

    def probfunc(s, a):
        return [0.2, 0.8], [s, (s + a) % 50]

    g = md.createStageCostMatrix(numStates=50, numActions=3, func=costfunc)
    P = md.createTransitionProbabilityTensor(numStates=50, numActions=3, func=probfunc)

### Matrix Preallocation
For large MDPs with sparse transition probability tensors, preallocating matrices can significantly improve performance. Use `madupite.MatrixPreallocation` to specify the number of non-zero elements per row.

Find more examples in Python and C++ in the `examples <examples>`_ directory.

Contributing
------------
Contributions to madupite are welcome! Whether it's reporting bugs, suggesting features, or submitting pull requests, we appreciate your input.

License
-------
``madupite`` is distributed under the MIT License. See the `LICENSE` file for more information.
