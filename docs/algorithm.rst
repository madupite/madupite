Algorithm
=============

Algorithms to solve for the optimal policy and cost function of an MDP are known as **dynamic programming** algorithms. The most prominent algorithms include *value iteration* (VI) and *policy iteration* (PI). Unfortunately, dynamic programming suffers from the curse of dimensionality, making it infeasible to solve large-scale MDPs. 

Various approximation methods have been developed to address this issue, e.g. *optimistic/modified policy iteration* (OPI) which approximates the expensive policy evaluation step in PI using a finite number VI iterations.

``madupite`` relies on a novel class of algorithms called *inexact policy iteration* (iPI) which allows for any approximation method to be used in the policy evaluation step. We refer to Sec. 4 in [Gargiani2024]_ for a detailed description of iPI. A key advantage of iPI's flexibility is that depending on the problem structure, one can choose the most suitable approximation method for the policy evaluation step.

``madupite`` exploits this flexibiliy by allowing the user to select any iterative method available in PETSc by specifying the ``-ksp_type`` option (see :doc:`Options <options>`). Available methods include GMRES, BiCGStab, TFQMR etc. A full list of available solvers can be found `here <https://petsc.org/release/manualpages/KSP/KSPType/>`_. These methods typically show better convergence properties for almost undiscounted MDPs :math:`(\gamma \approx 1)` as they are not :math:`\gamma`-contractive unlike the Bellman operator used in VI.

Does ``madupite`` still support standard dynamic programming algorithms?
-------------------------------------------------------------------------

Yes, but be warned of the curse of dimensionality. Since iPI is a more general class of algorithms, you can retrieve PI, VI and OPI by setting the inner solver and tolerances accordingly.

**Policy Iteration** (PI): To evaluate the policy exactly, use singular value decomposition as a preconditioner. The choice of inner solver (KSP type) then becomes irrelevant (i.e. you can leave it unspecified which defaults to GMRES). *PETSc only supports SVD for sequential execution!*

.. code-block:: bash

    $ python ex1.py -pc_type svd

**Value Iteration** (VI): Since a VI step is equivalent to a step of Richardson iteration for a scaling factor of 1, you can retrieve VI by the tolerance to a very low value and limiting the number of iterations to 1 (Proposition 19 in [Gargiani2024]_):

.. code-block:: bash

    $ python ex1.py -ksp_type richardson -ksp_richardson_scale 1.0 -alpha 1e-40 -max_iter_ksp 1

**Optimistic Policy Iteration** (OPI): OPI is equivalent to VI but with a fixed number of iterations. Set therefore the value of ``-max_iter_ksp`` to the desired number of VI iterations, e.g. 50:

.. code-block:: bash
    
    $ python ex1.py -ksp_type richardson -ksp_richardson_scale 1.0 -alpha 1e-40 -max_iter_ksp 50

.. rubric:: References

.. [Gargiani2024] Gargiani, M.; Sieber. R.; Balta, E.; Liao-McPherson, D.; Lygeros, J. *Inexact Policy Iteration Methods for Large-Scale Markov Decision Processes*. `<https://arxiv.org/abs/2404.06136>`_.