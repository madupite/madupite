Algorithm
=============

**Dynamic programming** (DP) comprises algorithms to compute the optimal cost and an optimal policy of an MDP. The main DP methods are *value iteration* (VI) and *policy iteration* (PI) [Bertsekas]_. As dynamic programming suffers from the so called *curse of dimensionality*, there is need for solution methods which scale and are distributable. 

Various approximation methods have been developed to address this issue, *e.g.*, *optimistic/modified policy iteration* (OPI) approximates the expensive policy evaluation step in PI using a finite number VI iterations.

``madupite`` relies on a novel class of algorithms called *inexact policy iteration* (iPI) methods which are based on an inexact policy evaluation step. We refer to Sec. 4 in [Gargiani2024]_ for a detailed description of iPI. A key advantage of iPI's flexibility is that depending on the problem structure, one can choose the most suitable approximation method for the policy evaluation step.

``madupite`` exploits this flexibiliy by allowing the user to select any iterative method available in PETSc via the ``-ksp_type`` option (see :doc:`Options <options>`). Available methods include GMRES, BiCGStab, TFQMR etc. A full list of available solvers can be found `here <https://petsc.org/release/manualpages/KSP/KSPType/>`_. Some of these methods show superior convergence properties for almost undiscounted MDPs :math:`(\gamma \approx 1)` than VI.

Does ``madupite`` still support standard dynamic programming algorithms such as VI and PI?
-------------------------------------------------------------------------

Yes, but be aware of their performance limitations. Since iPI is a more general class of algorithms, you can retrieve PI, VI and OPI by setting the inner solver and tolerances accordingly.

**Policy Iteration** (PI): To evaluate the policy exactly, use singular value decomposition as a preconditioner. The choice of inner solver (KSP type) in this case becomes irrelevant. You can therefore simply leave it unspecified which defaults to GMRES. *PETSc only supports SVD for sequential execution!*

.. code-block:: bash

    $ python ex1.py -pc_type svd

**Value Iteration** (VI): Since a VI step for policy evaluation is equivalent to one iteration of Richardson method with a scaling factor of 1, you can retrieve VI by limiting the number of inner iterations to 1 and selecting Richardson method as inner solver (Proposition 19 in [Gargiani2024]_).

.. code-block:: bash

    $ python ex1.py -ksp_type richardson -ksp_richardson_scale 1.0 -alpha 1e-40 -max_iter_ksp 1

**Optimistic Policy Iteration** (OPI): You can retrieve OPI by selecting the same parameters as for VI, but with a number of inner iterations that can potentially be greater than 1. In particular, set the value of ``-max_iter_ksp`` to the desired number of inner VI iterations you would like to be performed, *e.g.*, 50. Finally, be sure to set the inner tolerance parameter to a very small value such that it does not impact on the execution of the inner loop.

.. code-block:: bash
    
    $ python ex1.py -ksp_type richardson -ksp_richardson_scale 1.0 -alpha 1e-40 -max_iter_ksp 50

.. rubric:: References

.. [Bertsekas] D. P. Bertsekas. *Dynamic Programming and Optimal Control*, Vol. II, Athena Scientific, Belmont, Massachusets, 4th edition, 2012.
.. [Gargiani2024] Gargiani, M.; Sieber. R.; Balta, E.; Liao-McPherson, D.; Lygeros, J. *Inexact Policy Iteration Methods for Large-Scale Markov Decision Processes*. `<https://arxiv.org/abs/2404.06136>`_.