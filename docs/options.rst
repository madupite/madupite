Madupite Options
================

Madupite options are specified using ``MDP.setOption()``. This function accepts either two strings, or a single string for boolean options. Numeric values should be passed as strings, e.g., ``"20"``, ``"0.1"``, ``"1e-10"``. Madupite options are built on top of PETSc options, allowing any PETSc option to be passed as well.

Required Options
----------------

.. option:: -mode <STRING>

   Specifies the optimization mode for the Markov Decision Process.

   Accepted values: ``"MAXREWARD"`` or ``"MINCOST"``

   This option determines whether the algorithm will maximize rewards or minimize costs.

.. option:: -discount_factor <DOUBLE>

   Sets the discount factor for future rewards or costs.

   Value range: :math:`(0, 1)`

   The discount factor determines the present value of future rewards. A value closer to 1 gives more weight to future rewards, while a value closer to 0 emphasizes immediate rewards.

Optional Options
----------------

.. option:: -max_iter_pi <INT>

   Specifies the maximum number of iterations for the inexact policy iteration algorithm.

   Default: ``1000``

   The algorithm will terminate after this many iterations, even if convergence has not been achieved. Must be a positive integer.

.. option:: -max_iter_ksp <INT>

   Sets the maximum number of iterations for the Krylov subspace method.

   Default: ``1000``

   This option limits the iterations in the approximate policy evaluation step. The method will terminate after this many iterations, even without convergence. Must be a positive integer.

.. option:: -atol_pi <DOUBLE>

   Defines the absolute tolerance for the inexact policy iteration algorithm.

   Default: ``1e-8``

   The algorithm terminates if the difference between the Bellman residual infinity norm is smaller than this value. Must be a positive double.

.. option:: -alpha <DOUBLE>

   Sets the forcing sequence parameter for the approximate policy evaluation step.

   Default: ``1e-4``

   This parameter influences the accuracy of the policy evaluation step. Must be a positive double.

.. option:: -file_stats <STRING>

   Specifies a file to write convergence and runtime information.

   This option enables writing detailed statistics about the algorithm's performance, useful for plotting and benchmarking.

.. option:: -file_policy <STRING>

   Designates a file to write the optimal policy.

   The optimal policy will be written in ASCII format, with entries separated by line breaks.

.. option:: -file_cost <STRING>

   Specifies a file to write the optimal cost-to-go (or reward-to-go) function.

   The function values will be written in ASCII format, with entries separated by line breaks.

.. option:: -export_optimal_transition_probabilities <STRING>

   Defines a file to write the optimal transition probabilities matrix.

   Exports the :math:`n \times n`-matrix of optimal transition probabilities in ASCII and COO format. The file header contains ``num_rows``, ``num_cols``, ``num_nonzeros``. Subsequent lines contain the row, column, and value of non-zero entries.

.. option:: -export_optimal_stage_costs <STRING>

   Specifies a file to write the optimal stage costs (or rewards) vector.

   Exports the :math:`n`-dimensional vector of optimal stage costs (or rewards) in ASCII format, with entries separated by line breaks.

.. option:: -filename_prefix <STRING>

   Specify a file prefix that is added to all file names. Specifically useful to add the Slurm jobname in front.


Useful PETSc Options
--------------------

.. option:: -ksp_type <STRING>

   Selects the Krylov subspace method for the inner solver of inexact policy iteration.

   Default: ``"gmres"``

   For a list of available algorithms, refer to the PETSc documentation: https://petsc.org/release/manualpages/KSP/KSPType/

.. option:: -pc_type <STRING>

   Chooses the preconditioner to use before applying the inner solver.

   Default: ``"none"``

   Only preconditioners that rely on the (transposed) matrix-vector product are supported. For the standard (exact) policy iteration algorithm, set this to "svd" (available only for sequential execution, not recommended for large-scale problems).

   For a list of available preconditioners, see: https://petsc.org/release/manualpages/PC/PCType/

.. option:: -log_view

   Enables output of a detailed algorithm log to the console.

   This option is useful for debugging and benchmarking purposes.

Usage Example
-------------

Command line usage:

.. code-block:: bash

   ./pendulum -discount_factor 0.999 -mode MINCOST -max_iter_pi 500

Using options file:

.. code-block:: bash

   ./pendulum -options options_file

Where `options_file` contains:

.. code-block:: text

   -discount_factor 0.999
   -mode MINCOST
   -max_iter_pi 500

Hard-coded options:

.. code-block:: python

   mdp = md.MDP()
   mdp.setOption("-mode", "MINCOST")
   mdp.setOption("-discount_factor", "0.999")
   # or
   mdp["-mode"] = "MINCOST"
   mdp["-discount_factor"] = 0.999


.. code-block:: c++

   MDP mdp;
   mdp.setOption("-discount_factor", "0.999");
   mdp.setOption("-mode", "MINCOST");
   mdp.setOption("-max_iter_pi", "500");

For more information on available KSP types and preconditioners, refer to the PETSc documentation:

* KSP types: https://petsc.org/release/manualpages/KSP/KSPType/
* Preconditioner types: https://petsc.org/release/manualpages/PC/PCType/
* PETSc options: https://petsc.org/release/manualpages/Sys/