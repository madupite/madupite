Madupite Options
================

Madupite comes with a lot of options that allow the user to customize her/his MDP as well as the method used to solve it. Despite the name, some of these options are actually mandatory as they are needed to correctly define the optimization problem. Others are only optional and serve the purpose of customization. Madupite options are built on top of PETSc options, allowing any PETSc option to be passed as well.

Mandatory Options
----------------

The options listed here are **mandatory**, meaning that we require the user to actively specify them. 

.. option:: -mode <STRING>

   Specifies the optimization mode.

   Accepted values: ``"MAXREWARD"`` or ``"MINCOST"``

   This option determines whether the algorithm will maximize rewards or minimize costs.

.. option:: -discount_factor <DOUBLE>

   Sets the discount factor.

   Value range: :math:`(0, 1)`

   The discount factor determines the present value of future rewards. A value closer to 1 gives more weight to future rewards, while a value closer to 0 emphasizes immediate rewards.

Facultative Options
----------------

The options listed here are **facultative**. If the user does not actively specify these options, default values will be used. 

.. option:: -max_iter_pi <INT>

   Specifies the maximum number of iterations for the inexact policy iteration algorithm.

   Default: ``1000``

   The algorithm will terminate after this many iterations, even if convergence has not been achieved. Must be a positive integer.

.. option:: -max_iter_ksp <INT>

   Sets the maximum number of iterations for the Krylov subspace method.

   Default: ``1000``

   This option limits the iterations in the approximate policy evaluation step. The method will terminate after this many iterations, even if convergence has not been achieved. Must be a positive integer.

.. option:: -atol_pi <DOUBLE>

   Defines the absolute tolerance for the inexact policy iteration algorithm.

   Default: ``1e-8``

   The algorithm terminates if the infinity-norm of the Bellman residual function is smaller than this value. Must be a positive double.

.. option:: -alpha <DOUBLE>

   Sets the forcing sequence parameter for the approximate policy evaluation step.

   Default: ``1e-4``

   This parameter influences the accuracy of the policy evaluation step. In general, the smaller the value of this parameter and the more accurate is the cost returned by approximate policy evaluation step. Must be a positive double.

.. option:: -file_stats <STRING>

   Specifies a file to write convergence and runtime information.

   This option enables writing detailed statistics about the algorithm's performance, which can be used for plotting and benchmarking.

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

   Specify a file prefix that is added to all file names. It can for instance be used to add the job-ID when running benchmarks in a Slurm cluster.

.. option:: -verbose <BOOLEAN>

   Enable console output of option values and algorithm progress. 
   
   Default: ``false``

.. option:: -overwrite <BOOLEAN>

   Allow overwriting existing files. This might be useful while debugging to avoid creating new files for each run or to avoid e.g. adapting filenames in a plotting script. 
   
   Default: ``false``


Useful PETSc Options
--------------------

.. option:: -ksp_type <STRING>

   Selects the Krylov subspace method for the inner solver of inexact policy iteration.

   Default: ``"gmres"``

   For a list of available algorithms, refer to the PETSc documentation: https://petsc.org/release/manualpages/KSP/KSPType/

.. option:: -pc_type <STRING>

   Chooses the preconditioner to use before applying the inner solver.

   Default: ``"none"``

   Only preconditioners that rely on the (transposed) matrix-vector product are supported. For the standard (exact) policy iteration algorithm, set this to "svd" (available only for sequential execution, not recommended for large-scale problems) or "lu".

   For a list of available preconditioners, see: https://petsc.org/release/manualpages/PC/PCType/

.. option:: -ksp_view <BOOLEAN>

   Prints information about the Krylov subspace method (inner solver) to the console.

   Default: ``false``

.. option:: -pc_svd_monitor <BOOLEAN>

   When using SVD as preconditioner, this option outputs the condition number of the matrix :math:`P^\pi` to the console at each outer iteration. 

   Default: ``false``


How To Correctly Set the Options
--------------------------------

There are different ways to set the options in Madupite. Since it is best explained with examples, down below we use the pendulum example to showcase the different ways that can be used to set the options in Madupite.

Command line usage:

.. code-block:: bash

   mpirun -n <number_of_ranks> python pendulum.py -discount_factor 0.999 -mode MINCOST -max_iter_pi 500 -verbose True

Using options in a ``.txt`` file:

.. code-block:: bash

   mpirun -n <number_of_ranks> python pendulum.py -options <filename>.txt

Where ``<filename>.txt`` contains:

.. code-block:: text

   -discount_factor 0.999
   -mode MINCOST
   -max_iter_pi 500
   -verbose True

You can use the same logics to set the options also when using ``madupite`` from C++ or when running in single-core.

Hard-coded options:

.. code-block:: python

   # Python
   mdp = md.MDP()
   mdp.setOption("-mode", "MINCOST")
   mdp.setOption("-discount_factor", "0.999")
   mdp.setOption("-verbose", "True")
   # or
   mdp["-mode"] = "MINCOST"
   mdp["-discount_factor"] = 0.999
   mdp["verbose"] = True

.. code-block:: c++

   // C++
   MDP mdp;
   mdp.setOption("-discount_factor", "0.999");
   mdp.setOption("-mode", "MINCOST");
   mdp.setOption("-max_iter_pi", "500");
   mdp.setOption("-max_iter_pi", "True");


.. warning::

   As of ``madupite`` V1.0, ``MDP::setOption()`` uses a global options data base even though it is called on an instance of the MDP class. This means that ``mdp1.setOption("-discount_factor", "0.999")`` will also set the discount factor for ``mdp2``. This holds for both C++ and Python.

For more information on available KSP types and preconditioners, refer to the PETSc documentation:

* KSP types: https://petsc.org/release/manualpages/KSP/KSPType/
* Preconditioner types: https://petsc.org/release/manualpages/PC/PCType/
* PETSc options: https://petsc.org/release/manualpages/Sys/