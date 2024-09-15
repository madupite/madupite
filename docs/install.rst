Installation
============

To install ``madupite``, first clone the repository from GitHub:

::

   git clone https://github.com/madupite/madupite.git


--------------
 Dependencies
--------------

You need to make sure that the dependencies are available. A convenient way of installing them is via the conda environment file, which you can find in the repo. Make sure you have an updated version of conda installed, then run the following commands:
::
  
   cd madupite
   conda env create -f environment.yml
   conda activate madupiteenv

Users who want to use their own PETSc or MPI version should make sure that cmake can find them.

----------------
 Python package
----------------

After installing the necessary dependencies you can install ``madupite`` for Python via pip:
::

   pip install .

Make sure to run your executables with ``mpirun`` if you want to run them exploiting parallel-computing. ``<number_of_ranks>`` could be the number of cores on your machine. If you want to check whether your installation was successful, you can run the following command:
::

   mpirun -n <number_of_ranks> python examples/install/main.py

---------------
 C++
---------------
For advanced users who want to use the software from C++, they should make sure to have a working PETSc installation. From there, you can proceed to build the project using CMake as follows:
::

   mkdir build
   cd build
   cmake ..
   make

Make sure to run your executables with ``mpirun``. ``<number_of_ranks>`` could be the number of cores on your machine:
::

   mpirun -n <number_of_ranks> ./build/main

.. note::
   As of now, only the Python API is documented. However, C++ users can refer to the examples provided in the ``examples`` folder for guidance.

------------------------------
Euler (ETH Zurich Cluster)
------------------------------

**Python Version:**

To install the Python package on the Euler cluster, access a login node and clone the repository

::

   git clone https://github.com/madupite/madupite.git

Load the necessary software modules:

::

   source ./examples/euler/euler-load-modules.sh

Then, install the Python package using pip:

::

   pip install .

**Jupyter Notebook:**

If you want to use ``madupite`` with  `JupyterHub on Euler` you need to install the python package as shown above. After that, copy the module load script to Jupyter.

:: 

   mkdir -p ~/.config/euler/jupyterhub/jupyterlabrc 
   cp ~/madupite/examples/euler/euler-load-modules.sh ~/.config/euler/jupyterhub/jupyterlabrc

Now, you can leave the login node and go to `JupyterHub on Euler <https://jupyter.euler.hpc.ethz.ch/>`_.

**C++ Version:**

To build the C++ project on the Euler cluster, load the necessary software modules:

::

   source ./examples/euler/euler-load-modules.sh

Then, follow the same steps as for the general C++ installation:

::

   mkdir build
   cd build
   cmake ..
   make

The repository contains an example launch file `examples/euler/euler-launch.sh` to run the executables on the Euler cluster.
