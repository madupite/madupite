Installation
============

To install ``madupite``, first clone the repository from GitHub:

::

   git clone https://github.com/madupite/madupite.git


--------------
 Dependencies
--------------

You need to make sure that the dependencies are available. A convenient way of installing them is via the conda environment file, which you can find in the repo.
::

   conda env create -f environment.yml
   conda activate madupiteenv

Users that want to use their own PETSc or MPI version should make sure that cmake can find them.

----------------
 Python package
----------------

After installing the necessary dependencies you can install madupite for Python via pip:
::

   pip install .

Make sure to run your executables with mpirun. <number_of_ranks> could be the number of cores on your machine. If you want to check whether your installation was successful, you can run the following command:
::

   mpirun -n <number_of_ranks> python examples/install/main.py

---------------
 C++
---------------
For advanced users looking to use the software from C++, ensure you have a functional PETSc installation. From there, you can proceed to build the project using CMake as follows:
::

   mkdir build
   cd build
   cmake ..
   make

Make sure to run your executables with mpirun. <number_of_ranks> could be the number of cores on your machine:
::

   mpirun -n <number_of_ranks> ./build/main

.. note::
   As of now, only the Python API is documented. However, users can refer to the C++ example provided for guidance.

------------------------------
Euler (ETH Zurich Cluster)
------------------------------

**Python Version:**

To install the Python package on the Euler cluster, first load the necessary software modules:

::

   ./load-euler-modules.sh

Then, install the Python package using pip:

::

   pip install .

**C++ Version:**

To build the C++ project on the Euler cluster, load the necessary software modules:

::

   ./load-euler-modules.sh

Then, follow the same steps as for the general C++ installation:

::

   mkdir build
   cd build
   cmake ..
   make

The repository contains an example launch file `euler-launch.sh` to run the executables on the Euler cluster.
