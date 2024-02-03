Installation
============

--------------
 Dependencies
--------------

First, you need to make sure that the dependencies are available. A convenient way of installing them is via the conda environment file, which you can find in the repo.
::

   conda env create -f environment.yml
   conda activate petscmadupite

Users that want to use their own PETSc or MPI version should make sure that cmake can find them with pkg-config.

----------------
 Python package
----------------

Assuming you have a working version of PETSc, e.g. by activating the conda environment above, you can install with pip inside the repo.
::

   pip install .

Make sure to always run your executables with mpirun. <number_of_ranks> could be the number of cores on your machine:
::

   mpirun -N <number_of_ranks> python main.py


---------------
 C++
---------------
Assuming you have a working version of PETSc, e.g. by activating the conda environment above, call the following for a build with cmake.
::

   mkdir build
   cd build
   cmake ..
   make

Now, you can change the main.cc file according to your application.
Make sure to always run your executables with mpirun. <number_of_ranks> could be the number of cores on your machine:
::

   mpirun -N <number_of_ranks> ./build/main


------------------------------
 Euler (ETH Zurich Cluster)
------------------------------
Make sure to use the new software stack (run the command env2lmod). The file moduleload.sh is provided in the repo.
::

   # this loads the correct software dependencies
   source moduleload.sh

   # Python-Version:
   pip install .

   # C++-Version:
   mkdir build
   cd build
   cmake ..
   make

Specify your job and the compute ressources in the launch.sh file.
