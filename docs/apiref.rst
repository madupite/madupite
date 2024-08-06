API Reference
=====================
The `madupite` module provides interfaces for working with the Madupite library, including functions and classes for managing matrices, Markov Decision Processes (MDP), and MPI communications.

Functions
---------

.. autosummary::
   :toctree: _autosummary

   madupite.initialize_madupite
   madupite.getCommWorld
   madupite.mpi_rank_size
   madupite.createTransitionProbabilityTensor
   madupite.createStageCostMatrix
   madupite.writePETScBinary

Classes
-------

.. autosummary::
   :toctree: _autosummary

   madupite.Madupite
   madupite.MatrixType
   madupite.MatrixCategory
   madupite.MatrixPreallocation
   madupite.Matrix
   madupite.MDP