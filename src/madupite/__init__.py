"""
madupite is a Python package for solving Markov Decision Processes (MDPs).
"""
from ._madupite_impl import (  # type: ignore
    Madupite,
    MatrixPreallocation,
    Matrix,
    MDP,
    MatrixType,
    MatrixCategory,
    initialize_madupite,
    getCommWorld,
    mpi_rank_size,
    createTransitionProbabilityTensor,
    createStageCostMatrix,
    __doc__,
)
from .util import writePETScBinary

# Avoid having users forget to initialize madupite.
instance = initialize_madupite()
