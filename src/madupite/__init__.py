"""
madupite is a Python package for solving Markov Decision Processes (MDPs).
"""
from .madupite import (  # type: ignore
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
)

# Avoid having users forget to initialize madupite.
madupite.initialize_madupite()
