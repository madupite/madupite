import pytest

import madupite


def test_initialize_madupite():
    madupite_instance = madupite.initialize_madupite()
    assert madupite_instance is not None


def test_get_comm_world():
    comm_world = madupite.getCommWorld()
    assert comm_world is not None


def test_mpi_rank_size():
    rank, size = madupite.mpi_rank_size()
    assert isinstance(rank, int)
    assert isinstance(size, int)
    assert rank >= 0
    assert size > 0


def test_matrix_preallocation_properties():
    prealloc = madupite.MatrixPreallocation()
    prealloc.d_nz = 10
    prealloc.d_nnz = [1, 2, 3]
    prealloc.o_nz = 5
    prealloc.o_nnz = [1, 1]

    assert prealloc.d_nz == 10
    assert prealloc.d_nnz == [1, 2, 3]
    assert prealloc.o_nz == 5
    assert prealloc.o_nnz == [1, 1]


def test_matrix_type_to_string():
    dense_str = madupite.Matrix.typeToString(madupite.MatrixType.Dense)
    sparse_str = madupite.Matrix.typeToString(madupite.MatrixType.Sparse)
    assert dense_str == "dense"
    assert sparse_str == "aij"


def test_matrix_from_file():
    matrix = madupite.Matrix.fromFile(
        comm=madupite.getCommWorld(),
        name="matrix_name",
        filename="../examples/ci_test/100_50_0.1/g.bin",
        category=madupite.MatrixCategory.Dynamics,
        type=madupite.MatrixType.Dense,
    )
    assert matrix is not None


def test_matrix_write_to_file():
    matrix = madupite.Matrix.fromFile(
        comm=madupite.getCommWorld(),
        name="matrix_name",
        filename="../examples/ci_test/100_50_0.1/g.bin",
        category=madupite.MatrixCategory.Dynamics,
        type=madupite.MatrixType.Dense,
    )
    matrix.writeToFile("output_file.out", madupite.MatrixType.Dense)


def test_create_transition_probability_tensor():
    def mock_func(state, action):
        return [state * action], [0]

    tensor = madupite.createTransitionProbabilityTensor(
        name="tensor_name", numStates=5, numActions=3, func=mock_func
    )
    assert tensor is not None


def test_create_stage_cost_matrix():
    def mock_cost_func(state, action):
        return state + action

    matrix = madupite.createStageCostMatrix(
        name="cost_matrix", numStates=5, numActions=3, func=mock_cost_func
    )
    assert matrix is not None


def test_mdp_setup_and_solve():
    madupite_instance = madupite.initialize_madupite()
    mdp = madupite.MDP(madupite_instance)

    mdp.clearOptions()
    mdp.setOption("-mode", "MAXREWARD")
    mdp.setOption("-discount_factor", "0.9999")
    mdp.setOption("-max_iter_pi", "200")
    mdp.setOption("-max_iter_ksp", "1000")
    mdp.setOption("-alpha", "1e-4")
    mdp.setOption("-atol_pi", "1e-8")
    mdp.setOption("-file_stats", "ci_stats.json")
    mdp.setOption("-file_cost", "ci_reward.out")
    mdp.setOption("-file_policy", "ci_policy.out")
    mdp.setOption("-ksp_type", "gmres")
    g = madupite.Matrix.fromFile(
        comm=madupite.getCommWorld(),
        name="matrix_name",
        filename="../examples/ci_test/100_50_0.1/g.bin",
        category=madupite.MatrixCategory.Cost,
        type=madupite.MatrixType.Dense,
    )
    P = madupite.Matrix.fromFile(
        comm=madupite.getCommWorld(),
        name="matrix_name",
        filename="../examples/ci_test/100_50_0.1/P.bin",
        category=madupite.MatrixCategory.Dynamics,
        type=madupite.MatrixType.Sparse,
    )
    mdp.setStageCostMatrix(g)
    mdp.setTransitionProbabilityTensor(P)
    mdp.setUp()
    mdp.solve()


if __name__ == "__main__":
    pytest.main()
