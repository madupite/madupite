#include <iostream>
#include <utility>
#include <vector>

#include "MDP.h"

// 2D Grid world
constexpr int       _H = 25, _W = 25; // s = row-major index
std::pair<int, int> s2hw(int s) { return { s / _W, s % _W }; }
int                 hw2s(int h, int w) { return h * _W + w; }
/* bool                is_boundary(int h, int w) { return h == 0 || h == _H - 1 || w == 0 || w == _W - 1; } */

std::pair<std::vector<double>, std::vector<int>> P(const int s, const int a)
{
    std::vector<double> values;
    std::vector<int>    indices;
    auto [h, w] = s2hw(s);
    switch (a) {
    case 0: // stay
        values  = { 1.0 };
        indices = { s };
        break;
    case 1: // north
        if (h == 0) {
            values  = {};
            indices = {};
        } else {
            values  = { 1.0 };
            indices = { hw2s(h - 1, w) };
        }
        break;
    case 2: // east
        if (w == _W - 1) {
            values  = {};
            indices = {};
        } else {
            values  = { 1.0 };
            indices = { hw2s(h, w + 1) };
        }
        break;
    case 3: // south
        if (h == _H - 1) {
            values  = {};
            indices = {};
        } else {
            values  = { 1.0 };
            indices = { hw2s(h + 1, w) };
        }
        break;
    case 4: // west
        if (w == 0) {
            values  = {};
            indices = {};
        } else {
            values  = { 1.0 };
            indices = { hw2s(h, w - 1) };
        }
        break;
    default:
        std::cerr << "invalid action index\n";
    }
    return { values, indices };
}

int main(int argc, char** argv)
{
    // Initialize MPI, PETSc and Madupite, passing command line arguments.
    auto madupite = Madupite::initialize(&argc, &argv);

    MDP mdp(madupite);
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-discount_factor", "0.99");
    mdp.setOption("-max_iter_pi", "200");
    mdp.setOption("-max_iter_ksp", "1000");
    mdp.setOption("-alpha", "1e-6");
    mdp.setOption("-atol_pi", "1e-8");
    mdp.setOption("-file_stats", "maze_stats.json");
    mdp.setOption("-file_cost", "maze_cost.out");
    mdp.setOption("-file_policy", "maze_policy.out");
    mdp.setOption("-ksp_type", "tfqmr");

    auto comm = PETSC_COMM_WORLD;

    auto g_mat = Matrix::fromFile(comm, "g_file", "data/maze_25x25.bin", MatrixCategory::Cost, MatrixType::Dense);
    auto P_mat = createTransitionProbabilityTensor(comm, "P_func", _H * _W, 5, P);

    mdp.setStageCostMatrix(g_mat);
    mdp.setTransitionProbabilityTensor(P_mat);

    mdp.solve();
}
