#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "MDP.h"

// MDP: 1d grid world (=circle); 50 states; 3 actions (stay, left, right),
// 0: 90% stay, 5% left, 5% right
// 1: 10% stay, 90% left
// 2: 10% stay, 90% right
// goal state: 42 (index 41)

double r(const int s, const int a) { return (s == 41) ? 1.0 : 0.0; }

std::pair<std::vector<double>, std::vector<int>> P(const int s, const int a)
{
    std::vector<double> values;
    std::vector<int>    indices;
    switch (a) {
    case 0: // stay
        values  = { 0.9, 0.05, 0.05 };
        indices = { s, (s - 1 + 50) % 50, (s + 1) % 50 };
        break;
    case 1: // left
        values  = { 0.1, 0.9 };
        indices = { s, (s - 1 + 50) % 50 };
        break;
    case 2: // right
        values  = { 0.1, 0.9 };
        indices = { s, (s + 1) % 50 };
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
    auto comm     = PETSC_COMM_WORLD;

    // TODO need to specify Matrix P, g
    // 1. from function
    // 2. from hard-wired filename
    // 3. from CLI filename

    MDP mdp(madupite);
    mdp.setOption("-mode", "MAXREWARD");
    mdp.setOption("-discount_factor", "0.9999");
    mdp.setOption("-max_iter_pi", "200");
    mdp.setOption("-max_iter_ksp", "1000");
    mdp.setOption("-alpha", "1e-4");
    mdp.setOption("-atol_pi", "1e-8");
    mdp.setOption("-file_stats", "ci_stats.json");
    mdp.setOption("-file_cost", "ci_reward.out");
    mdp.setOption("-file_policy", "ci_policy.out");
    mdp.setOption("-ksp_type", "gmres");

    PetscInt numStates = 50, numActions = 3;

    auto g_mat = createStageCostMatrix(comm, "g_func", numStates, numActions, r);
    auto P_mat = createTransitionProbabilityTensor(comm, "P_func", numStates, numActions, P, { .d_nz = 3, .o_nz = 3 });

    mdp.setStageCostMatrix(g_mat);
    mdp.setTransitionProbabilityTensor(P_mat);

    mdp.solve();

    // Check solution
    // 42 (goal) has the highest reward, 17 has the lowest (1-based indexing)
    // only if rank 0 (else some error with stod?)
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        std::ifstream       file("ci_reward.out");
        std::string         line;
        std::vector<double> reward(50);
        double              max = -std::numeric_limits<double>::infinity(), min = std::numeric_limits<double>::infinity();
        int                 argmax = -1, argmin = -1;
        for (int i = 0; i < 50; ++i) {
            std::getline(file, line);
            reward[i] = std::stod(line);
            if (reward[i] > max) {
                max    = reward[i];
                argmax = i;
            }
            if (reward[i] < min) {
                min    = reward[i];
                argmin = i;
            }
        }
        file.close();

        assert(argmax == 41);
        assert(argmin == 16);
    }

    // Run 2: loading from binary file
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-discount_factor", "0.9");
    mdp.setOption("-alpha", "0.1");
    // mdp.setOption("-pc_type", "svd"); // standard PI (exact), only works in sequential

    g_mat = Matrix::fromFile(comm, "g_file", "../examples/ci_test/100_50_0.1/g.bin", MatrixCategory::Cost, MatrixType::Dense);
    P_mat = Matrix::fromFile(comm, "P_file", "../examples/ci_test/100_50_0.1/P.bin", MatrixCategory::Dynamics);

    mdp.setStageCostMatrix(g_mat);
    mdp.setTransitionProbabilityTensor(P_mat);

    mdp.solve();
    return 0;
}
