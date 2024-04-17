#include "MDP.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <petsc.h>
#include <string>
#include <utility>
#include <vector>

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
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR);

    MDP mdp;
    mdp.setOption("-mode", "MAXREWARD");
    mdp.setOption("-discount_factor", "0.9999");
    mdp.setOption("-max_iter_pi", "200");
    mdp.setOption("-max_iter_ksp", "1000");
    mdp.setOption("-num_pi_runs", "1");
    mdp.setOption("-rtol_ksp", "1e-4");
    mdp.setOption("-atol_pi", "1e-8");
    mdp.setOption("-num_states", "50");
    mdp.setOption("-num_actions", "3");
    mdp.setOption("-file_stats", "ci_stats.json");
    mdp.setOption("-file_cost", "ci_reward.out");
    mdp.setOption("-file_policy", "ci_policy.out");
    mdp.setOption("-ksp_type", "gmres");
    mdp.setValuesFromOptions();
    mdp.generateStageCostMatrix(r);
    mdp.generateTransitionProbabilityTensor(P, 3, MDP::emptyVec, 3, MDP::emptyVec);

    mdp.inexactPolicyIteration();

    // Check solution
    // 42 (goal) has the highest reward, 17 has the lowest (1-based indexing)
    // only if rank 0 (else some error with stod?)
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
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
    mdp.setOption("-file_probabilities", "../test/100_50_0.1/P.bin");
    mdp.setOption("-file_costs", "../test/100_50_0.1/g.bin");
    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-discount_factor", "0.9");
    mdp.setOption("-file_cost", "ci_cost_2.out");
    mdp.setOption("-file_policy", "ci_policy_2.out");
    mdp.setValuesFromOptions();
    mdp.loadFromBinaryFile();
    mdp.inexactPolicyIteration();

    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
