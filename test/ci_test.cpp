#include <petsc.h>
#include <iostream>
#include "MDP.h"
#include<random>
#include<utility>
#include<vector>

// MDP: circle world, 50 states, 3 actions (stay, left, right), 
// P(s, 0, s') = 0.9 if s' = s, 0.05 if s' = s-1, 0.05 if s' = s+1
// P(s, 1, s') = 0.1 if s' = s, 0.9 s' = s-1
// P(s, 2, s') = 0.1 if s' = s, 0.9 s' = s+1
// goal state: 42
// cost function: g(s, a) = 1 if s != 42, 0 if s = 42


double r(int s, int a) {
    // return (s == 41) ? 0.0 : 10.0;
    // return std::abs(s - 41);
    // reward of 1 for goal state, 0 otherwise
    return (s == 41) ? 1.0 : 0.0;
}

std::pair<std::vector<double>, std::vector<int>> P(int s, int a) {
    std::vector<double> values;
    std::vector<int> indices;
    switch(a) {
        case 0: // stay
            values = {0.9, 0.05, 0.05};
            indices = {s, (s-1+50) % 50, (s+1) % 50};
            break;
        case 1: // left
            values = {0.1, 0.9};
            indices = {s, (s-1+50) % 50};
            break;
        case 2: // right
            values = {0.1, 0.9};
            indices = {s, (s+1) % 50};
            break;
    }
    return {values, indices};
}


int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR);
    
    MDP mdp;
    mdp.setOption("-mode", "MAXREWARD");
    mdp.setOption("-discount_factor", "0.99999");
    mdp.setOption("-max_iter_pi", "200");
    mdp.setOption("-max_iter_ksp", "1000");
    mdp.setOption("-num_pi_runs", "1");
    mdp.setOption("-rtol_ksp", "1e-4");
    mdp.setOption("-atol_pi", "1e-8");
    mdp.setOption("-num_states", "50");
    mdp.setOption("-num_actions", "3");
    mdp.setOption("-file_stats", "ci_stats.json");
    mdp.setOption("-file_cost", "ci_cost.out");
    mdp.setOption("-file_policy", "ci_policy.out");
    mdp.setOption("-ksp_type", "gmres");
    mdp.setValuesFromOptions();
    mdp.generateStageCostMatrix(r);
    std::vector<int> empty = {};
    mdp.generateTransitionProbabilityTensor(P, 3, empty, 3, empty);

    mdp.inexactPolicyIteration();

    // Optimistic policy iteration
    mdp.setOption("-ksp_type", "richardson");
    mdp.setOption("-max_iter_ksp", "100");
    mdp.setOption("-rtol_ksp", "1e-40");
    mdp.setOption("-file_policy", "ci_policy_opi.out");
    mdp.setOption("-file_cost", "ci_cost_opi.out");
    mdp.setValuesFromOptions();
    mdp.inexactPolicyIteration();


    mdp.~MDP();

    // Finalize PETSc
    PetscFinalize();
    return 0;
}
