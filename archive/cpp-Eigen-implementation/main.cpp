#include "MDP.h"
#include <iostream>
#include "Timer.h"

int main() {

    // Bertsekas example
    //MDP mdp(2, 2, 0.9, "out.txt");
    //mdp.transitionMatrix_.setValues({{{0.75, 0.25}, {0.75, 0.25}}, {{0.25, 0.75}, {0.25, 0.75}}});
    //mdp.stageCosts_ << 2, 0.5, 1, 3;

    // generate MDP
    Timer t;
    t.start();
    int states = 31, actions = 20;
    double discount = 0.7;
    MDP mdp(states, actions, discount, "out.txt");
    mdp.generateTransitionMatrix();
    mdp.generateStageCosts();
    mdp.outputInfo();
    t.stop("Time to generate MDP: ");

    // initial value function
    Eigen::VectorXd V0 = Eigen::VectorXd::Ones(states);
    // generate greedy policy from V0
    Eigen::VectorXi policy0(states);
    mdp.extractGreedyPolicy_(V0, policy0);

    // s.t. plots start with same cost
    mdp.results_PI_.push_back(V0);
    mdp.results_VI_.push_back(V0);
    mdp.results_OPI_.push_back(V0);

    // run VI, PI and OPI
    t.start();
    mdp.policyIteration(policy0);
    t.stop("Time to run policy iteration: ");
    Eigen::VectorXd V_opt = mdp.results_PI_.back();

    t.start();
    mdp.valueIteration(V0, 10000, V_opt);
    t.stop("Time to run value iteration: ");

    t.start();
    mdp.optimisticPolicyIteration(policy0, 15);
    t.stop("Time to run optimistic policy iteration: ");

    // calculate errors and output of different methods
    std::ofstream res_PI("res_PI.txt");
    std::ofstream res_VI("res_VI.txt");
    std::ofstream res_OPI("res_OPI.txt");
    res_OPI.precision(std::numeric_limits<double>::max_digits10+1);
    res_PI.precision(std::numeric_limits<double>::max_digits10+1);
    res_VI.precision(std::numeric_limits<double>::max_digits10+1);

    for(auto& i : mdp.results_PI_) {
        res_PI << (i - V_opt).lpNorm<Eigen::Infinity>() << "\n";
    }
    for(auto& i : mdp.results_VI_) {
        res_VI << (i - V_opt).lpNorm<Eigen::Infinity>() << "\n";
    }
    for(auto& i : mdp.results_OPI_) {
        res_OPI << (i - V_opt).lpNorm<Eigen::Infinity>() << "\n";
    }

    res_PI.close();
    res_VI.close();
    res_OPI.close();

    t.stop("Time to write results: ");

    return 0;
}
