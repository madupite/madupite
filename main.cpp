#include "MDP.h"
#include <iomanip>
#include <iostream>
#include "Timer.h"

int main() {

    // Bertsekas example
    //MDP mdp(states, actions, 0.9, "out.txt");
    //mdp.transitionMatrix_.setValues({{{0.75, 0.25}, {0.75, 0.25}}, {{0.25, 0.75}, {0.25, 0.75}}});
    //mdp.stageCosts_ << 2, 0.5, 1, 3;

    // generate MDP
    Timer t;
    t.start();
    int states = 600, actions = 200;
    double discount = 0.9;
    MDP mdp(states, actions, discount, "out.txt");
    mdp.generateTransitionMatrix();
    mdp.generateStageCosts();
    mdp.outputInfo();
    t.stop("Time to generate MDP: ");

    // run VI, PI and OPI
    t.start();
    Eigen::VectorXd V0 = Eigen::VectorXd::Ones(states);
    mdp.valueIteration(V0, 10000);
    t.stop("Time to run value iteration: ");

    t.start();
    Eigen::VectorXi policy0 = Eigen::VectorXi::Zero(states);
    mdp.policyIteration(policy0);
    t.stop("Time to run policy iteration: ");

    t.start();
    mdp.optimisticPolicyIteration(policy0, 15);
    t.stop("Time to run optimistic policy iteration: ");

    // calculate errors of different methods
    Eigen::VectorXd V_opt = mdp.results_PI_.back();

    std::ofstream res_PI("res_PI.txt");
    std::ofstream res_VI("res_VI.txt");
    std::ofstream res_OPI("res_OPI.txt");
    res_OPI.precision(std::numeric_limits<double>::max_digits10+1);
    res_PI.precision(std::numeric_limits<double>::max_digits10+1);
    res_VI.precision(std::numeric_limits<double>::max_digits10+1);

    //Eigen::VectorXd PI_error_inf_norm(mdp.results_PI_.size());
    for(int i = 0; i < mdp.results_PI_.size(); ++i) {
        //PI_error_inf_norm(i) =
        res_PI << (mdp.results_PI_[i] - V_opt).lpNorm<Eigen::Infinity>() << "\n";
    }
    //Eigen::VectorXd VI_error_inf_norm(mdp.results_VI_.size());
    for(int i = 0; i < mdp.results_VI_.size(); ++i) {
        //VI_error_inf_norm(i) =
        res_VI << (mdp.results_VI_[i] - V_opt).lpNorm<Eigen::Infinity>() << "\n";
    }
    //Eigen::VectorXd OPI_error_inf_norm(mdp.results_OPI_.size());
    for(int i = 0; i < mdp.results_OPI_.size(); ++i) {
        //OPI_error_inf_norm(i) =
        res_OPI << (mdp.results_OPI_[i] - V_opt).lpNorm<Eigen::Infinity>() << "\n";
    }

    res_PI.close();
    res_VI.close();
    res_OPI.close();

    t.stop("Time to write results: ");

    return 0;
}
