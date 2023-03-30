//
// Created by robin on 21.03.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <string>
#include <vector>

class MDP {
public:
    MDP(unsigned int numStates, unsigned int numActions, double discount, const std::string& filename);
    ~MDP();
    void generateTransitionMatrix();
    void generateStageCosts();
    void outputInfo();
    void extractGreedyPolicy_(const Eigen::VectorXd& V, Eigen::VectorXi& policy);
    void valueIteration(Eigen::VectorXd& V0, int iterations, const Eigen::VectorXd& V_opt);
    void policyIteration(const Eigen::VectorXi& policy0);
    void optimisticPolicyIteration(const Eigen::VectorXi& policy0, int iterations);

//private:
    unsigned int numStates_;
    unsigned int numActions_;
    double discount_;
    const int seed_ = 2938;
    const int iterPrint_ = 100; // print output ever iterPrint iterations
    const double tol_ = 10e-13;  // tolerance for convergence
    Eigen::Tensor<double, 3, Eigen::ColMajor> transitionMatrix_;
    Eigen::MatrixXd stageCosts_;

    std::ofstream* logger_;

    std::vector<Eigen::VectorXd> results_VI_;
    std::vector<Eigen::VectorXd> results_PI_;
    std::vector<Eigen::VectorXd> results_OPI_;

    inline void constructCostAndTransitionsFromPolicy_(const Eigen::VectorXi& policy, Eigen::VectorXd& stageCost, Eigen::MatrixXd& P);


};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
