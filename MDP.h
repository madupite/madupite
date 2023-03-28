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
    void valueIteration(Eigen::VectorXd& V0, int iterations);
    void policyIteration(const Eigen::VectorXi& policy0);
    void optimisticPolicyIteration(const Eigen::VectorXi& policy0, int iterations);

//private:
    unsigned int numStates_;
    unsigned int numActions_;
    double discount_;
    int seed_ = 2938;
    int iterPrint_ = 20; // print output ever iterPrint iterations
    Eigen::Tensor<double, 3, Eigen::ColMajor> transitionMatrix_;
    Eigen::MatrixXd stageCosts_;

    std::ofstream* logger_;

    std::vector<Eigen::VectorXd> results_VI_;
    std::vector<Eigen::VectorXd> results_PI_;
    std::vector<Eigen::VectorXd> results_OPI_;

    inline void constructCostAndTransitionsFromPolicy_(const Eigen::VectorXi& policy, Eigen::VectorXd& stageCost, Eigen::MatrixXd& P);


};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
