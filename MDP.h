//
// Created by robin on 21.03.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <unsupported/Eigen/CXX11/Tensor>

class MDP {
public:
    MDP(unsigned int numStates, unsigned int numActions);
    void generateTransitionMatrix();
    void generateStageCosts();
    void outputInfo();

private:
    unsigned int numStates_;
    unsigned int numActions_;
    int seed_ = 42;
    Eigen::Tensor<double, 3, Eigen::ColMajor> transitionMatrix_;
    Eigen::MatrixXd stageCosts_;

};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
