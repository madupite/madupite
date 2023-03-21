//
// Created by robin on 21.03.23.
//

#include "MDP.h"
#include <random>
#include <iostream>

// constructor
MDP::MDP(unsigned int numStates, unsigned int numActions) :
    numStates_(numStates), numActions_(numActions) {
    transitionMatrix_ = Eigen::Tensor<double, 3>(numActions_, numStates_, numStates_);
    stageCosts_ = Eigen::MatrixXd(numStates_, numActions_);
}

void MDP::generateTransitionMatrix() {
    // fill tensor with random values between 0 and 1
    // normalize each row to sum to 1 to form a row-stochastic matrix for each action
    std::default_random_engine generator(seed_);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < numActions_; i++) {
        for (int j = 0; j < numStates_; j++) {
            double sum = 0;
            for (int k = 0; k < numStates_; k++) {
                double rand = distribution(generator);
                transitionMatrix_(i, j, k) = rand;
                sum += rand;
            }
            for (int k = 0; k < numStates_; k++) {
                transitionMatrix_(i, j, k) /= sum;
            }
        }
    }
}

void MDP::generateStageCosts() {
    // fill matrix with random values between 0 and 1
    std::default_random_engine generator(seed_+314);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < numStates_; i++) {
        for (int j = 0; j < numActions_; j++) {
            stageCosts_(i, j) = distribution(generator);
        }
    }
}

void MDP::outputInfo() {
    std::cout << "MDP has " << numStates_ << " states and " << numActions_ << " actions.\n\n";

    for(int i = 0; i < numActions_; ++i) {
        std::cout << "For action " << i << ":\n----------------\nTransition matrix:\n";
        std::cout << transitionMatrix_.chip(i, 0) << "\n\nStage costs:\n";
        std::cout << stageCosts_.col(i).transpose() << "\n\n\n";
    }
}