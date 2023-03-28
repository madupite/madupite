//
// Created by robin on 21.03.23.
//

#include "MDP.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <cassert>
#include "Timer.h"

// constructor
MDP::MDP(unsigned int numStates, unsigned int numActions, double discount, const std::string& filename) :
    numStates_(numStates), numActions_(numActions), discount_(discount) {
    transitionMatrix_ = Eigen::Tensor<double, 3>(numActions_, numStates_, numStates_);
    stageCosts_ = Eigen::MatrixXd(numStates_, numActions_);
    logger_ = new std::ofstream(filename);
}

// destructor
MDP::~MDP() {
    logger_->close();
    delete logger_;
}

void MDP::generateTransitionMatrix() {
    // fill tensor with random values between 0 and 1
    // normalize each row to sum to 1 to form a row-stochastic matrix for each action
    std::default_random_engine generator(seed_);
    *logger_ << "Generating transition matrix with seed " << seed_ << "\n";
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
    // fill matrix with random values between a and b
    double a = 1.0, b = 10.0;
    int seed = seed_ + 314;
    *logger_ << "Generating stage costs with seed " << seed << "\n";
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(a, b);
    for (int i = 0; i < numStates_; i++) {
        for (int j = 0; j < numActions_; j++) {
            stageCosts_(i, j) = distribution(generator);
        }
    }
}

void MDP::outputInfo() {
    *logger_ << "MDP has " << numStates_ << " states and " << numActions_ << " actions.\n\n";
    if(numStates_ < 10 && numActions_ < 10) {
        for (int i = 0; i < numActions_; ++i) {
            *logger_ << "For action " << i << ":\n----------------\nTransition matrix:\n";
            *logger_ << transitionMatrix_.chip(i, 0) << "\n\nStage costs:\n";
            *logger_ << stageCosts_.col(i).transpose() << "\n\n\n";
        }
    }
}

void MDP::valueIteration(Eigen::VectorXd& V0, int iterations) {
    // V0 must no be all zeros! (-> loop won't start)

    double tol = 10e-13;
    Timer t;

    Eigen::VectorXd V = V0;
    Eigen::VectorXd V_old = Eigen::VectorXd::Zero(numStates_);
    Eigen::MatrixXd costs(numStates_, numActions_);
    Eigen::VectorXd policy(numStates_);

    *logger_ << "Value iteration:\n";
    for(int i = 0; i < iterations && (V - V_old).lpNorm<Eigen::Infinity>() > tol; ++i) {
        // compute costs for each action
        if(i % iterPrint_ == 0) *logger_ << "Iteration " << i << ":\n";
        t.start();
        for(int j = 0; j < numActions_; ++j) {
            Eigen::Tensor<double, 2> tmp = transitionMatrix_.chip(j, 0);
            Eigen::MatrixXd P = Eigen::Map<Eigen::MatrixXd>(tmp.data(), numStates_, numStates_);
            costs.col(j) = stageCosts_.col(j) + discount_ * P * V;
        }
        t.stop("Time for computing costs: ");

        // find optimal action and cost for each state (greedy policy)
        t.start();
        V_old = V;
        for(int j = 0; j < numStates_; ++j) {
            int minIndex;
            V(j) = costs.row(j).minCoeff(&minIndex);
            policy(j) = minIndex; // optimal action
            if(i % iterPrint_ == 0) *logger_ << std::setw(12) << V(j) << " |" << std::setw(4) << policy(j) << "\n";
        }
        results_VI_.push_back(V);
        t.stop("Time for extracting greedy policy: ");
    }
    *logger_ << "\n\n";
}

inline void MDP::constructCostAndTransitionsFromPolicy_(const Eigen::VectorXi& policy, Eigen::VectorXd& stageCost, Eigen::MatrixXd& P) {
    // project costs and probabilities from tensor to matrix according to action for each state
    for(int stateInd = 0; stateInd < numStates_; ++stateInd) {
        int action = policy(stateInd);
        stageCost(stateInd) = stageCosts_(stateInd, action);
        Eigen::Tensor<double, 1> tmp = transitionMatrix_.chip(action, 0).chip(stateInd, 0);
        P.row(stateInd) = Eigen::Map<Eigen::MatrixXd>(tmp.data(), 1, numStates_);
    }
}

void MDP::policyIteration(const Eigen::VectorXi& policy0) {
    Eigen::VectorXi policy = policy0;                                       // current policy
    Eigen::VectorXi policy_old(numStates_);                                 // policy from previous iteration
    Eigen::MatrixXd costs = Eigen::MatrixXd::Zero(numStates_, numActions_); // stores $g + \gamma P V$ for each action
    Eigen::VectorXd V = Eigen::VectorXd::Zero(numStates_);                  // solution of policy evaluation
    Eigen::VectorXd stageCost = Eigen::VectorXd::Zero(numStates_);          // stage costs for current policy
    Eigen::MatrixXd P(numStates_, numStates_);                              // transition matrix for current policy

    Timer t;

    *logger_ << "Policy iteration:\n";
    for(int i = 0; ; ++i) {
        if (i % iterPrint_ == 0) *logger_ << "Iteration " << i << ":\n";

        // policy evaluation
        t.start();
        constructCostAndTransitionsFromPolicy_(policy, stageCost, P);
        t.stop("Time for constructing cost and transition matrix: ");
        t.start();
        Eigen::MatrixXd J = Eigen::MatrixXd::Identity(numStates_, numStates_) - discount_ * P;
        V = J.lu().solve(stageCost);
        results_PI_.push_back(V);
        t.stop("Time for solving policy evaluation LSE: ");

        // policy improvement
        t.start();
        for(int actionInd = 0; actionInd < numActions_; ++actionInd) {
            constructCostAndTransitionsFromPolicy_(policy, stageCost, P);
            costs.col(actionInd) = stageCosts_.col(actionInd) + discount_ * P * V;
        }
        // find optimal action for each state (obtain greedy policy)
        policy_old = policy;
        for(int j = 0; j < numStates_; ++j) {
            int minIndex;
            costs.row(j).minCoeff(&minIndex);
            policy(j) = minIndex; // optimal action
            if(i % iterPrint_ == 0) *logger_ << std::setw(12) << V(j) << " |" << std::setw(4) << policy_old(j) << "\n";
        }
        *logger_ << "\n\n";
        t.stop("Time for policy improvement: ");

        // converged if policy does not change
        if(policy == policy_old) break;
    }
}

void MDP::optimisticPolicyIteration(const Eigen::VectorXi &policy0, int iterations) {
    Eigen::VectorXi policy = policy0;
    Eigen::VectorXi policy_old(numStates_);
    Eigen::MatrixXd costs = Eigen::MatrixXd::Zero(numStates_, numActions_);
    Eigen::VectorXd V = Eigen::VectorXd::Zero(numStates_);
    Eigen::VectorXd stageCost = Eigen::VectorXd::Zero(numStates_);
    Eigen::MatrixXd P(numStates_, numStates_);

    Timer t;

    *logger_ << "Optimistic policy iteration (K = "<< iterations << " inner iterations):\n";
    for(int i = 0; ; ++i) {
        if(i % iterPrint_ == 0) *logger_ << "Iteration " << i << ":\n";

        // policy evaluation (value iteration)
        t.start();
        constructCostAndTransitionsFromPolicy_(policy, stageCost, P);
        for(int j = 0; j < iterations; ++j) { // perform as many value iterations as specified to approximate V
            V = stageCost + discount_ * P * V;
        }
        results_OPI_.push_back(V);
        t.stop("Time for approximating V using VI: ");

        // policy improvement
        t.start();
        for(int actionInd = 0; actionInd < numActions_; ++actionInd) {
            constructCostAndTransitionsFromPolicy_(policy, stageCost, P);
            costs.col(actionInd) = stageCosts_.col(actionInd) + discount_ * P * V;
        }
        // find optimal action for each state (obtain greedy policy)
        policy_old = policy;
        for(int j = 0; j < numStates_; ++j) {
            int minIndex;
            costs.row(j).minCoeff(&minIndex);
            policy(j) = minIndex; // optimal action
            if(i % iterPrint_ == 0) *logger_ << std::setw(12) << V(j) << " |" << std::setw(4) << policy_old(j) << "\n";
        }
        *logger_ << "\n\n";
        t.stop("Time for policy improvement: ");

        // converged if policy does not change
        if(policy == policy_old) break;
    }
}


