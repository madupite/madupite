//
// Created by robin on 05.07.23.
//

#include <iostream>
#include <boost/math/distributions/binomial.hpp>
#include "InfectiousDiseaseModel.h"

int main(int argc, char **argv) {
#if 0
    int n = 10; // Number of trials
    double p = 0.5; // Probability of success

    // Create a binomial distribution with n trials and probability of success p
    boost::math::binomial_distribution<double> binom(n, p);

    std::cout << "Binomial probabilities for " << n << " trials with probability of success " << p << ":\n";
    for (int k = 0; k <= n; ++k) {
        double prob = boost::math::pdf(binom, k);
        std::cout << "P(X = " << k << ") = " << prob << "\n";
    }
#endif

    PetscInitialize(&argc, &argv, nullptr, nullptr);

    InfectiousDiseaseModel idm;
    idm.setValuesFromOptions();

    idm.generateStageCosts();

    MatView(idm.stageCostMatrix_, PETSC_VIEWER_STDOUT_WORLD);

#if 0
    for(int i = 0; i < idm.numA1_; ++i) {
        std::cout << "r[" << i << "] = " << idm.r_[i] << std::endl;
    }
    for(int i = 0; i < idm.numA2_; ++i) {
        std::cout << "lambda[" << i << "] = " << idm.lambda_[i] << std::endl;
    }
    for(int i = 0; i < idm.numA1_; ++i) {
        std::cout << "cf_a1[" << i << "] = " << idm.cf_a1_[i] << std::endl;
    }
    for(int i = 0; i < idm.numA2_; ++i) {
        std::cout << "cf_a2[" << i << "] = " << idm.cf_a2_[i] << std::endl;
    }
    for(int i = 0; i < idm.numA1_; ++i) {
        std::cout << "cq_a1[" << i << "] = " << idm.cq_a1_[i] << std::endl;
    }
    for(int i = 0; i < idm.numA2_; ++i) {
        std::cout << "cq_a2[" << i << "] = " << idm.cq_a2_[i] << std::endl;
    }
    for(int i = 0; i < 3; ++i) {
        std::cout << "weights[" << i << "] = " << idm.weights_[i] << std::endl;
    }
#endif

    idm.~InfectiousDiseaseModel();
    PetscFinalize();

    return 0;
}