#include "MDP.h"

int main() {


    // MDP mdp(states, actions, 0.95);
    // mdp.generateTransitionMatrix();
    // mdp.generateStageCosts();

    int states = 2, actions = 2;
    MDP mdp(states, actions, 0.9);
    mdp.transitionMatrix_.setValues({{{0.75, 0.25}, {0.75, 0.25}}, {{0.25, 0.75}, {0.25, 0.75}}});
    mdp.stageCosts_ << 2, 0.5, 1, 3;

    mdp.outputInfo();
    Eigen::VectorXd V0 = Eigen::VectorXd::Ones(states);
    mdp.valueIteration(V0, 1000);
    Eigen::VectorXi policy0 = Eigen::VectorXi::Zero(states);
    policy0 << 0, 1;
    mdp.policyIteration(policy0);

    return 0;
}
