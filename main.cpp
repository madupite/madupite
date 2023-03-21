#include "MDP.h"

int main() {

    int states = 3, actions = 2;

    MDP mdp(states, actions, 0.95);
    mdp.generateTransitionMatrix();
    mdp.generateStageCosts();
    mdp.outputInfo();
    Eigen::VectorXd V0 = Eigen::VectorXd::Zero(states);
    mdp.valueIteration(V0, 200);

    return 0;
}
