#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

#include "MDP.h"

class Pendulum {
public:
    Pendulum(int num_x = 401, int num_xd = 401, int num_a = 201, double max_x = 2 * std::numbers::pi, double max_xd = 10.0, double max_a = 3.0,
        double r = 1.0, double q = 2.0, double dt = 0.01, double g = 9.81, double l = 1.0, double m = 1.0)
        : NUM_X(num_x)
        , NUM_XD(num_xd)
        , NUM_A(num_a)
        , MAX_X(max_x)
        , MAX_XD(max_xd)
        , MAX_A(max_a)
        , R(r)
        , Q(q)
        , DT(dt)
        , G(g)
        , L(l)
        , M(m)
        , NUM_STATES(NUM_X * NUM_XD)
        , NUM_ACTIONS(NUM_A)
    {

        init_vals();
    }

    std::function<std::pair<std::vector<double>, std::vector<int>>(int, int)> P_func() const
    {
        return [this](int s, int a) { return this->P(s, a); };
    }

    std::function<double(int, int)> g_func() const
    {
        return [this](int s, int a) { return this->g(s, a); };
    }

    const std::vector<double>& get_x_vals() const { return x_vals; }
    const std::vector<double>& get_xd_vals() const { return xd_vals; }
    const std::vector<double>& get_a_vals() const { return a_vals; }
    const int                  get_num_states() const { return NUM_STATES; }
    const int                  get_num_actions() const { return NUM_ACTIONS; }

private:
    int x2s(int x, int xd) const { return x * NUM_XD + xd; }

    std::pair<int, int> s2x(int s) const { return { s / NUM_XD, s % NUM_XD }; }

    std::pair<std::vector<std::pair<int, int>>, std::vector<double>> interpolate(
        double x, double y, const std::vector<double>& grid_x, const std::vector<double>& grid_y) const
    {
        int x_i = std::upper_bound(grid_x.begin(), grid_x.end(), x) - grid_x.begin();
        int y_i = std::upper_bound(grid_y.begin(), grid_y.end(), y) - grid_y.begin();

        x_i = std::clamp(x_i, 1, static_cast<int>(grid_x.size()) - 1);
        y_i = std::clamp(y_i, 1, static_cast<int>(grid_y.size()) - 1);

        double xl_v = grid_x[x_i - 1], xr_v = grid_x[x_i];
        double yl_v = grid_y[y_i - 1], yr_v = grid_y[y_i];

        double wx1 = (x - xl_v) / (xr_v - xl_v);
        double wy1 = (y - yl_v) / (yr_v - yl_v);
        double wx0 = 1 - wx1, wy0 = 1 - wy1;

        std::vector<std::pair<int, int>> indices = { { x_i - 1, y_i - 1 }, { x_i, y_i - 1 }, { x_i - 1, y_i }, { x_i, y_i } };
        std::vector<double>              weights = { wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1 };

        return { indices, weights };
    }

    std::pair<double, double> step(int x_t_i, int xd_t_i, int a) const
    {
        double x_t_v  = x_vals[x_t_i];
        double xd_t_v = xd_vals[xd_t_i];
        double a_t_v  = a_vals[a];

        double x_tpp_v  = circular_mod(x_t_v + xd_t_v * DT, MAX_X); // periodic boundary!
        double xdd_t_v  = -G / L * std::sin(x_t_v) + a_t_v / (M * L * L);
        double xd_tpp_v = std::clamp(xd_t_v + xdd_t_v * DT, -MAX_XD, MAX_XD);

        return { x_tpp_v, xd_tpp_v };
    }

    double stage_costs(int x_i, int xd_i, int a) const
    {
        return Q * (std::pow(x_vals[x_i] - std::numbers::pi, 2) + std::pow(xd_vals[xd_i], 2)) + R * std::pow(a_vals[a], 2);
    }

    std::pair<std::vector<double>, std::vector<int>> P(int s, int a) const
    {
        auto [x_t_i, xd_t_i]     = s2x(s);
        auto [x_tpp_v, xd_tpp_v] = step(x_t_i, xd_t_i, a);

        auto [indices, weights] = interpolate(x_tpp_v, xd_tpp_v, x_vals, xd_vals);

        std::vector<int> next_states;
        for (const auto& [x_i, xd_i] : indices) {
            next_states.push_back(x2s(x_i, xd_i));
        }

        return { weights, next_states };
    }

    double g(int s, int a) const
    {
        auto [x_t_i, xd_t_i] = s2x(s);
        return stage_costs(x_t_i, xd_t_i, a);
    }

    double circular_mod(double a, double b) const
    {
        double result = std::fmod(a, b);
        return result >= 0 ? result : result + b;
    }

    void init_vals()
    {
        x_vals.resize(NUM_X);
        xd_vals.resize(NUM_XD);
        a_vals.resize(NUM_A);

        for (int i = 0; i < NUM_X; ++i) {
            x_vals[i] = i * MAX_X / (NUM_X - 1);
        }
        for (int i = 0; i < NUM_XD; ++i) {
            xd_vals[i] = -MAX_XD + i * 2 * MAX_XD / (NUM_XD - 1);
        }
        for (int i = 0; i < NUM_A; ++i) {
            a_vals[i] = -MAX_A + i * 2 * MAX_A / (NUM_A - 1);
        }
    }

    const int    NUM_X, NUM_XD, NUM_A;
    const double MAX_X, MAX_XD, MAX_A;
    const double R, Q, DT, G, L, M;
    const int    NUM_STATES, NUM_ACTIONS;

    std::vector<double> x_vals, xd_vals, a_vals;
};

int main(int argc, char** argv)
{
    // Initialize MPI, PETSc and Madupite, passing command line arguments.
    auto madupite = Madupite::initialize(&argc, &argv);

    Pendulum pendulum(401, 401, 201);
    MDP      mdp(madupite);

    mdp.setOption("-mode", "MINCOST");
    mdp.setOption("-max_iter_pi", "200");
    mdp.setOption("-max_iter_ksp", "2000");
    mdp.setOption("-alpha", "1e-2");
    mdp.setOption("-ksp_type", "tfqmr");
    mdp.setOption("-atol_pi", "1e-9");
    mdp.setOption("-num_states", std::to_string(pendulum.get_num_states()).c_str());
    mdp.setOption("-num_actions", std::to_string(pendulum.get_num_actions()).c_str());
    mdp.setOption("-discount_factor", "0.999");

    // mdp.setOption("-file_stats", "pend_stats.json");
    // mdp.setOption("-file_policy", "pend_policy.out");
    // mdp.setOption("-file_cost", "pend_cost.out");

    mdp.setSourceTransitionProbabilityTensor(pendulum.P_func(), 4, {}, 4, {});
    mdp.setSourceStageCostMatrix(pendulum.g_func());

    mdp.solve();
}
