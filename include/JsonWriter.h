//
// Created by robin on 30.05.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_JSONWRITER_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_JSONWRITER_H

// #include <mpi.h>
#include <vector>
// #include <nlohmann/json.hpp>
#include "json.h"
#include <fstream>

class JsonWriter {
public:
    explicit JsonWriter(int rank)
        : rank_(rank)
    {
    }

    void add_solver_run()
    {
        if (rank_ == 0)
            runs.emplace_back();
    }

    void add_data(const std::string& key, const nlohmann::json& value)
    {
        if (rank_ == 0)
            runs.back()[key] = value;
    }

    void add_iteration_data(int pi_iteration, int krylov_iteration, double computation_time, double residual)
    {
        if (rank_ == 0) {
            nlohmann::json tmp;
            tmp["pi_iteration"]     = pi_iteration;
            tmp["ksp_iterations"]   = krylov_iteration;
            tmp["computation_time"] = computation_time;
            tmp["residual"]         = residual;
            runs.back()["iterations"].push_back(tmp);
        }
    }

    void write_to_file(const std::string& filename)
    {
        if (rank_ == 0) {
            nlohmann::json data;
            data["runs"] = runs;
            std::ofstream file(filename);
            file << data.dump(4);
        }
    }

private:
    std::vector<nlohmann::json> runs;
    int                         rank_;
};

#endif // DISTRIBUTED_INEXACT_POLICY_ITERATION_JSONWRITER_H
