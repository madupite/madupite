//
// Created by robin on 30.05.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_JSONWRITER_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_JSONWRITER_H

#include <mpi.h>
#include <vector>
#include <nlohmann/json.hpp>
#include <fstream>

class JsonWriter {
public:
    JsonWriter(int rank, int size) : rank_(rank), size_(size) {}

    void add_data(int pi_iteration, int krylov_iteration, double computation_time, double residual) {
        nlohmann::json data;
        data["pi_iteration"] = pi_iteration;
        data["ksp_iterations"] = krylov_iteration;
        data["computation_time"] = computation_time;
        data["residual"] = residual;

        all_data.push_back(data);
    }

    void write_to_file(const std::string& filename) {
        if (rank_ == 0) {
            nlohmann::json output;
            output["data"] = all_data;

            std::ofstream file(filename);
            file << output.dump(4);
        }
    }

private:
    std::vector<nlohmann::json> all_data;
    int rank_;
    int size_;
};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_JSONWRITER_H
