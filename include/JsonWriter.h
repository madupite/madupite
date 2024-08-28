#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include <json.h>
#include <mpi.h>
#include <petscsystypes.h>

#include <utils.h>

class JsonWriter {
    std::vector<nlohmann::json> runs;
    PetscMPIInt                 rank_;

public:
    explicit JsonWriter(MPI_Comm comm) { MPI_Comm_rank(comm, &rank_); }

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

    void write_to_file(const std::string& filename, bool overwrite = false)
    {
        if (filename.empty())
            return;

        if (rank_ == 0) {
            std::string safe_filename = get_safe_filename(filename, overwrite);

            nlohmann::json data;
            data["runs"] = runs.back();

            std::ofstream file(safe_filename);
            if (file.is_open()) {
                file << data.dump(4);
                file.close();
            } else {
                throw std::runtime_error("Failed to open file: " + safe_filename);
            }
        }
    }
};
