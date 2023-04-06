//
// Created by robin on 28.03.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_TIMER_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_TIMER_H

#include <chrono>
#include <iostream>

class Timer {
public:
    inline void start() {
        t1 = std::chrono::high_resolution_clock::now();
    }
    inline void stop(const std::string& msg) {
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << msg << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
    }

    std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> t1;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::duration<double>> t2;
};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_TIMER_H
