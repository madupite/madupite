cmake_minimum_required(VERSION 3.24)
project(distributed_inexact_policy_iteration)

set(CMAKE_CXX_STANDARD 17)

add_executable(distributed_inexact_policy_iteration main.cpp MDP.cpp)
