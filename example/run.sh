#!/bin/bash

cd ../build
make

nproc=4

mpirun -n $nproc ./distributed_inexact_policy_iteration \
-pc_type none \
-mode MINCOST \
-discountFactor 0.9 \
-maxIter_PI 20 \
-maxIter_KSP 1000 \
-numPIRuns 1 \
-rtol_KSP 1e-4 \
-atol_PI 1e-10 \
-file_P ../example/100_50_0.1/P.bin \
-file_g ../example/100_50_0.1/g.bin \
-file_stats stats.json \
-file_policy policy.out \
-file_cost cost.out

cd ../example # Go back to example directory