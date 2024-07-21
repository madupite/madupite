#!/bin/bash

# Check if SLURM_JOB_ID is set, if not, use a default value
if [ -z "$SLURM_JOB_ID" ]; then
    SLURMID="test_run"
else
    SLURMID=$SLURM_JOB_ID
fi

cd ../..
source ./euler_module_load.sh
make
cd examples/StrongScaling # TODO: Change to the correct path for each example

lscpu

# Create folder with SLURMID
mkdir -p $SLURMID

# Array of processor counts
processors=(1 2 4 8 12 16 20 24 28 32 36 40 44 48)
# processors=(1 4 16)

# Loop over the processors
for i in "${processors[@]}"
do
    echo "Running with $i processors"
    mpirun -n $i ./bin/StrongScaling -file_stats $SLURMID/stats_$i.json

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Run with $i processors completed successfully"
    else
        echo "Run with $i processors failed"
    fi
done

echo "Strong scaling analysis complete. Results are in the $SLURMID folder."
