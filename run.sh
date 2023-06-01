#!/bin/bash

# The name of your program
PROGRAM="./distributed_inexact_policy_iteration"

# The name of the file with arguments
ARGUMENTS_FILE="input.txt"

# Read all lines into an array
mapfile -t lines < "$ARGUMENTS_FILE"

# Join the lines into a single string of arguments
args="${lines[*]}"

# Run the program with the combined arguments

cd cmake-build-debug
echo "starting program in directory: $(pwd)"
$PROGRAM $args