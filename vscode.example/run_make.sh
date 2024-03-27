#!/bin/bash
set -e

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target>"
    exit 1
fi

CMAKE_BUILD=0
PETSC_ENV_FILE=${fileWorkspaceFolder}/petsc-env

if ((CMAKE_BUILD)); then
    cd ${fileWorkspaceFolder}/build
fi
if [ ! -f "Makefile" ]; then
    echo "No Makefile found in $PWD"
    exit 1
fi

source ${PETSC_ENV_FILE}

for t in $TARGET; do
    if (( ! CMAKE_BUILD)) && [ $t == "all" ]; then
        CMD="bear -- make all"
    else
        CMD="make $t"
    fi
    echo "Building target \"$t\" in \"$PWD\" ..."
    echo "# $CMD"
    eval $CMD
    echo "Target \"$t\" done."
    echo
done

COMPILE_COMMANDS="compile_commands.json"
if [ ! -f "$COMPILE_COMMANDS" ]; then
    echo "No $COMPILE_COMMANDS generated in $PWD"
elif ((CMAKE_BUILD)); then
    ln -sfv build/$COMPILE_COMMANDS ../$COMPILE_COMMANDS
fi
