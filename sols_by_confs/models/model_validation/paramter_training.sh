#!/usr/bin/env bash

if [ -z "$1"]; then
    echo "Error: No script provided."
    echo "Usage: ./parameter_training.sh <python file>"
    exit 1
fi

args_list=(
    "bc"
    "rc"
    "brc"
    "bcf"
    "rcf"
    "brcf"
)

script="$1"

for args in "${args_list[@]}"; do
    python $script $args
    # echo "Running: python $script $args"
    # timestamp=$(date+"%Y%m%d_%H%M")
    # python $script $args > "logs/run_${timestamp}.log" 2>&1
done