#!/bin/bash

# Script to generate SRS parameters for Dory
# Usage: ./generate_srs.sh <max_log_n>
# Example: ./generate_srs.sh 18

if [ $# -ne 1 ]; then
    echo "Usage: $0 <max_log_n>"
    echo "Example: $0 18"
    exit 1
fi

max_log_n=$1

# Validate input
if ! [[ "$max_log_n" =~ ^[0-9]+$ ]]; then
    echo "Error: max_log_n must be a positive integer"
    exit 1
fi

if [ "$max_log_n" -gt 25 ]; then
    echo "Warning: max_log_n > 25 may take a very long time and use lots of memory"
    echo "Polynomial size will be 2^$max_log_n = $((2**max_log_n)) elements"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "Generating SRS for max_log_n = $max_log_n..."
echo "Running: cargo run --release --bin generate_srs $max_log_n"

cargo run --release --bin generate_srs "$max_log_n"