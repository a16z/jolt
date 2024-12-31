#!/bin/bash

for file in *; do
    if [ -d "$file" ]; then
        echo "Running $file"
        cargo run --release -p "$file"
    fi
done
