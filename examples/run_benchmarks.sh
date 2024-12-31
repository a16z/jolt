#!/bin/bash

# Define the exclude list
exclude_list=("collatz" "overflow" "sha3-chain")

for file in *; do
  # Check if the file is in the exclude list
  skip=false
  for exclude_file in "${exclude_list[@]}"; do
    if [[ "$file" == "$exclude_file" ]]; then
      skip=true
      echo "skipping excluded example: $file"
      break
    fi
  done

  # If the file is not in the exclude list, process it
  if ! $skip; then
    if [ -d "$file" ]; then
        echo "Running $file"
        cargo run --release -p "$file"
    fi
  fi

done
