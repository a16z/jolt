#!/bin/bash

# This script runs all the benchmarks in the current directory except the ones in the exclusion list.
# Current script measures peak memory and wall time for each benchmark.
#
# TODOs - clarify requirements:
# - Are we measuring the correct the CPU time?
# - Build time is ignored and not included in the CPU time
# - Are we measuring the correct peak memory usage? MRS? Heap? Stack?
# - What about the benchmarks that require a lot of memory? These crash the github runners

# Define the exclude list
# exclusion_list=("collatz" "overflow" "sha3-chain")
exclusion_list=("fibonacci" "multi-function" "collatz" "overflow" "sha3-chain" "stdlib" "alloc" "sha3-ex" "sha2-ex" "sha2-chain" "memory-ops")
# JSON file to store results
output_file="benchmark_results.json"

# Write time or memory to JSON file
# Args:
# $1: name of the benchmark
# $2: metric value
# $3: unit of the test
function write_to_json() {
  # Append execution time to JSON file
  printf "  {\n" >>"$output_file"
  printf "        \"name\": \"%s\",\n" "$1" >>"$output_file"
  # printf "        \"unit\": \"s\",\n" >>"$output_file"
  printf "        \"unit\": \"%s\",\n" "$3" >>"$output_file"
  printf "        \"value\": %.4f,\n" "$2" >>"$output_file"
  printf "        \"extra\": \"\"\n" >>"$output_file"

}


# Initialize an array to hold directories
test_directories=()

# Loop through all items in the current directory
for item in *; do
  # Check if the item is a directory
  if [[ -d "$item" ]]; then
    # Check if the directory is not in the exclusion list
    if [[ ! " ${exclusion_list[@]} " =~ " $item " ]]; then
      # Add the directory to the array
      test_directories+=("$item")
    fi
  fi
done

echo "## List of Tests:"
echo "-----------------"
for dir in "${test_directories[@]}"; do
  echo "$dir"
done

# Start creating the JSON structure
printf "[\n" >"$output_file"
echo "-----------------"
for i in "${!test_directories[@]}"; do
  file="${test_directories[$i]}"

  # Build the benchmark
  echo "Building $file"
  cargo build --release -p "$file"

  echo "Running $file"
  command="cargo run --release -p \"$file\""
  # Use `time` to measure execution time

  # Use `time` to measure execution time and memory.
  # Output the information in the below custom format:
  # wall: 0:02.00 (HH:MM:SS)
  # real: 2.00 s
  # MRS: 1964 KB

  # exec_time=$( (time -p eval "$command") 2>&1 | grep real | awk '{print $2}')
  output=$(/usr/bin/time -f "wall: %E (HH:MM:SS)\nreal: %e s\nMRS: %M KB" eval "$command" 2>&1)
  echo "$output"
  # Extract 'real' time value using awk
  exec_time=$(echo "$output" | awk '/real:/ {print $2}') # in seconds
  echo "$exec_time"
  # Extract 'MRS' value using awk
  mem_used=$(echo "$output" | awk '/MRS:/ {print $2}') # in KB
  echo "$mem_used"

  # Append execution time to JSON file
  write_to_json "${file}-time" "$exec_time" "s"
  printf "    },\n" >>"$output_file"

  write_to_json "${file}-mem" "$mem_used" "KB"
  # Add a comma if it's not the last entry
  if [ "$i" -lt $((${#test_directories[@]} - 1)) ]; then
    printf "    },\n" >>"$output_file"
  else
    printf "    }\n" >>"$output_file"
  fi

done

# Close the JSON structure
printf "]\n" >>"$output_file"
