#!/bin/bash

# This script runs all the benchmarks in the current directory except the ones in the exclusion list.
# It uses GNU time to measure peak memory (maximum resident size) and wall time for each benchmark.
# The benchmarks in the exclusion list are not run since they require a large amount of memory and result 
# in the github runners getting killed.
# The results are stored in a JSON file $output_file.
#
# Nuances:
# - Measures wall time, but CPU time might be desired in some cases.
# - Build time is excluded by building the benchmarks before running them.
# - Maximum resident size is being used as a surrogate to peak memory usage. 

set -e # Exit on error

# Define the exclude list
exclusion_list=("collatz" "overflow" "sha3-chain")
# JSON file to store results
output_file="benchmark_results.json"

# Write time or memory to JSON file
# Args:
# $1: name of the benchmark
# $2: metric value
# $3: unit of the test
# $4: is_last: boolean to check if it's the last entry
function write_to_json() {
  # Append execution time to JSON file
  printf "  {\n" >>"$output_file"
  printf "        \"name\": \"%s\",\n" "$1" >>"$output_file"
  # printf "        \"unit\": \"s\",\n" >>"$output_file"
  printf "        \"unit\": \"%s\",\n" "$3" >>"$output_file"
  printf "        \"value\": %.4f,\n" "$2" >>"$output_file"
  printf "        \"extra\": \"\"\n" >>"$output_file"
  if [ "$4" = false ]; then
    printf "    },\n" >>"$output_file"
  else
    printf "    }\n" >>"$output_file"
  fi
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

# Determine appropriate `time` command based on OS and availability
if command -v gtime &> /dev/null; then
  TIME_CMD="gtime"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "GNU time (gtime) not found. Install it with 'brew install gnu-time'."
  exit 1
elif /usr/bin/time --version &> /dev/null; then
  TIME_CMD="/usr/bin/time"
else
  echo "Unable to find a compatible 'time' command. Please install GNU time."
  exit 1
fi

# Start creating the JSON structure
printf "[\n" >"$output_file"
echo "-----------------"
for i in "${!test_directories[@]}"; do
  file="${test_directories[$i]}"

  # Build the benchmark
  echo "Building $file"
  cargo build --release -p "$file"

  echo "Running $file"
  # Use `time` to measure execution time and memory.
  # Output the information in the below custom format:
  # wall: 0:02.00 (HH:MM:SS)
  # real: 2.00 s
  # MRS: 1964 KB
  # Capture timing and output into a temp file
  temp_output=$(mktemp)
  if ! $TIME_CMD -f "wall: %E (HH:MM:SS)\nreal: %e s\nMRS: %M KB" cargo run --release -p "$file" 2>&1 | tee "$temp_output"; then
    echo "Error running benchmark for $file. Skipping..."
    continue
  fi
  # Extract 'real' time value using awk
  exec_time=$(awk '/Prover runtime:/ {print $3}' "$temp_output") # in seconds
  # Extract 'MRS' value using awk
  mem_used=$(awk '/MRS:/ {print $2}' "$temp_output") # in KB
  echo "$output" # Print the output for debugging
  # Append execution time to JSON file
  write_to_json "${file}-time" "$exec_time" "s" false
  # Add a comma if it's not the last entry
  is_last=$([ "$i" -lt $((${#test_directories[@]} - 1)) ] && echo false || echo true)
  # Append memory usage to JSON file
  write_to_json "${file}-mem" "$mem_used" "KB" $is_last
  # Clean up
  rm "$temp_output"
done

# Close the JSON structure
printf "]\n" >>"$output_file"
