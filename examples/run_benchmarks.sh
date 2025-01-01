#!/bin/bash

# Define the exclude list
# exclusion_list=("collatz" "overflow" "sha3-chain")
exclusion_list=("multi-function" "collatz" "overflow" "sha3-chain" "stdlib" "alloc" "sha3-ex" "sha2-ex" "sha2-chain" "memory-ops")


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


# JSON file to store results
output_file="benchmark_results.json"

# Start creating the JSON structure
printf "[\n" > "$output_file"
echo "-----------------"
for i in "${!test_directories[@]}"; do
  file="${test_directories[$i]}"

  echo "Running $file"
  command="cargo run --release -p \"$file\""
  # Use `time` to measure execution time

  # Measure the execution time using `time`
  exec_time=$( (time -p eval "$command") 2>&1 | grep real | awk '{print $2}' )

  # Append result to JSON file
  printf "  {\n" >> "$output_file"
  printf "        \"name\": \"%s\",\n" "$file" >> "$output_file"
  printf "        \"unit\": \"s\",\n" >> "$output_file"
  printf "        \"value\": %.4f,\n" "$exec_time" >> "$output_file"
  printf "        \"extra\": \"\"\n" >> "$output_file"

  # Add a comma if it's not the last entry
  if [ $i -lt $((${#test_directories[@]} - 1)) ]; then
      printf "    },\n" >> "$output_file"
  else
      printf "    }\n" >> "$output_file"
  fi

done

# Close the JSON structure
printf "]\n" >> "$output_file"