#!/usr/bin/env bash
# pin_numa.sh — Generic NUMA pinning wrapper for any command or binary
# Usage:
#   ./pin_numa.sh <command> [args...]                     # defaults: PIN_NODES=0, PIN_POLICY=bind
#   ./pin_numa.sh ls -la                                  # run ls command with args
#   ./pin_numa.sh cargo run -p fibonacci --release        # run cargo command
#   ./pin_numa.sh ./my_binary input.txt                   # run custom binary
#   PIN_NODES=0-3 ./pin_numa.sh python3 script.py        # use nodes 0-3
#   PIN_NODES=0,2 PIN_POLICY=interleave ./pin_numa.sh ./benchmark
# 
# Environment variables:
#   - PIN_NODES: e.g. "0" (default), "0,1", "0-5", "0,2,4"
#   - PIN_POLICY: "bind" (default) or "interleave"
#   - SHOW_TIMING: set to 1 to show basic timing information

set -euo pipefail

# Parse command line arguments
if [[ $# -eq 0 ]]; then
  echo "Error: No command specified"
  echo "Usage: $0 <command> [args...]"
  echo "Example: $0 ls -la"
  echo "Example: $0 cargo run -p fibonacci --release"
  exit 1
fi

CMD=( "$@" )

: "${PIN_NODES:=0}"
: "${PIN_POLICY:=bind}"
: "${SHOW_TIMING:=0}"

echo "Running command '${CMD[*]}' with NUMA pinning"
echo "PIN_NODES=${PIN_NODES} | PIN_POLICY=${PIN_POLICY}"

# Check required tools
command -v lscpu >/dev/null 2>&1 || { 
  echo "Installing util-linux (for lscpu)..." 
  sudo apt-get update && sudo apt-get install -y util-linux
}
command -v numactl >/dev/null 2>&1 || { 
  echo "Installing numactl..." 
  sudo apt-get update && sudo apt-get install -y numactl
}

# Expand node ranges like "0-2,4" -> "0,1,2,4"
_expand_nodes() {
  echo "$1" | awk -F, '
    function expand_range(r){
      split(r,a,"-")
      if(length(a)==2){
        for(i=a[1];i<=a[2];i++) printf (out++?",":"") i
      } else {
        printf (out++?",":"") r
      }
    }
    {
      out=0
      n=split($0,b,",")
      for(i=1;i<=n;i++) expand_range(b[i])
      print ""
    }'
}

NODES_EXPANDED=$(_expand_nodes "$PIN_NODES")

# Build CPU list from those nodes
CPU_LIST=$(lscpu -e=CPU,NODE | awk -v nodes="$NODES_EXPANDED" '
  BEGIN{
    split(nodes,ok,",")
    for(i in ok) allow[ok[i]]=1
  }
  NR>1 && ($2 in allow){
    print $1
  }' | paste -sd, -)

if [[ -z "$CPU_LIST" ]]; then
  echo "Error: No CPUs found for nodes ${PIN_NODES}"
  echo "Available NUMA topology:"
  lscpu | grep -E "(NUMA node|CPU\(s\))"
  exit 1
fi

# Configure numactl options
if [[ "$PIN_POLICY" == "interleave" ]]; then
  NUMA_MEM_OPTS=(--interleave="${PIN_NODES}")
  NUMA_CPU_OPTS=(--cpunodebind="${PIN_NODES}")
else
  NUMA_MEM_OPTS=(--membind="${PIN_NODES}")
  NUMA_CPU_OPTS=(--cpunodebind="${PIN_NODES}")
fi

echo "CPU list: $CPU_LIST"
echo "NUMA memory policy: ${NUMA_MEM_OPTS[*]}"
echo "NUMA CPU policy: ${NUMA_CPU_OPTS[*]}"
echo ""

# Run with or without timing
if [[ "$SHOW_TIMING" == "1" ]]; then
  command -v /usr/bin/time >/dev/null 2>&1 || { 
    echo "Installing time for timing measurement..." 
    sudo apt-get update && sudo apt-get install -y time
  }
  
  echo "Running with timing measurement..."
  # Capture time output to convert KB to GB
  TIME_OUTPUT_FILE=$(mktemp)
  /usr/bin/time -f "%E,%M,%P" -o "$TIME_OUTPUT_FILE" \
    numactl "${NUMA_CPU_OPTS[@]}" "${NUMA_MEM_OPTS[@]}" \
    taskset -c "$CPU_LIST" \
    "${CMD[@]}"
  
  # Parse and display with GB conversion
  if [[ -f "$TIME_OUTPUT_FILE" ]]; then
    IFS=, read -r elapsed_time max_rss_kb cpu_pct < "$TIME_OUTPUT_FILE"
    max_rss_gb=$(python3 -c "print(f'{float('$max_rss_kb') / 1024 / 1024:.2f}')")
    echo "Elapsed: ${elapsed_time} | Max RSS: ${max_rss_gb} GB | CPU: ${cpu_pct}"
    rm -f "$TIME_OUTPUT_FILE"
  fi
else
  echo "Running command..."
  numactl "${NUMA_CPU_OPTS[@]}" "${NUMA_MEM_OPTS[@]}" \
    taskset -c "$CPU_LIST" \
    "${CMD[@]}"
fi

echo ""
echo "Command completed successfully!"
